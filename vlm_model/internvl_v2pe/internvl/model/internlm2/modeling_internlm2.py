# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on transformers/src/transformers/models/llama/modeling_llama.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch InternLM2 model."""
import math
import queue
import threading
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from einops import rearrange
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast,
                                           SequenceClassifierOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging,
                                replace_return_docstrings)
from internvl.train.compress_seq_trainer import chunk_with_boundaries
try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

from .configuration_internlm2 import InternLM2Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = 'InternLM2Config'
FINAL_SIZE = 100
flash_attn_func, flash_attn_varlen_func = None, None
pad_input, index_first_axis, unpad_input = None, None, None
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis as _index_first_axis
    from flash_attn.bert_padding import pad_input as _pad_input
    from flash_attn.bert_padding import unpad_input as _unpad_input

    flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
    pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    has_flash_attn = True
except:
    has_flash_attn = False
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, n_prime):

        super(AttentionPooling, self).__init__()
        self.query = nn.Linear(input_dim, n_prime)

    def forward(self, x):

        attention_scores = self.query(x)

        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, n_prime)

        output = torch.einsum('bni,bnd->bid', attention_weights, x)

        return output
class TopKPooling(nn.Module):
    def __init__(self, input_dim, n_prime):

        super(TopKPooling, self).__init__()
        self.query = nn.Linear(input_dim, 1)
        self.n_prime = n_prime

    def forward(self, x):
        attention_scores = self.query(x).squeeze(-1)  # (batch_size, seq_len)

        topk_scores, topk_indices = torch.topk(attention_scores, self.n_prime, dim=1)  # (batch_size, n_prime)

        batch_indices = torch.arange(x.size(0)).unsqueeze(-1).expand(-1, self.n_prime)  # (batch_size, n_prime)
        selected_x = x[batch_indices, topk_indices]  # (batch_size, n_prime, input_dim)

        attention_weights = F.softmax(topk_scores, dim=1).unsqueeze(-1)  # (batch_size, n_prime, 1)

        output = selected_x * attention_weights  # (batch_size, n_prime, input_dim)

        return output
class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
class Sigmoid(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 0.0,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x1,x2):
        return x1*torch.sigmoid(self.gate)+x2*(1-torch.sigmoid(self.gate))
def _import_flash_attn():
    global flash_attn_func, flash_attn_varlen_func
    global pad_input, index_first_axis, unpad_input
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
        from flash_attn import \
            flash_attn_varlen_func as _flash_attn_varlen_func
        from flash_attn.bert_padding import \
            index_first_axis as _index_first_axis
        from flash_attn.bert_padding import pad_input as _pad_input
        from flash_attn.bert_padding import unpad_input as _unpad_input
        flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
        pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    except ImportError:
        raise ImportError('flash_attn is not installed.')


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->InternLM2
class InternLM2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        InternLM2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from functools import partial

    from apex.normalization import FusedRMSNorm
    InternLM2RMSNorm = partial(FusedRMSNorm, eps=1e-6)   # noqa
    print('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternLM2RMSNorm')
except ImportError:
    # using the normal LlamaRMSNorm
    pass
except Exception:
    print('discovered apex but it failed to load, falling back to InternLM2RMSNorm')
    pass


# Copied from transformers.model.llama.modeling_llama.LlamaRotaryEmbedding with Llama->InternLM2
class InternLM2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = None
        # inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.max_seq_len_cached = -1 
        # Build here to make `torch.jit.trace` work.
        # self._set_cos_sin_cache(
        #     seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        # )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.inv_freq is None:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
            del self.inv_freq
            self.register_buffer('inv_freq', inv_freq, persistent=False)


        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device).to(dtype=self.inv_freq.dtype)

        # freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq.to(device=t.device))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # if self.max_seq_len_cached == -1:
        #     self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=x.device, dtype=x.dtype)

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:int(seq_len)].to(dtype=x.dtype),
            self.sin_cached[:int(seq_len)].to(dtype=x.dtype),
        )


class V2PE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0,scale_img=False):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = None
        self.scaling_factor=scaling_factor
        self.scale_img=scale_img
        # inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.max_seq_len_cached = -1 
        # Build here to make `torch.jit.trace` work.
        # self._set_cos_sin_cache(
        #     seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        # )

    def _set_cos_sin_cache(self, pos_id, device, dtype,selected):
        if self.inv_freq is None:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
            del self.inv_freq
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        pos_id=pos_id.squeeze(0)
        freqs = torch.outer(pos_id, self.inv_freq.to(device=pos_id.device))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, global_posid=None,selected=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self._set_cos_sin_cache(pos_id=global_posid, device=x.device, dtype=x.dtype,selected=selected)

        return (
            self.cos_cached[:].to(dtype=x.dtype),
            self.sin_cached[:].to(dtype=x.dtype),
        )

# Copied from transformers.model.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->InternLM2
class InternLM2LinearScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.inv_freq is None:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
            del self.inv_freq
            self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device).to(dtype=self.inv_freq.dtype)

        t = t / self.scaling_factor

        # freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq.to(device=t.device))

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)


# Copied from transformers.model.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->InternLM2
class InternLM2DynamicNTKScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.inv_freq is None:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
            del self.inv_freq
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device).to(dtype=self.inv_freq.dtype)

        # freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)


class InternLM2RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_position_embeddings=16, base=100, device=None):
        """
        For image of 16x16 tokens, only 16x16 position embeddings are needed
        Base is set to 100, distinguishing from the global implementation, smaller base is used for fewer max tokens
        Modify if needed
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        x = torch.arange(max_position_embeddings, device=device).to(dtype=theta.dtype)
        y = torch.arange(max_position_embeddings, device=device).to(dtype=theta.dtype)
        
        freqs_x = torch.outer(x, theta[0::2].to(device=x.device))
        freqs_y = torch.outer(y, theta[1::2].to(device=y.device))
        
        freqs_x = torch.cat((freqs_x, freqs_x), dim=-1)
        freqs_y = torch.cat((freqs_y, freqs_y), dim=-1)

        freqs = torch.zeros(max_position_embeddings, max_position_embeddings, self.dim, device=device, dtype=torch.float32)
        freqs[..., 0::2] = freqs_x[:, None, :]
        freqs[..., 1::2] = freqs_y[None, :, :]
        
        self.cos = freqs.cos()
        self.sin = freqs.sin()
        
    def forward(self, x: torch.Tensor, h: int, w: int):
        """
        h and w are shape of image
        shape of x does not matter since only dtype is used
        """
        return (
            self.cos[:h, :w].to(dtype=x.dtype),
            self.sin[:h, :w].to(dtype=x.dtype),
        )


# Copied from transformers.model.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)



# Copied from transformers.model.llama.modeling_llama.apply_rotary_pos_emb; float 
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids.long()].unsqueeze(unsqueeze_dim).float()
    sin = sin[position_ids.long()].unsqueeze(unsqueeze_dim).float()
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(dtype=q_dtype), k_embed.to(dtype=k_dtype)
def apply_rotary_pos_emb_single(states, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the states tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim).float()
    sin = sin[position_ids].unsqueeze(unsqueeze_dim).float()
    states_dtype = states.dtype
    states = states.float()
    states_embed = (states * cos) + (rotate_half(states) * sin)
    return states_embed.to(dtype=states_dtype)


class InternLM2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        return down_proj


# Copied from transformers.model.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Modified from transformers.model.llama.modeling_llama.LlamaAttention
class InternLM2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).'
            )

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
        )

        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        self._init_rope()

    def _init_rope(self):
        if self.training and self.config.rope_scaling['type'] == 'v2pe':
            self.config.rope_scaling['factor'] = 1.0

        if self.config.rope_pos_id_version != "default":
            if self.config.rope_scaling['type'] != 'v2pe':
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"!!!!!!!!!! The rope scaling type is changed from {self.config.rope_scaling['type']} to v2pe !!!!!!!!!!!!")
            self.config.rope_scaling['type'] = 'v2pe'
            self.config.rope_scaling['factor'] = 1.0

        if self.config.rope_scaling is None:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print('The self.config.rope_scaling is None, use InternLM2RotaryEmbedding as self.rotary_emb')
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f'{scaling_type=}, {scaling_factor=}, {self.training}')
            if scaling_type == 'dynamic':
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == 'linear':
                self.rotary_emb = InternLM2LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == 'v2pe':
                self.rotary_emb = V2PE(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                    scale_img=self.config.scale_img,
                )
            else:
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.")
        return self.rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.size()}'
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.size()}'
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Modified from transformers.model.llama.modeling_llama.InternLM2FlashAttention2
class InternLM2FlashAttention2(InternLM2Attention):
    """
    InternLM2 flash attention module. This module inherits from `InternLM2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def init_interactions(self):
        pass

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            selected: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        # q 【100， E】 
        # kv 【200， E】
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop('padding_mask')
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        qkv_states = self.wqkv(hidden_states)

        
        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )
        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        kv_seq_len=position_ids.max()+1
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if self.config.rope_scaling['type'] not in ['linear', 'dynamic']:
            cos, sin = self.rotary_emb(value_states, global_posid=position_ids, selected=selected)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, torch.arange(0,position_ids.shape[1]).unsqueeze(0))
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                group = local_group
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _unpad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(torch.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class InternLM2CrossAttention(nn.Module):
    """Cross-attention mechanism."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        # num heads = 16 num key value heads=4
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).'
            )

        # Query projection (for target hidden states)
        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        
        # Key-value projection (for encoder hidden states)

        self.wkv = nn.Linear(
            self.hidden_size, 2 * self.num_key_value_heads * self.head_dim, bias=config.bias
        )

        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        self._init_rope()
    def reuse_self_attention_params(self, self_attn: nn.Module):

        self.wo.weight.data = self_attn.wo.weight.data.clone()
        if self.config.bias:
            self.wo.bias.data = self_attn.wo.bias.data.clone() if self.config.bias else None

        group_num = self.num_key_value_heads
        wqkv_weight = self_attn.wqkv.weight # [num_heads * 3 * head_dim, hidden_size]
        chunks=torch.chunk(wqkv_weight,group_num,dim=0)
        q_weights_list=[c[:self.num_key_value_groups*self.head_dim,:] for c in chunks]
        kv_weights_list=[c[self.num_key_value_groups*self.head_dim:,:] for c in chunks]
        q_weights=torch.cat(q_weights_list,dim=0)
        kv_weights=torch.cat(kv_weights_list,dim=0)
        if self.config.bias:
            wqkv_bias = self_attn.wqkv.bias.data if self.config.bias else None

        q_end = self.num_heads * self.head_dim
        kv_end = q_end + 2 * self.num_key_value_heads * self.head_dim

        self.wq.weight.data = q_weights.clone()
        if self.config.bias:
            raise NotImplementedError()
            self.wq.bias.data = wqkv_bias[:q_end].clone()

        self.wkv.weight.data = kv_weights.clone()
        if self.config.bias:
            self.wkv.bias.data = wqkv_bias[q_end:kv_end].clone()
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.config.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'dynamic':
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.config.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == 'linear':
                self.rotary_emb = InternLM2LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.config.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # return attn_output, attn_weights, past_key_value
        bsz, q_len, _ = hidden_states.size()
        src_len = encoder_hidden_states.size(1)
        # Project the query from the target hidden states
        query_states = self.wq(hidden_states)
        # num key value groups =4 head dim=128
        query_states=rearrange(query_states,'b q (h gs d) -> b q h gs d', gs=self.num_key_value_groups  ,d=self.head_dim,)

        # Project the key and value from the encoder hidden states
        kv_states = self.wkv(encoder_hidden_states)
        kv_states = rearrange(
            kv_states, 'b q (h gs d) -> b q h gs d', gs= 2 ,d=self.head_dim,
        )
        key_states, value_states = kv_states.chunk(2, dim=-2)
        key_states=rearrange(key_states,'b q h gs d->b q (h gs) d')
        value_states=rearrange(value_states,'b q h gs d->b q (h gs) d')
        query_states=rearrange(query_states,'b q h gs d->b q (h gs) d')
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        q_seq_len = query_states.shape[-2]
        cos_q, sin_q = self.rotary_emb(value_states, seq_len=q_seq_len)
        cos_k, sin_k = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if position_ids is None:
            position_ids_q=torch.arange(0,q_seq_len).unsqueeze(0).cuda()
            position_ids_k=torch.arange(0,kv_seq_len).unsqueeze(0).cuda()   
        query_states, key_states = apply_rotary_pos_emb_single(query_states, cos_q, sin_q, position_ids_q),apply_rotary_pos_emb_single(key_states,cos_k,sin_k,position_ids_k)
        if past_key_value is not None:
            # Reuse k, v from past key-value states
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states=repeat_kv(value_states,self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_seq_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_seq_len, kv_seq_len)}, but is '
                f'{attn_weights.size()}'
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_seq_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_seq_len, kv_seq_len)}, but is '
                    f'{attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        if encoder_attention_mask is not None:
            if encoder_attention_mask.size() != (bsz, 1, 1, kv_seq_len):
                raise ValueError(
                    f'Encoder attention mask should be of size {(bsz, 1, 1, kv_seq_len)}, but is '
                    f'{encoder_attention_mask.size()}'
                )
            attn_weights = attn_weights + encoder_attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_seq_len, self.head_dim):
            raise ValueError(
                f'Attention output should be of size {(bsz, self.num_heads, q_seq_len, self.head_dim)}, but is '
                f'{attn_output.size()}'
            )

        # (bsz, q_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_seq_len, self.hidden_size)

        attn_output = self.wo(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output


class InternLM2CrossAttentionForPackedTraining(InternLM2FlashAttention2):
    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} '
                f'and `num_heads`: {self.num_heads}).'
            )

        self.wq = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.bias,
        )

        self.wkv = nn.Linear(
            self.hidden_size,  # key-value hidden_size
            2 * self.num_key_value_heads * self.head_dim,  # 2 * key_value_heads * head_dim
            bias=config.bias,
        )

        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

        self._init_rope()
    def reuse_self_attention_params(self, self_attn: nn.Module):

        self.wo.weight.data = self_attn.wo.weight.data.clone()
        if self.config.bias:
            self.wo.bias.data = self_attn.wo.bias.data.clone() if self.config.bias else None

        group_num = self.num_key_value_heads
        wqkv_weight = self_attn.wqkv.weight # [num_heads * 3 * head_dim, hidden_size]
        chunks=torch.chunk(wqkv_weight,group_num,dim=0)
        q_weights_list=[c[:self.num_key_value_groups*self.head_dim,:] for c in chunks]
        kv_weights_list=[c[self.num_key_value_groups*self.head_dim:,:] for c in chunks]
        q_weights=torch.cat(q_weights_list,dim=0)
        kv_weights=torch.cat(kv_weights_list,dim=0)
        if self.config.bias:
            wqkv_bias = self_attn.wqkv.bias.data if self.config.bias else None

        q_end = self.num_heads * self.head_dim
        kv_end = q_end + 2 * self.num_key_value_heads * self.head_dim

        self.wq.weight.data = q_weights.clone()
        if self.config.bias:
            raise NotImplementedError()
            self.wq.bias.data = wqkv_bias[:q_end].clone()

        self.wkv.weight.data = kv_weights.clone()
        if self.config.bias:
            self.wkv.bias.data = wqkv_bias[q_end:kv_end].clone()

    def forward(
        self,
        query_seq, key_value_seq,
        cu_seqlens_q, cu_seqlens_k,
        position_ids: Optional[Tuple] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop('padding_mask')
        output_attentions = False

        bsz, q_len, _ = query_seq.size()
        query_states = self.wq(query_seq)
        key_value_states = self.wkv(key_value_seq)
        query_states = rearrange(
            query_states,
            'b q (h gs d) -> b q h gs d',
            gs=self.num_key_value_groups,
            d=self.head_dim,
        )
        query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
        key_value_states=rearrange(
            key_value_states,
            'b q (h gs d) -> b q h gs d',
            gs=2,
            d=self.head_dim
        )
        key_states = key_value_states[..., 0, :]
        value_states = key_value_states[..., 1, :]
        

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        q_position_ids, kv_position_ids = position_ids
        kv_seq_len = kv_position_ids.max()+1
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        q_seq_len = q_position_ids.max()+1
        if past_key_value is not None:
            q_seq_len += past_key_value[0].shape[-2]
        
        # method B
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # -------------------------------------------------
        # method C
        cos, sin = self.rotary_emb(value_states, seq_len=q_seq_len)
        # ---------------------------------------------------
        # q_cos, q_sin = self.rotary_emb(query_states, seq_len=q_seq_len)
        

        if kv_position_ids[0][0]!=0:
            kv_position_ids=kv_position_ids-kv_position_ids[0][0]

        query_states, key_states = apply_rotary_pos_emb_single(query_states, cos, sin, q_position_ids), apply_rotary_pos_emb_single(key_states, cos, sin, kv_position_ids)
        
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        attn_output = self._flash_cross_attention_forward(
            query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        attn_output = self.wo(attn_output)
        
        if not output_attentions:
            attn_weights = None
        return attn_output
    def _flash_cross_attention_forward(
        self, query_states, key_states, value_states, 
        cu_seqlens_q, cu_seqlens_k, dropout=0.0, softmax_scale=None
        ):
        """
        Computes cross attention using Flash Attention. 

        Args:
            query_states (`torch.Tensor`):
                Input query states (shape: [1, total_q, nheads, headdim]).
            key_states (`torch.Tensor`):
                Input key states (shape: [1, total_k, nheads, headdim]).
            value_states (`torch.Tensor`):
                Input value states (shape: [1, total_k, nheads, headdim]).
            cu_seqlens_q (`torch.Tensor`):
                Cumulative sequence lengths of query sequences in the batch (shape: [batch_size + 1]).
            cu_seqlens_k (`torch.Tensor`):
                Cumulative sequence lengths of key/value sequences in the batch (shape: [batch_size + 1]).
            dropout (`float`, *optional*):
                Attention dropout.
            softmax_scale (`float`, *optional*):
                Scaling factor for QK^T before softmax (default: 1 / sqrt(headdim)).
        """
        # Remove the batch dimension (squeeze(0)) as Flash Attention expects flattened tensors.
        query_states = query_states.squeeze(0)  # (total_q, nheads, headdim)
        key_states = key_states.squeeze(0)      # (total_k, nheads, headdim)
        value_states = value_states.squeeze(0)  # (total_k, nheads, headdim)
        # Calculate the max sequence lengths for query and key sequences.
        cu_seqlens_q=cu_seqlens_q.squeeze(0)
        cu_seqlens_k=cu_seqlens_k.squeeze(0)

        with torch.no_grad():
            max_seqlen_q = max([
                cu_seqlens_q[idx + 1] - cu_seqlens_q[idx]
                for idx in range(cu_seqlens_q.size(0) - 1)
            ]).item()

            max_seqlen_k = max([
                cu_seqlens_k[idx + 1] - cu_seqlens_k[idx]
                for idx in range(cu_seqlens_k.size(0) - 1)
            ]).item()

        # Set causal=False for cross-attention (unless you need specific behavior).
        causal = self.is_causal
        # method B method C
        assert causal==False

        # Perform Flash Attention.
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        # Check for NaNs in the attention output.
        if torch.isnan(attn_output).any():
            raise ValueError("Attention output contains NaN values")

        # Add back the batch dimension (unsqueeze(0)).
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)

        return attn_output

INTERNLM2_ATTENTION_CLASSES = {
    'eager': InternLM2Attention,
    'flash_attention_2': InternLM2FlashAttention2,
}


# Modified from transformers.model.llama.modeling_llama.LlamaDecoderLayer
class InternLM2DecoderLayer(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = INTERNLM2_ATTENTION_CLASSES[config.attn_implementation](config=config)

        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config=config
    def init_interactions(self,compress_seq=False,fuse_method='add', compress_method='avg'):
        self.attention.init_interactions()
        
        if compress_seq:
            self.compress_seq=True
            self.interaction=INTERNLM2_ATTENTION_CLASSES[self.config.attn_implementation](config=self.config)
            self.layer_scale=LayerScale(self.config.hidden_size,init_values=1e-3)
            self.sigmoid_layer_scale = Sigmoid(self.config.hidden_size)
            # self.layer_scale.gamma.requires_grad=False
            self.fuse_method=fuse_method
            if self.fuse_method=='cross-attn':
                self.fuse_layer=InternLM2CrossAttentionForPackedTraining(self.config)
                self.fuse_layer.reuse_self_attention_params(self.attention)
            elif self.fuse_method=='simple-cross-attn':
                self.fuse_layer=InternLM2CrossAttention(self.config)
                self.fuse_layer.reuse_self_attention_params(self.attention)
            elif self.fuse_method=='add':
                self.fuse_layer=None
            else:
                raise NotImplementedError()
            self.compress_method=compress_method
            if compress_method=='attention':
                self.pooling_layer=AttentionPooling(self.config.hidden_size, FINAL_SIZE)
            elif compress_method=='topk':
                self.pooling_layer=TopKPooling(self.config.hidden_size, FINAL_SIZE)
            elif compress_method=='avg':
                self.pooling_layer=None
            else:
                raise NotImplementedError()
            # initialize
            for layer_param, interaction_param in zip(self.attention.parameters(), self.interaction.parameters()):
                interaction_param.data.copy_(layer_param.data)
        else:
            self.compress_seq=False

    def fuse(self,compressed_data,hidden_states,inner_idx=0,chunk_num=None,chunk_size=100,cu_seqlens_q=None, cu_seqlens_k=None,method='add',position_ids=None):
        if method=='add':
            return self.layer_scale(torch.sum(compressed_data[:,:inner_idx*chunk_size,:],dim=1))+hidden_states
        elif method=='cross-attn':
            cu_seqlens_k_list=chunk_with_boundaries(cu_seqlens_k[0][-1],cu_seqlens_k,chunk_num)
            if inner_idx==0:
                return hidden_states+0.0*self.fuse_layer(hidden_states,compressed_data[:,inner_idx*chunk_size:(inner_idx+1)*chunk_size,:],cu_seqlens_q,cu_seqlens_k_list[inner_idx],position_ids=(position_ids[0],position_ids[1][:,inner_idx*chunk_size:(inner_idx+1)*chunk_size]))
            else:
                return self.layer_scale(self.fuse_layer(hidden_states,compressed_data[:,(inner_idx-1)*chunk_size:inner_idx*chunk_size,:],cu_seqlens_q,cu_seqlens_k_list[inner_idx],position_ids=(position_ids[0],position_ids[1][:,(inner_idx-1)*chunk_size:inner_idx*chunk_size])))+hidden_states
        else:
            raise ValueError(f"Unknown method: {method}")
    def compress2(self, hidden_states, pos_ids, method='avg', final_size=FINAL_SIZE):
        if method == 'avg':
            B, N, C = hidden_states.shape

            step_size = N // final_size

            averaged_groups = [
                hidden_states[:, i * step_size: (i + 1) * step_size, :].mean(dim=1, keepdim=True)
                for i in range(final_size)
            ]

            pos_ids_groups = [
                pos_ids[:, i * step_size: (i + 1) * step_size].median(dim=1, keepdim=True).values 
                for i in range(final_size)
            ]

            result = torch.cat(averaged_groups, dim=1)
            pos_ids_res = torch.cat(pos_ids_groups, dim=1)

            return result, pos_ids_res
    def compress(self,hidden_states,method='avg',final_size=FINAL_SIZE):
        if method=='avg':
            B, N, C = hidden_states.shape

            step_size = N // final_size

            averaged_groups = [
                hidden_states[:, i * step_size: (i + 1) * step_size, :].mean(dim=1, keepdim=True) 
                for i in range(final_size)
            ]

            # (B, 100, C)
            result = torch.cat(averaged_groups, dim=1)

            return result
        elif method=='attention':
            return self.pooling_layer(hidden_states)
        elif method=='topk':
            return self.pooling_layer(hidden_states)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            origin_cu_seq_lens: Optional[torch.Tensor] = None,
            fuse_only: Optional[torch.Tensor] = False,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            selected: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. '
                'Please make sure use `attention_mask` instead.`'
            )
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        if not hasattr(self,'compress_seq'):
            self.compress_seq=False
        if self.compress_seq:
            if fuse_only:
                _, length, channels= hidden_states.shape
                PADDING_LENGTH=8192
                # padding hidden states to b,padding length, c
                padding_size = PADDING_LENGTH - length

                pad_hidden_states = torch.zeros((hidden_states.size(0), padding_size, channels), device=hidden_states.device).to(hidden_states.dtype)
                pad_hidden_states = torch.cat((hidden_states, pad_hidden_states), dim=1)
                pad_all_hiddenstates=GatherLayer.apply(pad_hidden_states)
                length_tensor = torch.tensor([length], dtype=torch.int).cuda()
                origin_length_tensor=GatherLayer.apply(length_tensor)

                # method C----------------------------------------
                
                if inner_idx>0:
                    prev_seq=pad_all_hiddenstates[:inner_idx]

                    prev_len=origin_length_tensor[:inner_idx]
                    B = prev_seq.size(1)  # batch size
                    C = prev_seq.size(3)  # channels

                    unpad_hidden_states_list = []

                    for i in range(prev_len.size(0)):  # num_processes
                        valid_hidden_states = prev_seq[i, :B, :prev_len[i], :]
                        unpad_hidden_states_list.append(valid_hidden_states)

                
                    prev_hidden_states = torch.cat(unpad_hidden_states_list, dim=1)
                else:
                    assert dist.get_rank()==0
                    prev_seq=pad_all_hiddenstates[:1]

                    prev_len=origin_length_tensor[:1]
                    B = prev_seq.size(1)  # batch size
                    C = prev_seq.size(3)  # channels

                    unpad_hidden_states_list = []
                    for i in range(prev_len.size(0)):  # num_processes
                        valid_hidden_states = prev_seq[i, :B, :prev_len[i], :]
                        unpad_hidden_states_list.append(valid_hidden_states)

                
                    prev_hidden_states = torch.cat(unpad_hidden_states_list, dim=1)
                # since batch size=1, only 1 sample packed
                # TODO: make compatible for other cases

                prev_position_id = torch.arange(0,prev_hidden_states.size(1)).unsqueeze(0).cuda()
                prev_hidden_states,prev_position_id=self.compress2(prev_hidden_states,prev_position_id)
                cu_seqlens_k = torch.tensor([[0,prev_hidden_states.size(1)]],dtype=attention_mask.dtype,device=attention_mask.device)
                right_bound = prev_len.sum().item()
                
                left_bound = right_bound-length_tensor.item()
                position_ids = torch.arange(left_bound,right_bound).unsqueeze(0).cuda()
                # ------------------------------------------------
            else:
                _, length, _ = hidden_states.shape
                length_tensor = torch.tensor([length], dtype=torch.int).cuda()
                compressed_chunk = self.compress(hidden_states,method=self.compress_method)
                B, N, C = compressed_chunk.shape
                compressed_data=GatherLayer.apply(compressed_chunk)
                origin_length_tensor=GatherLayer.apply(length_tensor)
                origin_length=torch.sum(origin_length_tensor,dim=0).unsqueeze(1)#shape B,1
                pn_size = compressed_data.size(0) * compressed_data.size(2)                
                compressed_data = compressed_data.reshape(-1, pn_size, compressed_data.size(3))
                new_length=compressed_data.shape[1]
                new_cu_seq_lens=origin_cu_seq_lens*new_length//origin_length
                new_cu_seq_lens=new_cu_seq_lens.to(torch.int32).to(hidden_states.device)
                compressed_pos_id=torch.arange(0,compressed_data.shape[1]).unsqueeze(0).repeat(B,1).cuda()
                compressed_data = self.interaction(compressed_data, new_cu_seq_lens, compressed_pos_id, None, output_attentions, use_cache)[0] # 1， 4*100， E
                chunk_num=compressed_data.size(1)//N
        
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            selected=selected,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


InternLM2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`InternLM2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->InternLM2
@add_start_docstrings(
    'The bare InternLM2 Model outputting raw hidden-states without any specific head on top.',
    InternLM2_START_DOCSTRING,
)
class InternLM2PreTrainedModel(PreTrainedModel):
    config_class = InternLM2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternLM2DecoderLayer']
    _skip_keys_device_placement = 'past_key_values'

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


InternLM2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
            when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Modified from transformers.model.llama.modeling_llama.LlamaModel
@add_start_docstrings(
    'The bare InternLM2 Model outputting raw hidden-states without any specific head on top.',
    InternLM2_START_DOCSTRING,
)
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size(local_group))]
        dist.all_gather(output, input, group=local_group)
        return torch.stack(output, 0)

    @staticmethod
    def backward(ctx, grads):
        (input,) = ctx.saved_tensors
        dist.all_reduce(grads, group=local_group)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank(local_group)]
        return grad_out

class InternLM2Model(InternLM2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLM2DecoderLayer`]

    Args:
        config: InternLM2Config
    """

    _auto_class = 'AutoModel'

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        if not has_flash_attn:
            self.config.attn_implementation = 'eager'
            print('Warning: Flash attention is not available, using eager attention instead.')

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([InternLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # global attn_type
        # attn_type = None
        self.post_init()

    def init_interactions(self,compress_seq, fuse_method='add', compress_method='avg'):
        for layer in self.layers:
            layer.init_interactions(compress_seq,fuse_method,compress_method)

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            compress_seq: Optional[bool] = False,
            group_list: Optional[List] = None,
            chunk_num: Optional[int] = None,
            origin_cu_seq_lens: Optional[torch.tensor] = None,
            interaction: Optional[bool] = True,
            selected: Optional[torch.tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # origin_cu_seq_lens: B,N
        global local_group
        if group_list is not None:
            for group_idx,group in enumerate(group_list):
                if type(group)==torch.distributed.distributed_c10d.ProcessGroup:
                    break
            global inner_idx
            inner_idx = dist.get_rank(group)
            local_group=group
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.attn_implementation == 'flash_attention_2':
            _import_flash_attn()

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)
        if self.config.attn_implementation == 'flash_attention_2':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            # in which process group
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            fuse_only = not interaction
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    origin_cu_seq_lens,
                    fuse_only,
                    None,
                    selected,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    origin_cu_seq_lens=origin_cu_seq_lens,
                    fuse_only=fuse_only,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    selected=selected,
                )
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
                
        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def fuse(self,idx ,compressed_data,hidden_states,inner_idx=0,chunk_num=None,chunk_size=100,cu_seqlens_q=None, cu_seqlens_k=None,method='add',fuse_layer=None,position_ids=None):
        if method=='add':
            return self.layer_scale[idx](torch.sum(compressed_data[:,:inner_idx*chunk_size,:],dim=1))+hidden_states
        elif method=='cross-attn':
            cu_seqlens_k_list=chunk_with_boundaries(cu_seqlens_k[0][-1],cu_seqlens_k,chunk_num)
            if inner_idx==0:
                return hidden_states+0.0*fuse_layer(hidden_states,compressed_data[:,inner_idx*chunk_size:(inner_idx+1)*chunk_size,:],cu_seqlens_q,cu_seqlens_k_list[inner_idx],position_ids=(position_ids[0],position_ids[1][:,inner_idx*chunk_size:(inner_idx+1)*chunk_size]))
            else:
                return self.layer_scale[idx](fuse_layer(hidden_states,compressed_data[:,(inner_idx-1)*chunk_size:inner_idx*chunk_size,:],cu_seqlens_q,cu_seqlens_k_list[inner_idx],position_ids=(position_ids[0],position_ids[1][:,(inner_idx-1)*chunk_size:inner_idx*chunk_size])))+hidden_states
        else:
            raise ValueError(f"Unknown method: {method}")
    def compress(self,idx,hidden_states, method='avg',final_size=FINAL_SIZE):
        if method=='avg':
            B, N, C = hidden_states.shape

            step_size = N // final_size

            averaged_groups = [
                hidden_states[:, i * step_size: (i + 1) * step_size, :].mean(dim=1, keepdim=True) 
                for i in range(final_size)
            ]

            result = torch.cat(averaged_groups, dim=1)

            return result
        elif method=='attention':
            return self.pooling_layers[idx](hidden_states)
        elif method=='topk':
            return self.pooling_layers[idx](hidden_states)
        else:
            raise ValueError(f"Unknown method: {method}")


# Modified from transformers.model.llama.modeling_llama.LlamaForCausalLM
class InternLM2ForCausalLM(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            compress_seq: Optional[bool] = False,
            group_list: Optional[List] = None,
            chunk_num: Optional[int] = 1,
            origin_cu_seq_lens: Optional[torch.tensor] = None,
            interaction: Optional[bool] = True,
            selected: Optional[torch.tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, InternLM2ForCausalLM

        >>> model = InternLM2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            compress_seq=compress_seq,
            group_list=group_list,
            chunk_num=chunk_num,
            origin_cu_seq_lens=origin_cu_seq_lens,
            interaction=interaction,
            selected=selected,
        )
        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output['logits'] = output['logits'].to(device)
        return output

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        elif position_ids is not None:
            if self.rope_pos_id_version!='default' and past_key_values is not None:
                position_ids=(position_ids[:,-1]+attention_mask[:,position_ids.shape[1]:].sum(dim=1)).unsqueeze(1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update(
            {
                'position_ids': position_ids,
                'past_key_values': past_key_values,
                'use_cache': kwargs.get('use_cache'),
                'attention_mask': attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = [], meta_instruction=''):
        if tokenizer.add_bos_token:
            prompt = ''
        else:
            prompt = tokenizer.bos_token
        if meta_instruction:
            prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
        for record in history:
            prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        return tokenizer([prompt], return_tensors='pt')

    @torch.no_grad()
    def chat(
            self,
            tokenizer,
            query: str,
            history: List[Tuple[str, str]] = [],
            streamer: Optional[BaseStreamer] = None,
            max_new_tokens: int = 1024,
            do_sample: bool = True,
            temperature: float = 0.8,
            top_p: float = 0.8,
            meta_instruction: str = 'You are an AI assistant whose name is InternLM (书生·浦语).\n'
                                    '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
                                    '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.',
            **kwargs,
    ):
        inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(['<|im_end|>'])[0]]
        outputs = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        outputs = outputs[0].cpu().tolist()[len(inputs['input_ids'][0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split('<|im_end|>')[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(
            self,
            tokenizer,
            query: str,
            history: List[Tuple[str, str]] = [],
            max_new_tokens: int = 1024,
            do_sample: bool = True,
            temperature: float = 0.8,
            top_p: float = 0.8,
            **kwargs,
    ):

        if BaseStreamer is None:
            raise ModuleNotFoundError(
                'The version of `transformers` is too low. Please make sure '
                'that you have installed `transformers>=4.28.0`.'
            )

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ''
                self.cache = []
                self.received_inputs = False
                self.queue.put((self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError('ChatStreamer only supports batch size 1')
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                self.cache.extend(value.tolist())
                token = self.tokenizer.decode(self.cache, skip_special_tokens=True)
                if token.strip() != '<|im_end|>':
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))
                    self.cache = []
                else:
                    self.end()

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()


# Copied from transformers.model.llama.modeling_llama.LlamaForSequenceClassification with Llama->InternLM2
@add_start_docstrings(
    """
    The InternLM2 Model transformer with a sequence classification head on top (linear layer).

    [`InternLM2ForSequenceClassification`] uses the last token in order to do the classification,
    as other causal models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    InternLM2_START_DOCSTRING,
)
class InternLM2ForSequenceClassification(InternLM2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLM2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
