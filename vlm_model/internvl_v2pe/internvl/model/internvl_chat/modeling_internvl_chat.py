# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel

logger = logging.get_logger(__name__)
import random


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))
def extract_local(value, rank, world_size, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.to(value.device)
def extract_local2(value, rank, world_size,  dim=1):
    dimension_size = value.shape[dim]
    sub_seq_length = dimension_size // world_size

    sub_seq_start = rank * sub_seq_length
    sub_seq_end = (rank + 1) * sub_seq_length
    local_value = value[:, sub_seq_start:sub_seq_end]

    return local_value.to(value.device)
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
class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.compress_seq = config.compress_seq
        self.attn_type = config.attn_type
        self.group_list = config.group_list
        self.chunk_num = config.chunk_num
        self.interaction = config.interaction

        self.img_emb_down_sample_ratio = getattr(config, 'img_emb_down_sample_ratio', None)
        if self.img_emb_down_sample_ratio is not None:
            self.num_image_token = int(self.num_image_token / self.img_emb_down_sample_ratio)

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'img_emb_down_sample_ratio: {self.img_emb_down_sample_ratio}, use its inverse number to downsample num_image_token for adaptive pooling')

        config.llm_config.rope_pos_id_version = config.rope_pos_id_version
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
            
        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)
    def init_embed(self):
        if hasattr(self,'local_posid'):
            nn.init.normal_(self.local_posid.weight, mean=0.0, std=0.02)
    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
    
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
            origin_cu_seq_lens: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if isinstance(position_ids,list):
            position_ids=torch.tensor(position_ids).to(input_ids.device)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        global local_group
        if self.group_list is not None:
            for group_idx,group in enumerate(self.group_list):
                if type(group)==torch.distributed.distributed_c10d.ProcessGroup:
                    # assert type(group)==torch.distributed.distributed_c10d.ProcessGroup
                    break        # print("Printing decoded input ids")
            local_group=group
        else:
            group=None
            local_group=None
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        if self.attn_type:
            if self.attn_type=='ring':
                group_size = dist.get_world_size(group)
                img_num_dim = 0
                pad_num=0
                if pixel_values.shape[img_num_dim] > group_size:
                    if pixel_values.shape[img_num_dim] % group_size!=0:
                        pad_num = group_size - pixel_values.shape[img_num_dim] % group_size
                        if pad_num < group_size:  
                            pad_shape = list(pixel_values.shape)
                            pad_shape[img_num_dim] = pad_num  
                            pad_pixel = torch.zeros(pad_shape, dtype=pixel_values.dtype, device=pixel_values.device)

                            pixel_values = torch.cat([pixel_values, pad_pixel], dim=img_num_dim)

                    chunked_pixel=torch.chunk(pixel_values, group_size, dim=img_num_dim)
                    local_pixel=chunked_pixel[dist.get_rank(group)]
                    local_vit_embeds=self.extract_feature(local_pixel)
                    vit_embeds=GatherLayer.apply(local_vit_embeds)
                    vit_embeds=vit_embeds.view(-1,vit_embeds.shape[-2],vit_embeds.shape[-1])
                    if pad_num>0:
                        vit_embeds=vit_embeds[:-pad_num]
                else:
                    vit_embeds = self.extract_feature(pixel_values)
            else:
                vit_embeds = self.extract_feature(pixel_values)
        else:
            vit_embeds = self.extract_feature(pixel_values)
        


        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]
        # print("Printing pixiel shape", pixel_values.shape)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            # ignore_flag = True
            ignore_flag = False

        input_embeds = input_embeds.reshape(B, N, C)
        if self.attn_type:
            if self.attn_type=='ulysses':
                input_embeds=extract_local2(input_embeds,dist.get_rank(group),dist.get_world_size(group))
                position_ids=extract_local2(position_ids,dist.get_rank(group),dist.get_world_size(group))
                labels=extract_local2(labels,dist.get_rank(group),dist.get_world_size(group))
                loss_weight=extract_local2(torch.tensor(loss_weight),dist.get_rank(group),dist.get_world_size(group))
                loss_weight=list(loss_weight.numpy())
                attention_mask=attention_mask//dist.get_world_size(group)
            elif self.attn_type=='ring':
                input_embeds=extract_local(input_embeds,dist.get_rank(group),dist.get_world_size(group))
                position_ids=extract_local(position_ids,dist.get_rank(group),dist.get_world_size(group))
                labels=extract_local(labels,dist.get_rank(group),dist.get_world_size(group))
                if loss_weight:
                    loss_weight=extract_local(torch.tensor(loss_weight),dist.get_rank(group),dist.get_world_size(group))
                    loss_weight=list(loss_weight.numpy())
                attention_mask=attention_mask//dist.get_world_size(group)
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            compress_seq=self.compress_seq,
            group_list=self.group_list,
            chunk_num=self.chunk_num,
            origin_cu_seq_lens=origin_cu_seq_lens,
            interaction=self.interaction,
            selected=selected
        )
        logits = outputs.logits
        
        loss = None
        if labels is not None and loss_weight is not None:
            # decoded_labels = global_tokenizer.decode(labels[0][labels[0]!=-100], skip_special_tokens=True)
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
                
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # self.update_log(log_dict)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
       
        vit_embeds = vit_embeds[:, 1:, :]

        # [batch_size, num_patches, vit_hidden_size] -> [batch_size, h, w, vit_hidden_size]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        if self.img_emb_down_sample_ratio is not None:
            vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
            vit_embeds = F.adaptive_avg_pool1d(vit_embeds, self.num_image_token)
            vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
            assert vit_embeds.shape[1] == self.num_image_token
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')
        
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        # tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=False)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, **kwargs):
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question


        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)


        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id


        template = get_conv_template(self.template)

        template.system_message = self.system_message

        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)


        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)

        query = template.get_prompt()


        if verbose and pixel_values is not None:

            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')

        input_ids = model_inputs['input_ids'].cuda()

        attention_mask = model_inputs['attention_mask'].cuda()

        generation_config['eos_token_id'] = eos_token_id
        if 'rope_pos_id_version' in kwargs:
            self.language_model.rope_pos_id_version=kwargs['rope_pos_id_version']
            pos_ids=[]
            ret={'input_ids':input_ids,'attention_mask':attention_mask}
            for i in range(input_ids.shape[0]):
                if kwargs['rope_pos_id_version'] == 'default':
                    cur_dtype = torch.long
                else:
                    cur_dtype = torch.float32

                if 'rope_pos_id_stride' in kwargs:
                    rope_pos_id_stride = kwargs['rope_pos_id_stride']
                else:
                    rope_pos_id_stride = None

                cur_pos_id = get_rope_pos_id(ret, tokenizer=tokenizer, num_tiles=kwargs['num_tiles'][i],
                                            dtype=cur_dtype,
                                           rope_pos_id_version=kwargs['rope_pos_id_version'],
                                           position_id=torch.arange(0,input_ids.shape[1]),
                                           IMG_START_TOKEN=IMG_START_TOKEN,
                                           IMG_END_TOKEN=IMG_END_TOKEN, rope_pos_id_stride=rope_pos_id_stride)

                cur_pos_id = torch.tensor(cur_pos_id).cuda()

                pos_ids.append(cur_pos_id)
                
            pos_ids=torch.stack(pos_ids)
            if self.attn_type=='ulysses' or self.attn_type=='ring':
                if input_ids.shape[1]%(2*dist.get_world_size())!=0:
                    num_padding = 2*dist.get_world_size()-input_ids.shape[1]%(2*dist.get_world_size())

                    padding_shape = (input_ids.shape[0], num_padding)
                    input_padding = torch.full(padding_shape, 1, dtype=input_ids.dtype, device=input_ids.device)
                    attn_mask_padding = torch.full(padding_shape, 0, dtype=attention_mask.dtype, device=attention_mask.device)

                    input_ids = torch.cat([input_ids, input_padding], dim=1)
                    attention_mask=torch.cat([attention_mask,attn_mask_padding],dim=1)

                    max_pos_id = pos_ids.max() + 1
                    pos_padding = torch.arange(max_pos_id, max_pos_id + num_padding, device=input_ids.device)
                    pos_padding = pos_padding.unsqueeze(0).expand(input_ids.shape[0], -1)
                    pos_ids = torch.cat([pos_ids, pos_padding], dim=1)
            generation_output = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                **generation_config,
            )
        else:
            self.language_model.rope_pos_id_version='default'
            if self.attn_type=='ulysses' or self.attn_type=='ring':
                if input_ids.shape[1]%(2*dist.get_world_size())!=0:
                    num_padding = 2*dist.get_world_size()-input_ids.shape[1]%(2*dist.get_world_size())

                    padding_shape = (input_ids.shape[0], num_padding)
                    input_padding = torch.full(padding_shape, 1, dtype=input_ids.dtype, device=input_ids.device)
                    attn_mask_padding = torch.full(padding_shape, 0, dtype=attention_mask.dtype, device=attention_mask.device)

                    input_ids = torch.cat([input_ids, input_padding], dim=1)
                    attention_mask=torch.cat([attention_mask,attn_mask_padding],dim=1)
            generation_output = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config,
            )

        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]

        response = response.split(template.sep)[0].strip()

        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:

            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            # [1, sequence_length, embedding_dim] -> [sequence_length, embedding_dim]
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            # [1, sequence_length] -> [sequence_length]
            input_ids = input_ids.reshape(B * N)

            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0

            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:

            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        if self.attn_type:
            if self.attn_type=='ulysses':
                assert dist.get_world_size()==4
                
                input_embeds=extract_local2(input_embeds,dist.get_rank(),dist.get_world_size())
                attention_mask=extract_local2(attention_mask,dist.get_rank(),dist.get_world_size())
            elif self.attn_type=='ring':
                former_shape = input_embeds.shape
                input_embeds=extract_local(input_embeds,dist.get_rank(),dist.get_world_size())
                attention_mask=extract_local(attention_mask,dist.get_rank(),dist.get_world_size())
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    def update_log(self, new_log_dict):
        if not hasattr(self, 'log_dict'):
            self.log_dict = {}
        for key, value in new_log_dict.items():
            if 'loss' in key:
                if key not in self.log_dict:
                    self.log_dict[key] = value
                else:
                    self.log_dict[key] += value
            else:
                # just copy it
                self.log_dict[key] = value

def get_rope_pos_id(ret, num_tiles, dtype, rope_pos_id_version='default', position_id=None,
                    IMG_START_TOKEN='<img>',IMG_END_TOKEN='</img>',rope_pos_id_stride=None, tokenizer=None):
    image_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    num_image_token=256
    rope_pos_id_list = []

    input_ids_0 = ret['input_ids'][0]
    attention_mask_0 = ret['attention_mask'][0]
    image_start_token_id_idxs = torch.where(input_ids_0 == image_start_token_id)[0]
    image_end_token_id_idxs = torch.where(input_ids_0 == image_end_token_id)[0]

    last_record_pos_id = -1
    start_index = 0

    assert rope_pos_id_version in ['v2pe_fix', 'v2pe_rnd', 'default'], f'{rope_pos_id_version} not supported for eval'


    for i in range(len(image_start_token_id_idxs)):

        num_tile = num_tiles[i]

        rope_pos_id_pre = attention_mask_0[start_index:image_start_token_id_idxs[i] + 1].long().cumsum(-1) - 1 + (last_record_pos_id + 1)
        rope_pos_id_pre.masked_fill_(attention_mask_0[start_index:image_start_token_id_idxs[i] + 1] == 0, 1)
        rope_pos_id_list.append(rope_pos_id_pre)

        last_record_pos_id = rope_pos_id_pre[-1].long()

        if rope_pos_id_version == 'v2pe_fix':
            assert rope_pos_id_stride is not None, 'when rope_pos_id_version is fix, self.rope_pos_id_stride should not be None'
            small_stride = rope_pos_id_stride / num_image_token
            split_img_id_idxs = torch.arange(last_record_pos_id, last_record_pos_id + small_stride * (num_image_token * num_tile + 1), small_stride)[1:].to(dtype=dtype)
            rope_pos_id_list.append(split_img_id_idxs)
            last_record_pos_id = torch.ceil(split_img_id_idxs[-1]).long()
        elif rope_pos_id_version == 'v2pe_rnd':
            random_from=[1,2,4,8,16,32,64,128,256]
            rope_pos_id_stride=random.choice(random_from)
            small_stride = rope_pos_id_stride / num_image_token
            split_img_id_idxs = torch.arange(last_record_pos_id, last_record_pos_id + small_stride * (num_image_token * num_tile + 1), small_stride)[1:].to(dtype=dtype)
            rope_pos_id_list.append(split_img_id_idxs)
            last_record_pos_id = torch.ceil(split_img_id_idxs[-1]).long()
        elif rope_pos_id_version == 'default':
            split_img_id_idxs = torch.linspace(last_record_pos_id,
                                               last_record_pos_id + (num_tile - 1) * num_image_token,
                                               (num_tile - 1) * num_image_token + 1)[1:].to(dtype=dtype)
            rope_pos_id_list.append(split_img_id_idxs)
            thumbnail_id_idxs = torch.linspace(last_record_pos_id + (num_tile - 1) * num_image_token,
                                               last_record_pos_id + num_tile * num_image_token,
                                               num_image_token + 1)[1:].to(dtype=dtype)
            rope_pos_id_list.append(thumbnail_id_idxs)
            last_record_pos_id = (last_record_pos_id + num_tile * num_image_token).long()
        else:
            raise NotImplementedError(f'not implement for {rope_pos_id_version}')

        start_index = image_start_token_id_idxs[i] + num_tile * num_image_token + 1
        assert input_ids_0[start_index] == image_end_token_id
        assert start_index == image_end_token_id_idxs[i]

    assert image_end_token_id_idxs[-1] == start_index
    rope_pos_id_pre = attention_mask_0[start_index:].long().cumsum(-1) - 1 + (last_record_pos_id + 1)
    rope_pos_id_pre.masked_fill_(attention_mask_0[start_index:] == 0, 1)
    rope_pos_id_list.append(rope_pos_id_pre)

    rope_pos_id_list=[_.to('cpu') for _ in rope_pos_id_list]
    rope_pos_id = torch.cat(rope_pos_id_list).to(dtype=dtype)
    if rope_pos_id_version == 'default':
        rope_pos_id = rope_pos_id.long()
        assert torch.equal(rope_pos_id, position_id.to(rope_pos_id.device)), (rope_pos_id, position_id.to(rope_pos_id.device))
        assert torch.allclose(rope_pos_id, position_id.to(rope_pos_id.device), atol=1e-32)

    assert rope_pos_id.shape == input_ids_0.shape

    return list(rope_pos_id.numpy())
