import gc
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional
import hashlib

import numpy as np
import orjson as json
import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
from internvl.train.compress_seq_trainer import chunkTrainer
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
import internvl.train.trainer_monkey_patch
import internvl.model.internlm2

# Upgrade transformers to v4.37.2, we don't need it anymore
# replace_llama2_attn_with_flash_attn()
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()
replace_train_dataloader()

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained tokenizer from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v1',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )
    compress_seq: bool = field(
        default=False,
        metadata={'help': 'Set to True to compress the sequence length.'},
    )
    chunk_num: Optional[int] = field(
        default=1,
        metadata={'help': 'The number of chunks to split the sequence. Default is 1.'},
    )
    interaction: Optional[bool] = field(
        default=True,
        metadata={'help': 'Set to True to enable the interaction between subsequences.'},
    )
    fuse_method: Optional[str] = field(
        default='add',
        metadata={'help': 'Specify the fusion method for the compressed sequence.'},
    )
    compress_method: Optional[str] = field(
        default='avg',
        metadata={'help': 'Specify the compression method for the sequence length.'},
    )
    attn_type: Optional[str] = field(
        default=None,
        metadata={'help': "Specify attn type"},
    )
    img_emb_down_sample_ratio: Optional[int] = field(
        default=None,
        metadata={'help': 'Set the desired down-sampling ratio for the image embedding for ablation. '
                          'Not that the true ratio is its inverse number, Default is None.'},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internvl_zh', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    dynamic_max_patch: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic max patch size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    min_num_frame: Optional[int] = field(
        default=4,
        metadata={'help': 'The minimum number of frames. Default is 4.'},
    )
    max_num_frame: Optional[int] = field(
        default=20,
        metadata={'help': 'The maximum number of frames. Default is 20.'},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )
    use_packed_ds: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for training. Default is False.'},
    )
    num_images_expected: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of images per packed sample. Default is 12.'},
    )
    max_packed_tokens: Optional[int] = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: Optional[int] = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: Optional[int] = field(
        default=1000,
        metadata={'help': 'The log frequence of the packed dataset. Default is 1000.'},
    )
    strict_mode: Optional[bool] = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: Optional[str] = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is `token`'},
    )
    loss_reduction_all_gather: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to all gahter when loss reduction. Default is False'},
    )
    scale: Optional[float] = field(
        default=25.0,
        metadata={'help': 'layer scale lr scale'}
    )
    final_size: Optional[int] = field(
        default=100,
        metadata={'help': 'compressed final_size'}
    )
    rope_pos_id_version: Optional[str] = field(
        default='default',
        metadata={'help': 'version for get_rope_pos_id'},
    )
    rope_pos_id_stride: Optional[int] = field(
        default=None,
        metadata={'help': 'stride for the version v4 of get_rope_pos_id'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            template_name,
            meta,
            tokenizer,
            tcs_loader,
            ds_name,
            num_image_token,
            image_size=224,
            is_train=True,
            pad2square=False,
            group_by_length=False,
            dynamic_image_size=False,
            dynamic_max_patch=False,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            max_num_frame=20,
            min_num_frame=4,
            sampling_method='rand',
            repeat_time=1,
            normalize_type='imagenet',
            # hyperparameters for packed training
            use_packed_ds=False,
            data_rank=0,
            data_world_size=1,
            distributed_mode=False,
            force_shuffle=False,
            random_seed=0,
            group_list=None,
            rope_pos_id_version='default',
            rope_pos_id_stride=None,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        self.rope_pos_id_version = rope_pos_id_version
        self.rope_pos_id_stride = rope_pos_id_stride
        print(f'[Dataset] num_image_token: {num_image_token}')
        print(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        print(f'[Dataset] use_thumbnail: {use_thumbnail}')
        print(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')
        print(f'[Dataset] rope_pos_id_version: {rope_pos_id_version}')
        print(f'[Dataset] rope_pos_id_stride: {rope_pos_id_stride}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # used for quick resume
        self._state_dict = {}
        print('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        total_ranks = torch.distributed.get_world_size()
        self.total_ranks = total_ranks
        current_rank = torch.distributed.get_rank()
        basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
        data_dir = os.path.join(os.path.dirname(meta['annotation']), basename)
        data_dir = data_dir.replace('metas/', 'metas/cache/')
        os.makedirs(data_dir, exist_ok=True)
        temp_path = os.path.join(data_dir, f'{basename}_{current_rank}_of_{total_ranks}.jsonl')

        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                self.raw_data = f.readlines()
        else:
            with open(meta['annotation'], 'r') as f:
                self.raw_data = f.readlines()
            if repeat_time < 1:
                self.raw_data = random.sample(self.raw_data, int(len(self.raw_data) * repeat_time))
            elif repeat_time > 1:
                repeat_time_int = int(repeat_time)
                self.raw_data = self.raw_data * repeat_time_int + random.sample(self.raw_data,int(len(self.raw_data) * (repeat_time - repeat_time_int)))

            total_lines = len(self.raw_data)
            print(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
            if group_list is None:
                lines_per_rank = total_lines // total_ranks
                lines_per_rank = max(1, lines_per_rank)
                start_line = lines_per_rank * current_rank
                end_line = start_line + lines_per_rank
                self.raw_data = self.raw_data[start_line:end_line]
                with open(temp_path, 'w') as f:
                    f.writelines(self.raw_data)
            else:
                for group_idx, group in enumerate(group_list):
                    if type(group) == torch.distributed.distributed_c10d.ProcessGroup:
                        print(group_idx, 'rank', dist.get_rank())
                        # assert type(group)==torch.distributed.distributed_c10d.ProcessGroup
                        break
                lines_per_group = total_lines // len(group_list)
                lines_per_group = max(1, lines_per_group)
                start_line = lines_per_group * group_idx
                end_line = start_line + lines_per_group
                self.raw_data = self.raw_data[start_line:end_line]
        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        gc.collect()
        self.root = meta.get('root', '')
        if self.root is None:
            self.root = ''
        if self.root != '' and not self.root.endswith('/') and not self.root.endswith(':'):
            self.root += '/'
        self.root2 = meta.get('root2', None)
        self.http_root = meta.get('http_root', None)

        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.dynamic_max_patch = dynamic_max_patch
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        self.local_sample_count = 0  # count samples
        gc.collect()

    def __len__(self):
        if not self.use_packed_ds:
            return len(self.raw_data) * self.total_ranks
        else:
            return len(self.raw_data)

    def encode_hash_sha256(self, web_url):
        hash_object = hashlib.sha256(web_url.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

    def multi_modal_get_item(self, data_item):
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        if data_item['image'].startswith('s3://'):
            image_path = self.root + data_item['image']
        elif data_item['image'].startswith('http') or data_item['image'].startswith('HTTP'):
            image_path = self.http_root + self.encode_hash_sha256(data_item['image'])
        else:
            image_path = os.path.join(self.root, data_item['image'])

        num_tiles = []
        all_boxes = []
        image_list = []

        if self.tcs_loader is not None and 's3:' in image_path:
            image = self.tcs_loader(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
        orig_size = image.size
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type
        )
        if self.dynamic_image_size:
            images, boxes = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                               image_size=self.image_size, use_thumbnail=self.use_thumbnail,
                                               return_box=True)
            num_tiles.append(len(images))
            all_boxes.append(boxes)
            image_list.append(images)
        else:
            images = [image]
            image_list.append([image])
            num_tiles.append(1)
            all_boxes.append([(0, 0, orig_size[0], orig_size[1]), ])

        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name.startswith('internlm2'):
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item['conversations'])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name
        )
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][
                    0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        rope_pos_id = self.get_rope_pos_id(ret, num_tiles, torch.float32, self.rope_pos_id_version, position_ids[0])

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=rope_pos_id,
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long))
        return ret

    def get_rope_pos_id(self, ret, num_tiles, dtype, rope_pos_id_version='default', position_id=None):
        image_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        rope_pos_id_list = []

        input_ids_0 = ret['input_ids'][0]
        attention_mask_0 = ret['attention_mask'][0]
        image_start_token_id_idxs = torch.where(input_ids_0 == image_start_token_id)[0]
        image_end_token_id_idxs = torch.where(input_ids_0 == image_end_token_id)[0]

        last_record_pos_id = -1
        start_index = 0

        for i in range(len(image_start_token_id_idxs)):
            num_tile = num_tiles[i]

            rope_pos_id_pre = attention_mask_0[start_index:image_start_token_id_idxs[i] + 1].long().cumsum(-1) - 1 + (
                        last_record_pos_id + 1)
            rope_pos_id_pre.masked_fill_(attention_mask_0[start_index:image_start_token_id_idxs[i] + 1] == 0, 1)
            rope_pos_id_list.append(rope_pos_id_pre)

            last_record_pos_id = rope_pos_id_pre[-1].long()

            if rope_pos_id_version == 'v2pe_fix':
                assert self.rope_pos_id_stride is not None, 'when rope_pos_id_version is fix, self.rope_pos_id_stride should not be None'
                small_stride = self.rope_pos_id_stride / self.num_image_token
                split_img_id_idxs = torch.arange(last_record_pos_id, last_record_pos_id + small_stride * (
                            self.num_image_token * num_tile + 1), small_stride)[1:].to(dtype=dtype)
                rope_pos_id_list.append(split_img_id_idxs)
                last_record_pos_id = torch.ceil(split_img_id_idxs[-1]).long()
            elif rope_pos_id_version == 'v2pe_rnd':
                random_from = [1, 2, 4, 8, 16, 32, 64, 128, 256]
                rope_pos_id_stride = random.choice(random_from)
                small_stride = rope_pos_id_stride / self.num_image_token
                split_img_id_idxs = torch.arange(last_record_pos_id, last_record_pos_id + small_stride * (
                            self.num_image_token * num_tile + 1), small_stride)[1:].to(dtype=dtype)
                rope_pos_id_list.append(split_img_id_idxs)
                last_record_pos_id = torch.ceil(split_img_id_idxs[-1]).long()

            elif rope_pos_id_version == 'default':
                split_img_id_idxs = torch.linspace(last_record_pos_id,
                                                   last_record_pos_id + (num_tile - 1) * self.num_image_token,
                                                   (num_tile - 1) * self.num_image_token + 1)[1:].to(dtype=dtype)
                rope_pos_id_list.append(split_img_id_idxs)
                thumbnail_id_idxs = torch.linspace(last_record_pos_id + (num_tile - 1) * self.num_image_token,
                                                   last_record_pos_id + num_tile * self.num_image_token,
                                                   self.num_image_token + 1)[1:].to(dtype=dtype)
                rope_pos_id_list.append(thumbnail_id_idxs)
                last_record_pos_id = (last_record_pos_id + num_tile * self.num_image_token).long()
            else:
                raise NotImplementedError(f'not implement for {rope_pos_id_version}')

            start_index = image_start_token_id_idxs[i] + num_tile * self.num_image_token + 1
            assert input_ids_0[start_index] == image_end_token_id
            assert start_index == image_end_token_id_idxs[i]

        assert image_end_token_id_idxs[-1] == start_index
        rope_pos_id_pre = attention_mask_0[start_index:].long().cumsum(-1) - 1 + (last_record_pos_id + 1)
        rope_pos_id_pre.masked_fill_(attention_mask_0[start_index:] == 0, 1)
        rope_pos_id_list.append(rope_pos_id_pre)

        rope_pos_id = torch.cat(rope_pos_id_list).to(dtype=dtype)
        if rope_pos_id_version == 'default':
            rope_pos_id = rope_pos_id.long()
            assert torch.equal(rope_pos_id, position_id)
            assert torch.allclose(rope_pos_id, position_id, atol=1e-32)

        assert rope_pos_id.shape == input_ids_0.shape

        return list(rope_pos_id.numpy())

    def multi_modal_multi_image_get_item(self, data_item):
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type
        )
        images, num_tiles = [], []
        num_image = len(data_item['image'])
        all_boxes = []
        image_list = []
        for image_path in data_item['image']:
            if image_path.startswith('s3://'):
                image_path = self.root + image_path
            elif self.root2 is not None:
                image_path = os.path.join(self.root2, image_path)
            elif image_path.startswith('http') or image_path.startswith('HTTP'):
                image_path = self.http_root + self.encode_hash_sha256(image_path)
            elif image_path.startswith('/mnt'):
                pass
            else:
                image_path = os.path.join(self.root, image_path)

            if self.tcs_loader is not None and ('s3:' in image_path):
                image = self.tcs_loader(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            orig_size = image.size
            if self.dynamic_image_size:

                image, boxes = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=self.max_dynamic_patch // num_image if self.dynamic_max_patch else self.max_dynamic_patch,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                    return_box=True
                )
                images += image
                image_list.append(image)
                all_boxes.append(boxes)
                num_tiles.append(len(image))

            else:
                images.append(image)
                image_list.append([image])
                num_tiles.append(1)
                all_boxes.append([(0, 0, orig_size[0], orig_size[1]), ])

        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name.startswith('internlm2'):
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess

        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item['conversations'])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_image
        )
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        rope_pos_id = self.get_rope_pos_id(ret, num_tiles, torch.float32, self.rope_pos_id_version, position_ids[0])

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=rope_pos_id,
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long))
        return ret

    def video_get_item(self, data_item):
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        video_file = data_item['video']
        # video_path = os.path.join(self.root, video_file)
        video_path = self.root + video_file
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type
        )

        clip = data_item.get('clip', None)
        raw_image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=clip
        )

        images, num_tiles = [], []
        num_image = len(raw_image_list)
        all_boxes = []
        image_list = []

        # for video, we don't use dynamic image process
        assert self.dynamic_image_size == False

        for raw_image in raw_image_list:
            orig_size = raw_image.size
            if self.dynamic_image_size:
                image, boxes = dynamic_preprocess(
                    raw_image,
                    min_num=self.min_dynamic_patch,
                    max_num=self.max_dynamic_patch // num_image if self.dynamic_max_patch else self.max_dynamic_patch,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                    return_box=True
                )
                images += image
                image_list.append(image)
                all_boxes.append(boxes)
                num_tiles.append(len(image))

            else:
                image = raw_image
                images.append(image)
                image_list.append([image])
                num_tiles.append(1)
                all_boxes.append([(0, 0, orig_size[0], orig_size[1]), ])

        special_tokens = '\n'.join(['Frame{}:<image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name.startswith('internlm2'):
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess

        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item['conversations'])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_image
        )
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        rope_pos_id = self.get_rope_pos_id(ret, num_tiles, torch.float32, self.rope_pos_id_version, position_ids[0])

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=rope_pos_id,
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def pure_text_get_item(self, data_item):
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        all_boxes = []
        image_list = []
        num_tiles = []

        images, boxes = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail,
                                           return_box=True)

        all_boxes.append(boxes)
        num_tiles.append(len(images))
        image_list.append(images)

        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name.startswith('internlm2'):
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        rope_pos_id = position_ids[0]

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=rope_pos_id,
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long))
        return ret

    def _enable_worker_distributed(self):
        if (
                self.distributed_mode
                and not self.worker_distributed
                and self.worker_id is not None
        ):
            self.worker_distributed = True
            num_worker_per_rank = self.num_workers // self.total_ranks
            self.raw_data = self.raw_data[self.worker_id % num_worker_per_rank::num_worker_per_rank]
            gc.collect()
            print(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if not self.use_packed_ds:
                i = i % len(self.raw_data)
            else:
                raise NotImplementedError

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load images, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0
        samples_processed = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']
            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            print(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            samples_processed += 1

            self._update_samples_processed_file(samples_processed)

            yield self[i]

    def _update_samples_processed_file(self, samples_processed):
        folder_name = "samples_num"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        filename = os.path.join(folder_name,
                                f"worker_{self.worker_id}_rank_{self.data_rank}_{self.ds_name}_samples_processed.txt")

        with open(filename, 'w') as f:
            f.write(
                f"WorkerID: {self.worker_id}, Rank: {self.data_rank}, Dataset: {self.ds_name}, Samples Processed: {samples_processed}\n")


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    dynamic_max_patch=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=6,
    normalize_type='imagenet',
    group_list=None,
    rope_pos_id_version='default',
    rope_pos_id_stride=None,
    min_num_frame=4,
    max_num_frame=20,
):
    datasets = []
    lengths = []
    # TODO: support TP or PP
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            print(f'In {ds_name}, max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch

        if 'dynamic_image_size' in ds_collections[ds_name]:
            cur_dynamic_image_size = ds_collections[ds_name]['dynamic_image_size']
            print(f'In {ds_name}, dynamic_image_size is set to {max_num} according to the meta file')
        else:
            cur_dynamic_image_size = dynamic_image_size

        dataset = LazySupervisedDataset(
            data_args.conv_style,
            ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=cur_dynamic_image_size,
            dynamic_max_patch=dynamic_max_patch,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
            group_list=group_list,
            rope_pos_id_version=rope_pos_id_version,
            rope_pos_id_stride=rope_pos_id_stride,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
        )
        print(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
            # sample_counter=sample_counter,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    """
    Calculate the weight based on the input length and loss reduction method.

    Args:
        x (int or float): The input length.
        loss_reduction (str): The method for loss reduction. It can be one of the following:
            - 'token': Returns a constant weight of 1.
            - 'sample': Returns the inverse of the input length (1/x).
            - 'square': Returns the inverse of the square root of the input length (1/sqrt(x)).

    Returns:
        float: The calculated weight based on the specified loss reduction method.

    Raises:
        NotImplementedError: If the specified loss reduction method is not supported.
    """
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def main():

    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.use_packed_ds = data_args.use_packed_ds

    use_chunkTrainer = False
    if model_args.attn_type:
        if model_args.attn_type != 'packed':
            use_chunkTrainer = True
    if use_chunkTrainer:
        # create new groups, chunk_num gpus one group
        num_groups = dist.get_world_size() // model_args.chunk_num
        group_list = []
        for i in range(num_groups):
            group_list.append(
                dist.new_group(ranks=list(range(i * model_args.chunk_num, (i + 1) * model_args.chunk_num))))

        dist.barrier()
        internvl.train.trainer_monkey_patch.SCALE = data_args.scale
        internvl.model.internlm2.FINAL_SIZE = data_args.final_size

    else:
        group_list = None
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')
    logger.info(f'Model parameters {model_args}')
    logger.info(f'Data parameters {data_args}')

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    set_seed(training_args.seed)

    if model_args.tokenizer_path:
        tokenizer_path = model_args.tokenizer_path
    else:
        tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False)

    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if os.environ.get('no_tcs', False):
        tcs_loader = None
    else:
        tcs_loader = TCSLoader('~/petreloss_zy.conf') if has_tcs_loader else None

    if data_args.use_packed_ds or (model_args.attn_type is not None):
        if model_args.attn_type is None:
            model_args.attn_type = 'packed'

        replace_internlm2_attention_class(model_args.attn_type)
        replace_qwen2_attention_class()
        replace_llama_attention_class()
    elif model_args.attn_type is None:
        model_args.attn_type = 'packed'
    elif model_args.attn_type is None:
        model_args.attn_type = 'packed'

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        if os.environ.get('DEBUG_FLAG', False):
            config.llm_config.num_hidden_layers = 2
            config.vision_config.num_hidden_layers = 2

        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.dynamic_max_patch = data_args.dynamic_max_patch
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        config.rope_pos_id_version = data_args.rope_pos_id_version
        config.rope_pos_id_stride = data_args.rope_pos_id_stride
        config.img_emb_down_sample_ratio = model_args.img_emb_down_sample_ratio
        config.min_num_frame = data_args.min_num_frame
        config.max_num_frame = data_args.max_num_frame

        if os.environ.get('DEBUG_FLAG', False):
            model = InternVLChatModel(config)
        else:
            model = InternVLChatModel.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                config=config)
        model.compress_seq = model_args.compress_seq
        model.chunk_num = model_args.chunk_num
        model.interaction = model_args.interaction
        model.group_list = group_list
        model.attn_type = model_args.attn_type
        model.language_model.model.init_interactions(model_args.compress_seq, model_args.fuse_method, model_args.compress_method)
        model.init_embed()

    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        if os.environ.get('DEBUG_FLAG', False):
            vision_config.num_hidden_layers = 2
            vision_model = InternVisionModel(vision_config)
        else:
            vision_model = InternVisionModel.from_pretrained(
                model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        if os.environ.get('DEBUG_FLAG', False):
            llm_config.num_hidden_layers = 2
            llm = model_type(llm_config)

        else:
            llm = model_type.from_pretrained(
                model_args.llm_path,
                torch_dtype=torch.bfloat16,
                config=llm_config,
                trust_remote_code=True)

        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(),
            llm_config.to_dict(),
            downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square,
            template=data_args.conv_style,
            select_layer=model_args.vision_select_layer,
            dynamic_image_size=data_args.dynamic_image_size,
            dynamic_max_patch=data_args.dynamic_max_patch,
            use_thumbnail=data_args.use_thumbnail,
            ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
            min_num_frame=data_args.min_num_frame,
            max_num_frame=data_args.max_num_frame,
            compress_seq=model_args.compress_seq,
            attn_type=model_args.attn_type,
            group_list=group_list,
            chunk_num=model_args.chunk_num,
            interaction=model_args.interaction,
            rope_pos_id_version=data_args.rope_pos_id_version,
            rope_pos_id_stride=data_args.rope_pos_id_stride,
            img_emb_down_sample_ratio=model_args.img_emb_down_sample_ratio)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
        model.language_model.model.init_interactions(model_args.compress_seq, model_args.fuse_method,
                                                     model_args.compress_method)
        model.init_embed()
    model.img_context_token_id = img_context_token_id
    model.neftune_alpha = data_args.neftune_alpha

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if model.img_emb_down_sample_ratio is not None:
        model.num_image_token = int(model.num_image_token / model.img_emb_down_sample_ratio)

    print(f'the model.num_image_token is set to {model.num_image_token}')
    print(f'{model.language_model.config.rope_scaling=}')

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    logger.info("\n\n\nBuilding dataset!\n\n\n")
    rope_pos_id_version = getattr(data_args, 'rope_pos_id_version', 'default')
    logger.info(f'rope_pos_id_version: {rope_pos_id_version}')
    rope_pos_id_stride = getattr(data_args, 'rope_pos_id_stride', None)
    logger.info(f'rope_pos_id_stride: {rope_pos_id_stride}')
    img_emb_down_sample_ratio = getattr(model_args, 'img_emb_down_sample_ratio', None)
    logger.info(f'img_emb_down_sample_ratio: {img_emb_down_sample_ratio}')

    logger.info(f'min_num_frame: {data_args.min_num_frame}, max_num_frame: {data_args.max_num_frame}')
    train_dataset = build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        dynamic_max_patch=data_args.dynamic_max_patch,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
        normalize_type=data_args.normalize_type,
        group_list=group_list,
        rope_pos_id_version=rope_pos_id_version,
        rope_pos_id_stride=rope_pos_id_stride)
    logger.info("\n\n\nDataset build done!\n\n\n")

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    set_seed(training_args.seed)

    if model_args.use_custom_trainer:
        replace_create_optimizer()

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator
    if use_chunkTrainer:
        trainer = chunkTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=collator,
            chunk_num=model_args.chunk_num,
            group_list=group_list,
        )
    else:
        # training_args.dataloader_num_workers = 0
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=collator)

    logger.info("\n\n\nTraining!\n\n\n")
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    # destroy process groups
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
