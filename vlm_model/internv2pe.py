import re
import torch
import math
from functools import partial
from .model_utils import LLM, truncate_images

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "internvl_v2pe"))
from .internvl_v2pe.internvl.train.dataset import build_transform, dynamic_preprocess

def load_image(image_file, dynamic_image_size=True, input_size=448, max_num=12, return_additional_info=False):
    if not return_additional_info:
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(is_train=False, input_size=input_size)
        if dynamic_image_size:
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    else:
        image = Image.open(image_file).convert('RGB')
        orig_size = image.size

        transform = build_transform(is_train=False, input_size=input_size)
        if dynamic_image_size:
            images, boxes = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num, return_box=True)
        else:
            images = [image]
            boxes = [(0,0,orig_size[0],orig_size[1]), ]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values, images, boxes, orig_size

def split_model_3(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    gpu0_rate = 0.5
    num_layers_per_gpu = math.ceil(num_layers / (world_size - gpu0_rate))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * gpu0_rate)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def split_model_2_5(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    model_name = model_name.split("/")[-1]
    if "-AWQ" in model_name:
        model_name = model_name.replace("-AWQ", "")
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def split_model_2(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    model_name = model_name.split("/")[-1]
    if "v2pe" in model_name.lower():
        model_name = 'InternVL2-2B'
    if world_size <= 2:
        return "auto"
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


@torch.no_grad()
def self_revised_generate(
        self,
        pixel_values = None,
        input_ids = None,
        attention_mask = None,
        visual_features = None,
        generation_config = None,
        output_hidden_states = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
        input_ids = input_ids.reshape(B, N)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    prefill_output = self.language_model.model(inputs_embeds=input_embeds[..., :-1, :], attention_mask=attention_mask[..., :-1])
    past_key_values = prefill_output.past_key_values
    # del prefill_output
    # torch.cuda.empty_cache()

    generate_kwargs["past_key_values"] = past_key_values

    outputs = self.language_model.generate(
        input_ids=input_ids,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    # TODO only support batch_size == 1
    outputs = outputs[:, input_ids.shape[1]:]

    return outputs


def extract_feature_batch(self, pixel_values, vision_batch_size=32):
    total_samples = pixel_values.shape[0]
    all_vit_embeds = []
    for i in range(0, total_samples, vision_batch_size):
        batch_pixel_values = pixel_values[i:i + vision_batch_size]
        if self.select_layer == -1:
            batch_output = self.vision_model(
                pixel_values=batch_pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            batch_output = self.vision_model(
                pixel_values=batch_pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        batch_vit_embeds = batch_output[:, 1:, :]
        # pixel shuffle + mlp
        h = w = int(batch_vit_embeds.shape[1] ** 0.5)
        batch_vit_embeds = batch_vit_embeds.reshape(batch_vit_embeds.shape[0], h, w, -1)
        batch_vit_embeds = self.pixel_shuffle(batch_vit_embeds, scale_factor=self.downsample_ratio)
        batch_vit_embeds = batch_vit_embeds.reshape(batch_vit_embeds.shape[0], -1, batch_vit_embeds.shape[-1])
        batch_vit_embeds = self.mlp1(batch_vit_embeds)

        all_vit_embeds.append(batch_vit_embeds)
    vit_embeds = torch.cat(all_vit_embeds, dim=0)

    return vit_embeds


# FIXME: ImportError: When runing InternVL2.5-26B, you may see a bug:
# FIXME: fused_layer_norm_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
# FIXME: Solution: pip uninstall apex (see https://github.com/huggingface/diffusers/issues/8624)
class InternV2PEModel(LLM):
    def __init__(
            self,
            model_name,
            temperature=0.9,
            top_p=0.9,
            max_length=32768,
            generation_max_length=2048,
            generation_min_length=0,
            do_sample=True,
            stop_newline=False,
            use_chat_template=False,
            **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        model_kwargs = {}
        model_kwargs["offload_state_dict"] = kwargs.get("offload_state_dict", False)
        model_kwargs["attn_implementation"] = kwargs.get("attn_implementation", "flash_attention_2")
        self.v2pe_step = kwargs.get("v2pe_step", 64)
        self.vision_batch_size = kwargs.get("vision_batch_size", 32)
        self.do_prefill = kwargs.get("do_prefill", False)
        self.image_resize = kwargs.get("image_resize", None)
        self.max_image_num = kwargs.get("max_image_num", None)
        if self.image_resize is not None:
            self.num_crops = max(int(4 * self.image_resize * self.image_resize), 1)
        else:
            self.num_crops = 4
        self.max_length = max_length
        self.processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        tokenizer = self.processor
        tokenizer.model_max_length = 256000
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.dtype = kwargs.get("torch_dtype", torch.bfloat16)
        device_map = split_model_2(model_name)

        # use int8
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        if self.load_in_8bit:
            print("!!!!!!NOTICE: Using INT8 to load the model!!!!!!")
            model_kwargs["load_in_8bit"] = True

        from .internvl_v2pe.internvl.model.internvl_chat import InternVLChatModel
        from .internvl_v2pe.internvl.model.internvl_chat import InternVLChatConfig
        config = InternVLChatConfig.from_pretrained(model_name, trust_remote_code=True)
        config.llm_config.use_cache = True

        self.model = InternVLChatModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=True,
            # **model_kwargs
        )

        import types
        self.model.extract_feature = types.MethodType(
            partial(extract_feature_batch, vision_batch_size=self.vision_batch_size), self.model)

        if self.do_prefill:
            import types
            self.model.generate = types.MethodType(self_revised_generate, self.model)

        if kwargs.get("torch_compile", True):
            self.model = torch.compile(self.model)


        # use the default if possible, append if necessary
        self.device = self.model.device

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        all_pixel_values = []
        all_num_patches = []
        all_boxes = []
        all_orig_sizes = []
        all_images = []
        all_num_tile = []

        for image_path in image_list:
            pixel_values, images, boxes, orig_size = load_image(
                image_path,
                dynamic_image_size = True,
                max_num=self.num_crops,
                return_additional_info=True
            )
            pixel_values = pixel_values.to(self.dtype)

            # 收集信息
            all_pixel_values.append(pixel_values)
            all_num_patches.append(pixel_values.size(0))
            all_boxes.append(boxes)
            all_orig_sizes.append(orig_size)
            all_images.append(images)
            all_num_tile.append(len(images))

        pixel_values = torch.cat(all_pixel_values, dim=0)

        return {
            "text": text,
            "pixel_values": pixel_values,
            "num_patches_list": all_num_patches,
            "all_boxes": all_boxes,
            "orig_sizes": all_orig_sizes,
            "image_list": all_images,
            "num_tiles": all_num_tile
        }

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.device)

        generation_config = dict(max_new_tokens=self.generation_max_length,
                                 min_new_tokens=self.generation_min_length,
                                 do_sample=self.do_sample,
                                 temperature=self.temperature if self.do_sample else None,
                                 top_p=self.top_p if self.do_sample else None,
                                 top_k=None,
                                 pad_token_id=self.processor.pad_token_id,
                                 output_scores=False)

        text, history = self.model.chat(self.processor,
                                        inputs["pixel_values"],
                                        inputs["text"],
                                        generation_config,
                                        num_patches_list=inputs["num_patches_list"],
                                        num_tiles=[inputs["num_tiles"], ],
                                        all_boxes=[inputs["all_boxes"], ],
                                        orig_sizes=[inputs["orig_sizes"], ],
                                        image_list=[inputs["image_list"], ],
                                        history=None,
                                        return_history=True,
                                        rope_pos_id_version="v2pe_fix",
                                        rope_pos_id_stride=self.v2pe_step)
        # print(f"using {self.v2pe_step} stride")

        save_prompt = inputs["text"]
        if len(save_prompt) > 6000:
            save_prompt = save_prompt[:2000] + " <skip> " + save_prompt[-2000:]
        return {
            "output": text,
            "input_len": -1,
            "output_len": -1,
            "input_text": save_prompt,
        }