import re
import torch
import math
from functools import partial
from .model_utils import LLM, truncate_images

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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
class InternVLModel(LLM):
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.dtype = kwargs.get("torch_dtype", torch.bfloat16)
        if "InternVL3-" in model_name:
            device_map = split_model_3(model_name)
        elif "InternVL2_5-" in model_name:
            device_map = split_model_2_5(model_name)
        elif "InternVL2-" in model_name:
            device_map = split_model_2(model_name)
        else:
            raise ValueError(f"Wrong InternVL model name {model_name}")

        # use int8
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        if self.load_in_8bit:
            print("!!!!!!NOTICE: Using INT8 to load the model!!!!!!")
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs
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
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
        if stop_newline:
            stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
            stop_token_ids = list(
                set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + stop_token_ids))
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in stop_token_ids:
                stop_token_ids.remove(tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids
        self.device = self.model.device

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        image_list = [load_image(image, max_num=self.num_crops).to(self.dtype) for image in image_list]
        num_patches_list = [image.size(0) for image in image_list]
        pixel_values = torch.cat(image_list, dim=0)

        return {"text": text, "pixel_values": pixel_values, "num_patches_list": num_patches_list}

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.device)

        generation_config = dict(max_new_tokens=self.generation_max_length,
                                 min_new_tokens=self.generation_min_length,
                                 do_sample=self.do_sample,
                                 temperature=self.temperature if self.do_sample else None,
                                 top_p=self.top_p if self.do_sample else None,
                                 top_k=None,
                                 eos_token_id=self.stop_token_ids,
                                 pad_token_id=self.processor.pad_token_id,
                                 output_scores=False)

        text, history = self.model.chat(self.processor,
                                        inputs["pixel_values"],
                                        inputs["text"],
                                        generation_config,
                                        num_patches_list=inputs["num_patches_list"],
                                        history=None,
                                        return_history=True)

        save_prompt = inputs["text"]
        if len(save_prompt) > 6000:
            save_prompt = save_prompt[:2000] + " <skip> " + save_prompt[-2000:]
        return {
            "output": text,
            "input_len": -1,
            "output_len": -1,
            "input_text": save_prompt,
        }