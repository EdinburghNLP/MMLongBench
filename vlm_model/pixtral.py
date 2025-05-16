import re
import torch
from functools import partial
from .model_utils import LLM

import types
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


# As PixtralVisionTower doesn't support flash attention or sdpa,
# we need to split the images into small batches to fit the GPU memory
def get_image_features_in_batches(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer,
        vision_feature_select_strategy: str,
        vision_batch_size=4,
        **kwargs,
):
    """
    Obtains image last hidden states from the vision tower and apply multimodal projection.

    Args:
        pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
           The tensors corresponding to the input images.
        vision_feature_layer (`Union[int, List[int]]`):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        vision_feature_select_strategy (`str`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`
    Returns:
        image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
    """
    if vision_feature_select_strategy not in ["full"]: # Pixtral doesn't support default actually
        raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    image_sizes = kwargs["image_sizes"]

    total_images = pixel_values.shape[0]
    num_batches = (total_images + vision_batch_size - 1) // vision_batch_size

    assert isinstance(vision_feature_layer, int), "LlavaForConditionalGeneration.get_image_features doesn't support getting image features from multiple layers"

    # the implementation of vision_tower (PixtralVisionModel) has O(N^2) memory complexity, every inefficient
    # N (N, H, W, C) (N * H * W, C)
    # (N * H * W) * (N * H * W)
    all_image_features = []
    for i in range(num_batches):
        start_idx = i * vision_batch_size
        end_idx = min((i + 1) * vision_batch_size, total_images)
        batch_pixel_values = pixel_values[start_idx:end_idx]
        batch_image_sizes = image_sizes[start_idx:end_idx]
        image_outputs = self.vision_tower(batch_pixel_values, output_hidden_states=False, image_sizes=batch_image_sizes)
        image_outputs = image_outputs.last_hidden_state
        if vision_feature_select_strategy == "default":
            image_outputs = image_outputs[:, 1:]
        all_image_features.append(image_outputs)
    all_image_features = torch.cat(all_image_features, dim=-2)


    image_features = self.multi_modal_projector(all_image_features)
    return image_features


class PixtralModel(LLM):
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
        model_kwargs["attn_implementation"] = {"text_config": "flash_attention_2", "vision_config": "eager"}
        self.image_resize = kwargs.get("image_resize", None)
        self.vision_batch_size = kwargs.get("vision_batch_size", 6)

        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.dtype = kwargs.get("torch_dtype", torch.bfloat16)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

        self.model.get_image_features = types.MethodType(
            partial(get_image_features_in_batches, vision_batch_size=self.vision_batch_size), self.model)
        print(f"Vision encoder will encode {self.vision_batch_size} images each time.")

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

    def format_chat(self, text, image_list, system_prompt):
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                new_content.append({
                    "type": "image"
                })
                image_idx += 1
            else:
                new_content.append({
                    "type": "text",
                    "content": c
                })
        assert image_idx == len(image_list)
        messages = [{"role": "user", "content": new_content},
                    {"role": "assistant", "content": system_prompt}]
        return messages


    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        messages = self.format_chat(text, image_list, data["system_template"])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        image_list = [Image.open(image).convert('RGB') for image in image_list]

        if self.image_resize is not None:
            from .model_utils import resize_image
            image_list = resize_image(image_list, self.image_resize)

        inputs = self.processor(
            text=text,
            images=image_list,
            padding=True,
            return_tensors="pt",
        )
        return inputs


    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs = inputs.to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        input_len = inputs.input_ids.size(1)
        # if hasattr(self.model, "model") and self.do_prefill:
        #     # prefill without calculating the logits (save memory for large vocab models)
        #     prefill = self.model.model(input_ids=inputs.input_ids[..., :-1],
        #                                attention_mask=inputs.attention_mask[..., :-1])
        #     past_key_values = prefill.past_key_values
        #     inputs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
        #               "past_key_values": past_key_values}
        #     if past_key_values is None:
        #         self.do_prefill = False
        #         logger.warning("past key values is None, not able to prefill with KVs, disabling...")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            top_k = None,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        text = self.processor.decode(outputs['sequences'][0, input_len:], skip_special_tokens=True)

        if input_len > 1500:
            save_prompt = self.processor.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.processor.decode(
                inputs["input_ids"][0][-500:])
        else:
            save_prompt = self.processor.decode(inputs["input_ids"][0])
        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs['sequences'].size(1) - input_len,
            "input_text": save_prompt,
        }