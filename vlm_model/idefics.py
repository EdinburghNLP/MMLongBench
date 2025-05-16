import re
import torch
from functools import partial
from typing import List, Optional, Tuple, Union

from .model_utils import LLM

from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


def forward_image_batch(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        vision_batch_size = 32,
) -> Union[Tuple, BaseModelOutput]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size = pixel_values.size(0)
    if patch_attention_mask is None:
        patch_size = self.config.patch_size
        patch_attention_mask = torch.ones(
            (
                batch_size,
                pixel_values.size(2) // patch_size,
                pixel_values.size(3) // patch_size,
            )
        )
        patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

    hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

    patch_attention_mask = patch_attention_mask.view(batch_size, -1)
    # The call to `_upad_input` in `_flash_attention_forward` is expensive
    # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
    # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
    if not torch.any(~patch_attention_mask):
        patch_attention_mask = None
    elif not self._use_flash_attention_2:
        patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

    all_hidden_states = []
    total_images = hidden_states.shape[0]
    for i in range(0, total_images, vision_batch_size):
        start_idx = i
        end_idx = min(i + vision_batch_size, total_images)

        current_hidden_states = hidden_states[start_idx:end_idx]
        current_attention_mask = None
        if patch_attention_mask is not None:
            current_attention_mask = patch_attention_mask[start_idx:end_idx]

        current_encoder_outputs = self.encoder(
            inputs_embeds=current_hidden_states,
            attention_mask=current_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        current_last_hidden_state = current_encoder_outputs[0]
        all_hidden_states.append(current_last_hidden_state)

    all_hidden_state = torch.cat(all_hidden_states, dim=0)
    last_hidden_state = all_hidden_state
    last_hidden_state = self.post_layernorm(last_hidden_state)

    return BaseModelOutput(
        last_hidden_state=last_hidden_state,
    )


def forward_perceiver_batch(self, image_hidden_states, attention_mask, vision_batch_size=32):
    total_images = image_hidden_states.shape[0]
    all_processed_states = []
    for i in range(0, total_images, vision_batch_size):
        start_idx = i
        end_idx = min(i + vision_batch_size, total_images)
        current_hidden_states = image_hidden_states[start_idx:end_idx]

        current_hidden_states = self.modality_projection(current_hidden_states)

        current_attention_mask = None
        if attention_mask is not None:
            current_attention_mask = attention_mask[start_idx:end_idx]
        current_hidden_states = self.perceiver_resampler(
            context=current_hidden_states,
            attention_mask=current_attention_mask
        )

        all_processed_states.append(current_hidden_states)
    processed_image_hidden_states = torch.cat(all_processed_states, dim=0)
    return processed_image_hidden_states


# FIXME: Idefics2 has a bug of repeated generation, which is usually a problem of smaller models (like gpt2)
# FIXME: https://huggingface.co/HuggingFaceM4/idefics2-8b/discussions/80
class IdeficsModel(LLM):
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
        self.do_prefill = kwargs.get("do_prefill", False)
        self.image_resize = kwargs.get("image_resize", None)
        self.vision_batch_size = kwargs.get("vision_batch_size", 32)

        self.max_length = max_length

        processor_kwargs = {}
        if "idefics3" in model_name.lower():
            processor_kwargs["do_resize"] = False
            # Idefics3's dynamic tiling is special, different from InternVL and Phi. It resizes each images to the max size of max_crops (default:16).
            # For example, in icl, images smaller than 448x448 are resized to 1456x1456. 32k examples will produce 200K tokens
            # Thus, we turn max_crops off here. processor_kwargs["do_resize"] = False
            # Then, there is no tiles limit for each image,
            # we need to compromise the image size with image_resize=0.5 or do_image_splitting=False
            # 1. When testing natural images with 64K and 128K length, we want the num_crops=1, so do_image_splitting=False
            # 2. When testing text-rich images with 64K and 128K length, we want the num_crops=4 (while the default for Idefics3 is 16).
            # However, the num_crops is hard to set for Idefics3, so we use image_resize=0.5 instead.


        if "do_image_splitting" in kwargs:
            self.processor = AutoProcessor.from_pretrained(model_name,
                                                           use_fast=True,
                                                           do_image_splitting=kwargs["do_image_splitting"],
                                                           **processor_kwargs)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True,
                                                           **processor_kwargs)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

        if "idefics2" in model_name.lower():
            import types
            self.model.model.vision_model.forward = types.MethodType(
                partial(forward_image_batch, vision_batch_size=self.vision_batch_size), self.model.model.vision_model)
            self.model.model.connector.forward = types.MethodType(
                partial(forward_perceiver_batch, vision_batch_size=self.vision_batch_size), self.model.model.connector)

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
                    "text": c
                })
        assert image_idx == len(image_list)
        messages = [{"role": "user", "content": new_content},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": system_prompt}
                    ]}]
        return messages


    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        messages = self.format_chat(text, image_list, data["system_template"])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        image_list = [load_image(image) for image in image_list]

        if self.image_resize is not None:
            from .model_utils import resize_image
            image_list = resize_image(image_list, self.image_resize)

        inputs = self.processor(
            text=text,
            images=image_list,
            return_tensors="pt",
        )
        # print(inputs["pixel_values"].shape)
        return inputs


    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)
        if hasattr(self.model, "model") and self.do_prefill:
            # prefill without calculating the logits (save memory for large vocab models)
            prefill = self.model.model(input_ids=inputs.input_ids[..., :-1],
                                       attention_mask=inputs.attention_mask[..., :-1],
                                       pixel_values=inputs.pixel_values,
                                       pixel_attention_mask=inputs.pixel_attention_mask)
            past_key_values = prefill.past_key_values
            del prefill
            torch.cuda.empty_cache()
            inputs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
                      "pixel_values": inputs.pixel_values, "pixel_attention_mask": inputs.pixel_attention_mask,
                      "past_key_values": past_key_values}
            if past_key_values is None:
                self.do_prefill = False
                print("past key values is None, not able to prefill with KVs, disabling...")
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