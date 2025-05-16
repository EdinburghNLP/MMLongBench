import re
import torch
from functools import partial

from .model_utils import LLM, truncate_images

from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def get_image_features_batch(self, pixel_values: torch.Tensor, vision_batch_size=32) -> torch.Tensor:
    """
    Projects the last hidden state from the vision model into language model space.

    Args:
        pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
           The tensors corresponding to the input images.
        vision_batch_size (int): Number of images to process in each batch.
    Returns:
        image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
    """
    all_image_features = []
    num_images = pixel_values.shape[0]

    for start_idx in range(0, num_images, vision_batch_size):
        end_idx = min(start_idx + vision_batch_size, num_images)
        batch_pixel_values = pixel_values[start_idx: end_idx]
        batch_vision_outputs = self.vision_tower(pixel_values=batch_pixel_values).last_hidden_state
        batch_image_features = self.multi_modal_projector(batch_vision_outputs)
        all_image_features.append(batch_image_features)
    all_image_features = torch.cat(all_image_features, dim=0)

    return all_image_features


class Gemma3VLModel(LLM):
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
        self.max_image_num = kwargs.get("max_image_num", None)
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

        import types
        self.model.get_image_features = types.MethodType(
            partial(get_image_features_batch, vision_batch_size=self.vision_batch_size), self.model)

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
                    "type": "image",
                    "url": image_list[image_idx]
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
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        messages = self.format_chat(text, image_list, data["system_template"])


        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, continue_final_message=True,
            return_dict=True, return_tensors="pt"
        )

        return inputs


    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs = inputs.to(self.model.device)
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