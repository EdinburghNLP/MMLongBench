import re
import torch
from .model_utils import LLM, truncate_images

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from transformers.cache_utils import DynamicCache

# FIXME: For both Phi-3-vision-128k-instruct and Phi-3.5-vision-instruct, you may get a bug about DynamicCache.
# FIXME: See https://github.com/huggingface/transformers/issues/36071 and revise the code.
# Here is a simple solution
# check get_max_length
if not hasattr(DynamicCache, 'get_max_length'):
    # set get_max_length as the alias of get_max_cache_shape
    DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
    print("Added get_max_length method to DynamicCache for backward compatibility")

class Phi3Model(LLM):
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
        self.max_image_num = kwargs.get("max_image_num", None)
        if self.image_resize is not None:
            num_crops = max(int(4 * self.image_resize * self.image_resize), 1)
        else:
            num_crops = 4
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name,
                                                       use_fast=True,
                                                       trust_remote_code=True,
                                                       num_crops=num_crops)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

        if kwargs.get("torch_compile", True):
            self.model = torch.compile(self.model)

        # use the default if possible, append if necessary
        stop_token_ids = [self.processor.tokenizer.eos_token_id, 32007]
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
                new_content.append(f"<|image_{image_idx + 1}|>")
                image_idx += 1
            else:
                new_content.append(c)
        assert image_idx == len(image_list)
        new_content = "".join(new_content)
        messages = [{"role": "user", "content": new_content},
                    {"role": "assistant", "content": system_prompt}]
        return messages


    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        messages = self.format_chat(text, image_list, data["system_template"])

        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        if text.endswith('<|endoftext|>'):
            text = text.rstrip('<|endoftext|>')
        image_list = [Image.open(image).convert('RGB') for image in image_list]
        inputs = self.processor(
            text=text,
            images=image_list,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def safe_decode(self, input_ids, skip_special_tokens=False):
        safe_ids = input_ids.clone()
        safe_ids[safe_ids < 0] = self.processor.tokenizer.pad_token_id
        return self.processor.tokenizer.decode(safe_ids, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)
        if hasattr(self.model, "model") and self.do_prefill:
            # prefill without calculating the logits (save memory for large vocab models)
            prefill = self.model.model(input_ids=inputs.input_ids[..., :-1],
                                       attention_mask=inputs.attention_mask[..., :-1],
                                       pixel_values=inputs.pixel_values,
                                       image_sizes=inputs.image_sizes)
            past_key_values = prefill.past_key_values
            del prefill
            torch.cuda.empty_cache()
            inputs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
                      "pixel_values": inputs.pixel_values, "image_sizes": inputs.image_sizes,
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
        text = self.safe_decode(outputs['sequences'][0, input_len:], skip_special_tokens=True)

        if input_len > 1500:
            save_prompt = self.safe_decode(inputs["input_ids"][0][:500]) + " <skip> " + self.safe_decode(
                inputs["input_ids"][0][-500:])
        else:
            save_prompt = self.safe_decode(inputs["input_ids"][0])
        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs['sequences'].size(1) - input_len,
            "input_text": save_prompt,
        }