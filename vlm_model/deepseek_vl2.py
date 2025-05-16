import re
import torch
from .model_utils import LLM

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
DeepseekVLV2ForCausalLM._supports_flash_attn_2 = True

### FIXME: First, you need to install deepseek_vl2 package
### FIXME: Current DeepSeek-VL2 only support flash-attention within transformers<=4.47.1
### FIXME: Reason is that "LlamaFlashAttention2" is removed in later versions of transformers
### FIXME: If you want to do inference with DeepSeek-VL2, you need to downgrade the transformers version
### FIXME: In our experiments, we used transformers==4.38.2 (see https://github.com/deepseek-ai/DeepSeek-VL2/issues/4)
class DeepseekVL2Model(LLM):
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
        self.max_length = max_length
        self.processor = DeepseekVLV2Processor.from_pretrained(model_name, use_fast=True)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding
        self.tokenizer = tokenizer

        # set config
        config = DeepseekVLV2Config.from_pretrained(model_name)
        config._attn_implementation = "flash_attention_2"
        config.language_config._attn_implementation = "flash_attention_2"

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        if torch_dtype == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = quantization_config
            torch_dtype = None
            config.torch_dtype = None
        elif torch_dtype == "int4":
            model_kwargs["load_in_4bit"] = True
            torch_dtype = None
            config.torch_dtype = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            config=config,
            **model_kwargs
        )

        if kwargs.get("torch_compile", True):
            self.model = torch.compile(self.model)

        # use the default if possible, append if necessary
        stop_token_ids = self.tokenizer.eos_token_id
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
        assert text.count("<image>") == len(image_list)
        messages = [{"role": "<|User|>", "content": text,
                     "images": image_list},
                    {"role": "<|Assistant|>", "content": ""}]
        return messages


    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        messages = self.format_chat(text, image_list, data["system_template"])
        pil_images = load_pil_images(messages)

        inputs = self.processor(
            conversations=messages,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )
        return inputs


    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)

        # print(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)# .to(self.model.device)

        outputs = self.model.language.generate(
            input_ids=inputs["input_ids"].to(self.model.device),
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.stop_token_ids,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            top_k = None,
            return_dict_in_generate=True,
            output_scores=False,
        )
        text = self.tokenizer.decode(outputs['sequences'][0, input_len:], skip_special_tokens=True)

        if input_len > 1500:
            save_prompt = self.tokenizer.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.tokenizer.decode(
                inputs["input_ids"][0][-500:])
        else:
            save_prompt = self.tokenizer.decode(inputs["input_ids"][0])
        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs['sequences'].size(1) - input_len,
            "input_text": save_prompt,
        }