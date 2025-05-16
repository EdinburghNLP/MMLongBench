import re
import torch
from copy import deepcopy
from .model_utils import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenVLModel(LLM):
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
        model_kwargs["use_flash_attn"] = True
        # Qwen-VL usese flash-attention by default
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left" # we truncate elder history than recent one
        self.tokenizer.padding_side = "left" # batch generation needs left padding

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        if torch_dtype == "int8":
            raise ValueError("dtype doesn't support int8")
        elif torch_dtype == "int4":
            # model_kwargs["load_in_4bit"] = True
            # Please use Qwen-VL-Chat-Int4
            assert model_name == "Qwen/Qwen-VL-Chat-Int4"
            torch_dtype = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

        # if kwargs.get("torch_compile", True):
        #     self.model = torch.compile(self.model)

        # use the default if possible, append if necessary
        stop_token_ids = self.model.generation_config.eos_token_id
        stop_token_ids = [stop_token_ids] if not isinstance(stop_token_ids, list) else stop_token_ids
        if stop_newline:
            stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"]))
            stop_token_ids = list(
                set([self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + stop_token_ids))
            if self.tokenizer.unk_token_id is not None and self.tokenizer.unk_token_id in stop_token_ids:
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids
        self.device = self.model.device
        self.processor = self.tokenizer

    def format_chat(self, text, image_list, system_prompt):
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                new_content.append({
                    "image": image_list[image_idx]
                })
                image_idx += 1
            else:
                new_content.append({
                    "text": c
                })
        assert image_idx == len(image_list)
        return new_content


    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        messages = self.format_chat(text, image_list, data["system_template"])

        inputs = self.tokenizer.from_list_format(messages)

        return inputs


    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        input_len = -1
        generation_config = deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = self.generation_max_length
        generation_config.do_sample = self.do_sample
        generation_config.temperature = self.temperature if self.do_sample else None
        generation_config.top_p = self.top_p if self.do_sample else None
        generation_config.top_k = None
        generation_config.eos_token_id = self.stop_token_ids

        text, history = self.model.chat(self.tokenizer, query=inputs, history=None,
                                        generation_config=generation_config)

        save_prompt = inputs
        if len(save_prompt) > 6000:
            save_prompt = save_prompt[:2000] + " <skip> " + save_prompt[-2000:]
        return {
            "output": text,
            "input_len": input_len,
            "output_len": -1,
            "input_text": save_prompt,
        }