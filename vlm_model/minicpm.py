import torch
from .model_utils import LLM

from PIL import Image
from transformers import AutoModel, AutoTokenizer


class MiniCPMModel(LLM):
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
        self.max_length = max_length
        self.processor = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        tokenizer = self.processor
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs
        )

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
        new_content = [Image.open(image).convert('RGB') for image in image_list]
        text_content = text.replace("<image>", "")
        new_content.append(text_content)
        messages = [{"role": "user", "content": new_content}]
        return messages


    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        inputs = self.format_chat(text, image_list, data["system_template"])

        return inputs


    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        text = self.model.chat(
            image=None,
            msgs=inputs,
            tokenizer=self.processor,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            top_k=None,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.processor.pad_token_id,
        )

        save_prompt = [i if isinstance(i, str) else "<image>" for i in inputs[0]["content"]]
        save_prompt = " ".join(save_prompt)
        if len(save_prompt) > 6000:
            save_prompt = save_prompt[:2000] + " <skip> " + save_prompt[-2000:]
        return {
            "output": text,
            "input_len": -1,
            "output_len": -1,
            "input_text": save_prompt,
        }