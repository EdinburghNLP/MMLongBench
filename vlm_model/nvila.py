import re
import torch
from .model_utils import LLM, truncate_images
from transformers import AutoProcessor, AutoModel, AutoConfig


# FIXME the modeling code of NVILA cannot pass "flash_attention_2" properly, if you want to use it.
# FIXME modeling_vila.py: In VILAPretrainedModel.from_pretrained, add "if config is None:" before the config initialization (at about line 421).
# FIXME builder.py: in the build_llm_and_tokenizer function, change "llm_cfg._attn_implementation = attn_implementation" to "llm_cfg._attn_implementation = config._attn_implementation" (at about line 191)
class NVILAModel(LLM):
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
        self.max_image_num = kwargs.get("max_image_num", None)

        if "2b" in model_name.lower():
            from .nvila_2b_ef8fa9c8.modeling_vila import VILAForCausalLM
            from .nvila_2b_ef8fa9c8.auto_processor import VILAProcessor
        else:
            from .nvila_8b_e2481b0c.modeling_vila import VILAForCausalLM
            from .nvila_8b_e2481b0c.auto_processor import VILAProcessor

        self.max_length = max_length
        self.processor = VILAProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # config._attn_implementation = "flash_attention_2"

        self.model = VILAForCausalLM.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            # config=config,
            **model_kwargs
        )

        self.processor.vision_tower = self.model.vision_tower
        print("config", self.model.config._attn_implementation)
        print("vision_tower_cfg", self.model.vision_tower._modules['vision_tower']._modules['vision_model'].encoder._modules['layers']._modules['0'].self_attn)
        print("llm_config", self.model.llm.config._attn_implementation)

        if kwargs.get("torch_compile", True):
            self.model = torch.compile(self.model)

        # use the default if possible, append if necessary
        stop_token_ids = self.processor.tokenizer.eos_token_id
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
                    "path": image_list[image_idx]
                })
                image_idx += 1
            else:
                new_content.append({
                    "type": "text",
                    "text": c
                })
        assert image_idx == len(image_list)
        messages = [{"role": "user", "content": new_content},
                    {"role": "assistant", "content": system_prompt}]
        return messages


    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        messages = self.format_chat(text, image_list, data["system_template"])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        inputs = self.processor([text],
            padding=True,
            return_tensors="pt",
        )
        return inputs


    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)

        generation_config = self.model.default_generation_config if self.model.generation_config is None else self.model.generation_config
        generation_config.do_sample = self.do_sample
        generation_config.temperature = self.temperature if self.do_sample else None
        generation_config.top_p = self.top_p if self.do_sample else None
        generation_config.top_k = None
        generation_config.eos_token_id = self.stop_token_ids
        generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            generation_config=generation_config
        )
        text = self.processor.decode(outputs[0, input_len:], skip_special_tokens=True)

        if input_len > 1500:
            save_prompt = self.processor.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.processor.decode(
                inputs["input_ids"][0][-500:])
        else:
            save_prompt = self.processor.decode(inputs["input_ids"][0])
        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs.size(1) - input_len,
            "input_text": save_prompt,
        }