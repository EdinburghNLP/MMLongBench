import re
import torch
from .model_utils import LLM
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from .qwen2_with_prefill.qwen2vl import Qwen2VLForLogitsReducedGeneration
from .qwen2_with_prefill.qwen2_5vl import Qwen2_5_VLForLogitsReducedGeneration


class Qwen2VLModel(LLM):
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
        self.do_prefill = kwargs.get("do_prefill", True) # Note: We overrided the Qwen2VL and Qwen2.5VL code and reduce the num_logits_to_keep to the last one token.
        self.max_length = max_length

        # Note for preprocessor: You may find the processor of QVQ AWQ models
        # has more parameters than the BF16 version. For example, (1) "do_convert_rgb": true, "do_normalize": true, "do_rescale": true, "do_resize": true,
        # (2) "resample": 3, "rescale_factor": 0.00392156862745098,
        # However, we find those values are default for QVQ, which means they are not necessary to be written explicitly.
        # If you load processors of QVQ-72B and QVQ-72B-AWQ and print them, the only difference is the proessor name.
        if "qvq-" in model_name.lower():
            # the preprocessor config of QVQ-72B-Preview-AWQ has a bug: ValueError: size must have one of the following set of keys: ({'height', 'width'}, {'shortest_edge'}, {'longest_edge', 'shortest_edge'}, {'longest_edge'}, {'max_height', 'max_width'}), got dict_keys(['max_pixels', 'min_pixels'])
            # TODO After this fix, I found we can just change use_fast=False to fix this bug
            self.processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview", use_fast=True)
        elif model_name == "Qwen/Qwen2-VL-72B-Instruct-AWQ":
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

        # set config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "Qwen2.5" in model_name:
            config.max_position_embeddings = 32768 # this is set wrong in qwen2.5-vl, fix it according to qwen2.5
        if "use_yarn" in kwargs and kwargs["use_yarn"]:
            # To check whether you load yarn correctly, debug into the class Qwen2_5_VLRotaryEmbedding(nn.Module) and transformers.modeling_rope_utils
            # Here is a guideline for using yarn, we indeed need max_position_embeddings=32768: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/discussions/5
            config.rope_scaling = {
                "type": "yarn",
                "mrope_section": [
                    16,
                    24,
                    24
                ],
                "factor": 4,
                "original_max_position_embeddings": 32768
            }

        if "Qwen2.5" in model_name:
            model_cls = Qwen2_5_VLForLogitsReducedGeneration
        else: # "Qwen2" and "QVQ"
            model_cls = Qwen2VLForLogitsReducedGeneration

        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            config=config,
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
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                new_content.append({
                    "type": "image",
                    "image": image_list[image_idx]
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
        messages = self.format_chat(text, image_list, data["system_template"])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs


    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)
        # if hasattr(self.model, "model") and self.do_prefill:
        #     # prefill without calculating the logits (save memory for large vocab models)
        #     prefill = self.model.model(input_ids=inputs.input_ids[..., :-1],
        #                                attention_mask=inputs.attention_mask[..., :-1],
        #                                pixel_values=inputs.pixel_values,
        #                                image_sizes=inputs.image_sizes)
        #     past_key_values = prefill.past_key_values
        #     inputs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
        #               "past_key_values": past_key_values}

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