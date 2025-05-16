from .model_utils import LLM
import torch
import transformers
from transformers import PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig


def format_chat(message, include_assistant=False, assistant_message="You are a helpful assistant."):
    if include_assistant:
        chat = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_message}
        ]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


def tokenize(sample, data, tokenizer, max_length, generation_max_length, use_chat_template=False):
    def format_input(sample):
        if use_chat_template:
            chat = format_chat(
                data["user_template"].format(**sample),
                include_assistant=True,
                assistant_message=data.get("system_template", "You are a helpful assistant.")
            )
            try:
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True)
            except Exception as e:
                chat = format_chat(
                    data["user_template"].format(**sample),
                )
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            tokenized_input = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        else:
            prompt = data["prompt_template"].format(**sample)
            tokenized_input = tokenizer([prompt], return_tensors="pt")
        return tokenized_input

    if "Phi3SmallTokenizer" in str(type(tokenizer)):
        buffer = 64 if max_length == 131072 else 0  # there is some problem with their rotary emb implementation
    else:
        buffer = 0

    tokenized_input = format_input(sample)
    if tokenized_input.input_ids.size(1) > max_length - generation_max_length - buffer:
        truncate_length = tokenized_input.input_ids.size(1) - (max_length - generation_max_length - buffer)

        # handle non-fast hf tokenizers (e.g., phi-3-small)
        if isinstance(tokenizer, PreTrainedTokenizer) and not tokenizer.is_fast:
            context_tokens = tokenizer(sample["context"])
            new_context = tokenizer.decode(context_tokens["input_ids"][:-truncate_length])
        else:
            context_tokens = tokenizer([sample["context"]], return_offsets_mapping=True)
            new_context = sample["context"][:context_tokens["offset_mapping"][0][-truncate_length][0]]

        sample["context"] = new_context
        tokenized_input = format_input(sample)
    return tokenized_input


class DummyTextProcessor: # A dummy processor to fit into the current code framework for VLMs.
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


class TextOnlyModel(LLM):
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
        self.do_prefill = kwargs.get("do_prefill", True)
        model_kwargs["attn_implementation"] = kwargs.get("attn_implementation", "flash_attention_2")
        if "recurrentgemma" in model_name or "yarn" in model_name.lower():
            model_kwargs = {}

        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        self.processor = DummyTextProcessor(self.tokenizer)

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "rope_theta" in kwargs and kwargs["rope_theta"] is not None:
            print(f"Override rope theta to {kwargs['rope_theta']}")
            config.rope_theta = kwargs["rope_theta"]
        if "use_yarn" in kwargs and kwargs["use_yarn"] and "qwen2" in model_name.lower():
            # To check whether you load yarn correctly, debug into the class Qwen2RotaryEmbedding(nn.Module) and transformers.modeling_rope_utils
            # Here is a guideline for using yarn: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/discussions/5
            print(f"using yarn for {model_name}")
            config.rope_scaling = {
                "factor": 4,
                "original_max_position_embeddings": 32768,
                "type": "yarn",
            }


        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
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
                set([self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + stop_token_ids))
            if "llama" in model_name.lower():
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            stop_token_ids = [x for x in stop_token_ids if x is not None]
        self.stop_token_ids = stop_token_ids
        self.device = self.model.device

        if "gemma" in model_name.lower():
            self.do_prefill = False
            print("gemma models cannot prefill with past kvs due to cache implementation, need to change the code manually if you need to prefill")

    def prepare_inputs(self, test_item, data):
        return tokenize(
            test_item,
            data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            generation_max_length=self.generation_max_length,
            use_chat_template=self.use_chat_template,
        )

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            inputs = self.tokenizer([prompt], return_tensors="pt",
                                    max_length=self.max_length - self.generation_max_length, truncation=True,
                                    padding=True)

        inputs = inputs.to(self.model.device)
        input_len = inputs.input_ids.size(1)
        if hasattr(self.model, "model") and self.do_prefill:
            # prefill without calculating the logits (save memory for large vocab models)
            extra = {}
            if "jamba" in str(type(self.model)).lower():
                from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache
                cache = HybridMambaAttentionDynamicCache(self.model.config, inputs.input_ids.shape[0], self.model.dtype,
                                                         device=self.model.device)
                extra = {"past_key_values": cache}

            prefill = self.model.model(input_ids=inputs.input_ids[..., :-1],
                                       attention_mask=inputs.attention_mask[..., :-1], **extra)
            past_key_values = prefill.past_key_values
            inputs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
                      "past_key_values": past_key_values}
            if past_key_values is None:
                self.do_prefill = False
                print("past key values is None, not able to prefill with KVs, disabling...")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        text = self.tokenizer.decode(outputs['sequences'][0, input_len:], skip_special_tokens=True)
        save_prompt = self.tokenizer.decode(inputs["input_ids"][0][:500]) + " <skip> " + self.tokenizer.decode(
            inputs["input_ids"][0][-500:])

        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs['sequences'].size(1) - input_len,
            "input_text": save_prompt,
        }