from transformers import AutoConfig, AutoTokenizer, AutoModel
from .model_utils import LLM


from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, 'get_max_length'):
    # set get_max_length as the alias of get_max_cache_shape
    DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
    print("Added get_max_length method to DynamicCache for backward compatibility")

from PIL import Image


class mPLUGOwl3Model(LLM):
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


        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # print(self.config)

        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        # self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.processor = self.model.init_processor(self.tokenizer)

  
    def format_chat(self, text, image_list, system_prompt):
        formatted_text = text.replace("<image>", "<|image|>")
        messages = [{"role": "user", "content": formatted_text},{"role": "assistant", "content": ""}]
        return messages

    
    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        # import pdb; pdb.set_trace()

        # Convert file paths to PIL Images
        pil_images = [Image.open(img_path).convert("RGB") for img_path in image_list]

        messages = self.format_chat(text, image_list, data["system_template"])
        inputs = self.processor(
            messages,
            images=pil_images,
            return_tensors="pt"
        )
        return inputs