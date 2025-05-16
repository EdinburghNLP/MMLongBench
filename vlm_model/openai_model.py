import re, os
from .model_utils import LLM, call_api, truncate_images
from .model_utils import encode_image_base64
from functools import partial
import openai
from PIL import Image
import time
import requests


def is_url(path):
    return path.startswith('http://') or path.startswith('https://')

# huggingface datasets cannot support a folder with more than 10K iamge
# we split the images in mm-niah/obelics and vh/train2017
def string_to_int_mod(string_value, mod_num=20):
    char_sum = sum(ord(char) for char in string_value)
    result = char_sum % mod_num
    return result

def add_url_bucket(url):
    new_url = url
    check_dir_list = ["vh/train2017/", "mm-niah/obelics/"]
    for check_dir in check_dir_list:
        if check_dir in url:
            prefix, file_name = url.split(check_dir)
            bucket_id = string_to_int_mod(file_name)
            bucket_str = f"bucket_{bucket_id}"
            new_url = f"{prefix}{check_dir}{bucket_str}/{file_name}"

    return new_url


class OpenAIModel(LLM):
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
            use_chat_template=True,
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
        self.api_model = True
        self.max_image_num = kwargs.get("max_image_num", None)
        self.api_sleep = kwargs.get("api_sleep", 10)
        self.image_detail = kwargs.get("image_detail", "auto")
        self.image_resize = kwargs.get("image_resize", None)

        if "claude" in self.model_name:
            self.model = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],  # Your Anthropic API key
                base_url="https://api.anthropic.com/v1/"  # Anthropic's API endpoint
            )
        else:
            self.model = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        print(f"----------currently, using image_detail: {self.image_detail}----------")

        # make sure to set the OPENAI_API_KEY environment variable

        self.model_name = model_name
        self.processor = None

    def format_chat(self, text, image_list, system_prompt, is_url_image=False):
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                if is_url_image:
                    curr_image_url = image_list[image_idx]
                else:
                    curr_image_url = f"data:image/png;base64,{image_list[image_idx]}"
                new_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": curr_image_url,
                        "detail": self.image_detail
                    },
                })
                image_idx += 1
            else:
                new_content.append({
                    "type": "text",
                    "text": c
                })
        assert image_idx == len(image_list)
        messages = [{"role": "user", "content": new_content}]
        return messages

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        if is_url(image_list[0]):
            # image_flag = [check_image_url_mime(url) for url in image_list]
            image_list = [add_url_bucket(url) for url in image_list]
            messages = self.format_chat(text, image_list, data["system_template"], is_url_image=True)
        else:
            image_list = [Image.open(image).convert('RGB') for image in image_list]

            if self.image_resize is not None:
                from .model_utils import resize_image
                image_list = resize_image(image_list, self.image_resize)

            # convert all format images to png
            image_list = [encode_image_base64(image) for image in image_list]

            messages = self.format_chat(text, image_list, data["system_template"])

        return {"messages": messages}

    def generate(self, inputs=None, prompt=None, **kwargs):
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.

        save_prompt = [c["text"] if c["type"] == "text" else "<image>" for c in inputs["messages"][0]["content"]]
        save_prompt = "".join(save_prompt)
        if len(save_prompt) > 6000:
            save_prompt = save_prompt[:2000] + " <skip> " + save_prompt[-2000:]

        func = partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs["messages"],
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            **kwargs,
        )

        start_time = time.time()
        try:
            output = call_api(func, limit=5, pause=5)
        except Exception as e:
            print("current call_api failed, assign None to output.")
            output = None

        end_time = time.time()

        print(f"example finished, used {end_time - start_time} secs, sleep {max(self.api_sleep - (end_time - start_time), 0.1)} secs")
        time.sleep(max(self.api_sleep - (end_time - start_time), 0.1))

        if output is not None and output.choices[0].message.content is not None:
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": save_prompt,
            }
        return {"output": "", "input_len": -1, "output_len": -1, "input_text": save_prompt}
