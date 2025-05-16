import re
import os

from transformers.models.auto.image_processing_auto import model_type

from .model_utils import LLM, logger, call_api, truncate_images
from .model_utils import image_to_io
from functools import partial
from google import genai
from PIL import Image
import time
import concurrent.futures
from functools import partial


class GeminiModel(LLM):
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

        # TODO change this to API_KEY = os.environ["GEMINI_API_KEY"]
        self.api_model = True
        self.max_image_num = kwargs.get("max_image_num", None)
        self.max_image_size = kwargs.get("max_image_size", None)
        self.api_sleep = kwargs.get("api_sleep", 10)
        self.api_key = os.environ["GEMINI_API_KEY"]


        self.model = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.processor = None

    def format_chat(self, text, image_list, system_prompt):
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                new_content.append({"mime_type": "image/png", "image": image_list[image_idx]})
                image_idx += 1
            else:
                new_content.append(c)
        assert image_idx == len(image_list)
        return new_content


    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)
        # loading images instead of using image paths, since gemini doesn't support gif
        image_list = [Image.open(image).convert('RGB') for image in image_list]

        if self.max_image_size is not None:
            from .model_utils import resize_image_max_size
            image_list = resize_image_max_size(image_list, self.max_image_size)

        # convert all format images to png
        image_list = [image_to_io(image) for image in image_list]

        messages = self.format_chat(text, image_list, data["system_template"])

        return {"contents": messages}

    def upload_new_seek(self, file, config):
        if hasattr(file, "seek"):
            file.seek(0)
        return self.model.files.upload(file=file, config=config)

    def upload_content(self, inputs):
        contents = inputs["contents"]
        new_contents = [None] * len(contents)

        upload_tasks = []
        for i, c in enumerate(contents):
            if isinstance(c, str):
                new_contents[i] = c
            else:
                upload_tasks.append((i, c))

        if upload_tasks:
            def upload_single_image(c_tmp):
                func = partial(self.upload_new_seek, file=c_tmp["image"], config={"mime_type": c_tmp["mime_type"]})
                file_uri = call_api(func, limit=3, pause=5, return_rate_limit=True)
                return file_uri

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(upload_tasks))) as executor:
                future_to_idx = {executor.submit(upload_single_image, c): ti for ti, c in upload_tasks}
                for future in concurrent.futures.as_completed(future_to_idx):
                    file_uri = future.result()
                    if file_uri == "rate limit":
                        for f in future_to_idx:
                            if not f.done():
                                f.cancel()
                        raise ValueError("Current Key reached rate limit when uploading")
                    idx = future_to_idx[future]
                    new_contents[idx] = file_uri
        return new_contents


    def generate(self, inputs=None, prompt=None, **kwargs):
        from google.genai.types import GenerateContentConfig

        # see the bug about max_output_tokens: https://github.com/googleapis/python-genai/issues/626
        if any(model_series in self.model_name for model_series in ["gemini-2.0-flash-thinking", "gemini-2.5-pro", "gemini-2.5-flash"]):
            generation_config = GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                candidate_count=1)
        elif self.model_name == "gemini-2.0-flash-001":
            generation_config = GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=self.generation_max_length,
                candidate_count=1)
        else:
            raise ValueError("Wrong Gemini model name.")

        start_time = time.time()
        contents = self.upload_content(inputs)

        func = partial(
            self.model.models.generate_content,
            model=self.model_name,
            contents=contents,
            config=generation_config
        )
        output = call_api(func, pause=5)

        save_prompt = [c if isinstance(c, str) else "<image>" for c in inputs["contents"]]
        save_prompt = "".join(save_prompt)
        if len(save_prompt) > 6000:
            save_prompt = save_prompt[:2000] + " <skip> " + save_prompt[-2000:]

        end_time = time.time()
        print(f"example finished, used {end_time - start_time} secs, sleep {max(self.api_sleep - (end_time - start_time), 0.1)} secs")
        time.sleep(max(self.api_sleep - (end_time - start_time), 0.1))

        return {
            "output": output.text if output is not None and output.text is not None else "",
            "input_len": output.usage_metadata.prompt_token_count
            if output is not None and output.usage_metadata.prompt_token_count is not None else 0,
            "output_len": output.usage_metadata.candidates_token_count
            if output is not None and output.usage_metadata.candidates_token_count is not None else 0,
            "input_text": save_prompt,
        }