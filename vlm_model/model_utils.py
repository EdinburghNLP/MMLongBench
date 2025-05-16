import re
import io
import time
import base64
from PIL import Image

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def resize_image(image_list, image_resize):
    new_image_list = []
    for img in image_list:
        width, height = img.size
        new_width = int(width * image_resize)
        new_height = int(height * image_resize)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_image_list.append(img)
    return new_image_list


def resize_image_max_size(image_list, max_image_size):
    new_image_list = []
    for img in image_list:
        width, height = img.size
        if width <= max_image_size and height <= max_image_size:
            new_image_list.append(img)
            continue

        if width > height:
            new_width = max_image_size
            new_height = min(int(max_image_size / width * height), max_image_size)
        else:
            new_height = max_image_size
            new_width = min(int(max_image_size / height * width), max_image_size)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_image_list.append(img)

    return new_image_list


def image_to_io(image: Image.Image, format: str = 'PNG') -> io.BytesIO:
    img_io = io.BytesIO()
    image.save(img_io, format=format)
    img_io.seek(0)
    return img_io


def encode_image_base64(pil_image, format="PNG"):
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


def truncate_images(text, image_list, max_image_num=None):
    """
    keep the last max_image_num images in the example. Truncate image_list and remove beginning <image> marker in text.

    Args:
        text (str): query with <image>
        image_list (list): list of image path (or PIL.Image)
        max_image_num (int, optional): Max number of kept images

    Returns:
        tuple: revised text and image_list
    """
    if max_image_num is None or len(image_list) <= max_image_num:
        return text, image_list

    segments = re.split(r'(<image>)', text)

    # compute remove number
    keep_count = max_image_num
    remove_count = len(image_list) - keep_count

    # compute <image> marker numbers
    image_tags_count = segments.count('<image>')

    # safe check
    assert image_tags_count == len(image_list), f"Warning: Number of <image> tags ({image_tags_count}) doesn't match image_list length ({len(image_list)})"

    # build new text
    new_segments = []
    removed = 0
    for segment in segments:
        if segment == '<image>' and removed < remove_count:
            # replace with ""
            new_segments.append('')
            removed += 1
        else:
            new_segments.append(segment)

    # join all segments
    new_text = ''.join(new_segments)

    # only keep last kepp_count images
    new_image_list = image_list[-keep_count:]

    return new_text, new_image_list


def format_chat(text, image_list, system_prompt):
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


def call_api(func, limit: int=5, pause: int=10):
    """
    Call the API function with retries and rate limit handling.
    TODO: more error handling?
    """
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            msg = str(e).lower()

            if "rate limit" in msg or "rate_limit" in msg or "quota" in msg or "429" in msg or ("overloaded" in msg and count >= limit):
                logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
            count += 1
            if count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                time.sleep(pause)
            else:
                logger.info("Skipping generation due to unknown error")
                raise e
    return output


class LLM:
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
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.generation_min_length = generation_min_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.stops = None
        if stop_newline:
            self.stops = ["\n", "\n\n"]

    def prepare_inputs(self, test_item, data):
        raise NotImplementedError("prepare_inputs not implemented for LLM")
    
    def generate(self, inputs=None, prompt=None, **kwargs):
        raise NotImplementedError("generate not implemented for LLM")
