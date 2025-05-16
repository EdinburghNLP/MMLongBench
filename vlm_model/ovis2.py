import re
import torch
from PIL import Image
from .model_utils import LLM, truncate_images
from typing import List, Optional
from functools import partial

from transformers import AutoModelForCausalLM

IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_TOKEN = "<image>"
IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305]


def merge_multimodal(
        self,
        text_input_ids: torch.Tensor,
        text_attention_masks: torch.Tensor,
        text_labels: Optional[torch.Tensor],
        pixel_values: List[Optional[torch.Tensor]],
        left_padding: bool = False,
        vision_batch_size=30,
):
    input_device = text_input_ids.device
    visual_vocab_szie = self.get_visual_tokenizer().config.vocab_size
    visual_indicator_embeds = self.get_vte()(
        torch.tensor(
            list(range(visual_vocab_szie - 5, visual_vocab_szie)),
            dtype=torch.long,
            device=self.get_visual_tokenizer().device
        )
    ).to(device=input_device)

    if self.training:
        # When training, to be compatible with deepspeed zero, each sample has to include pixel_value tensor.
        # For text-only sample, one can simply use a full zero tensor as pixel_value, which will be ignored
        # (see below in this function); so, the gradient will not be affected.
        num_images = [x.shape[0] for x in pixel_values]
        visual_tokens = self.visual_tokenizer(torch.cat([x for x in pixel_values], dim=0))
        visual_embeds = torch.split(self.get_vte()(visual_tokens).to(dtype=self.dtype, device=input_device),
                                    split_size_or_sections=num_images, dim=0)
        visual_input_ids = torch.split(torch.argmax(visual_tokens, dim=-1).to(device=input_device),
                                       split_size_or_sections=num_images, dim=0)
        visual_labels = [torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device) for x in
                         visual_input_ids]
    else:
        # When inference, sample can include only text with `None` pixel_value
        num_images = [x.shape[0] if x is not None else 0 for x in pixel_values]
        if sum(num_images) > 0:
            all_images = torch.cat([x for x in pixel_values if x is not None], dim=0)
            total_images = all_images.shape[0]
            # encode in batches
            visual_embeds_list = []
            visual_input_ids_list = []
            for i in range(0, total_images, vision_batch_size):
                batch_end = min(i + vision_batch_size, total_images)
                batch_images = all_images[i:batch_end]
                # encode current batch
                batch_visual_tokens = self.visual_tokenizer(batch_images)
                batch_visual_embeds = self.get_vte()(batch_visual_tokens).to(dtype=self.dtype, device=input_device)
                batch_visual_input_ids = torch.argmax(batch_visual_tokens, dim=-1).to(device=input_device)
                # collect all results
                visual_embeds_list.append(batch_visual_embeds)
                visual_input_ids_list.append(batch_visual_input_ids)

            # concatenate all reslts
            all_visual_embeds = torch.cat(visual_embeds_list, dim=0)
            all_visual_input_ids = torch.cat(visual_input_ids_list, dim=0)
            # split by image tiles number
            visual_embeds = torch.split(all_visual_embeds, split_size_or_sections=num_images, dim=0)
            visual_input_ids = torch.split(all_visual_input_ids, split_size_or_sections=num_images, dim=0)
            visual_labels = [torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device) for x in
                             visual_input_ids]
        else:
            # just placeholders
            visual_embeds = [None] * len(num_images)
            visual_input_ids = [None] * len(num_images)
            visual_labels = [None] * len(num_images)
        # just placeholders
        if text_labels is None:
            text_labels = torch.full(text_input_ids.shape, IGNORE_ID, dtype=torch.long, device=input_device)

    input_embeds = []
    attention_masks = []
    labels = []
    for text_input_id, text_label, text_attention_mask, visual_embed, visual_input_id, visual_label in zip(
            text_input_ids, text_labels, text_attention_masks, visual_embeds, visual_input_ids, visual_labels
    ):
        placeholder_token_mask = torch.lt(text_input_id, 0)
        text_embed = self.get_wte()(torch.masked_fill(text_input_id, placeholder_token_mask, 0))
        for i, indicator_id in enumerate(IMAGE_INDICATOR_IDS):
            text_embed[text_input_id == indicator_id] = visual_indicator_embeds[i]
        image_atom_positions = torch.where(torch.eq(text_input_id, IMAGE_ATOM_ID))[0].tolist()
        if len(image_atom_positions) > 0:
            input_embed_parts = []
            attention_mask_parts = []
            label_parts = []
            prev_image_atom_position = -1
            for index, image_atom_position in enumerate(image_atom_positions):
                input_embed_parts.append(
                    text_embed[prev_image_atom_position + 1:image_atom_position, :])
                label_parts.append(
                    text_label[prev_image_atom_position + 1:image_atom_position])
                attention_mask_parts.append(
                    text_attention_mask[prev_image_atom_position + 1:image_atom_position])
                input_embed_parts.append(visual_embed[index])
                attention_mask_parts.append(
                    torch.ones_like(visual_label[index], dtype=torch.bool))
                label_parts.append(visual_label[index])
                prev_image_atom_position = image_atom_position
            if prev_image_atom_position + 1 < text_input_id.shape[0]:
                input_embed_parts.append(
                    text_embed[prev_image_atom_position + 1:, :])
                attention_mask_parts.append(
                    text_attention_mask[prev_image_atom_position + 1:])
                label_parts.append(
                    text_label[prev_image_atom_position + 1:])
            input_embed = torch.cat(input_embed_parts, dim=0)
            attention_mask = torch.cat(attention_mask_parts, dim=0)
            label = torch.cat(label_parts, dim=0)
        else:
            input_embed = text_embed
            attention_mask = text_attention_mask
            label = text_label
            if self.training:
                # Make visual_embed & visual_indicator_embeds involved in the backward graph,
                # to be compatible with deepspeed zero and ddp.
                input_embed += torch.sum(visual_embed * 0.0) + torch.sum(visual_indicator_embeds * 0.0)
        input_embeds.append(input_embed)
        attention_masks.append(attention_mask)
        labels.append(label)

    if self.training:  # padding to self.config.multimodal_max_length for increased training speed
        padding_size = max(0, self.config.multimodal_max_length - len(input_embeds[0]))
        input_embeds[0] = torch.nn.ConstantPad2d((0, 0, 0, padding_size), 0.0)(input_embeds[0])
        attention_masks[0] = torch.nn.ConstantPad1d((0, padding_size), False)(attention_masks[0])
        labels[0] = torch.nn.ConstantPad1d((0, padding_size), IGNORE_ID)(labels[0])
    batch_input_embeds = self.pad_truncate_sequence(input_embeds, batch_first=True, padding_value=0.0,
                                                    left_padding=left_padding)
    batch_attention_mask = self.pad_truncate_sequence(attention_masks, batch_first=True, padding_value=False,
                                                      left_padding=left_padding)
    batch_labels = self.pad_truncate_sequence(labels, batch_first=True, padding_value=IGNORE_ID,
                                              left_padding=left_padding)

    return visual_input_ids, batch_input_embeds, batch_labels, batch_attention_mask


# We add resize to do token compression, then add a image truncation
class Ovis2Model(LLM):
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
        # model_kwargs["attn_implementation"] = kwargs.get("attn_implementation", "flash_attention_2")
        # ovis2 uses flash_attention by default, not need to set this.
        self.max_length = max_length
        self.image_resize = kwargs.get("image_resize", None)
        self.max_image_num = kwargs.get("max_image_num", None)
        self.vision_batch_size = kwargs.get("vision_batch_size", 30)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            multimodal_max_length=131072,
            trust_remote_code=True,
            device_map="auto",
            **model_kwargs
        )

        import types
        self.model.merge_multimodal = types.MethodType(
            partial(merge_multimodal, vision_batch_size=self.vision_batch_size), self.model)

        print("current attention implementation:", self.model.llm.base_model._modules['layers']._modules['0']._modules[
            'self_attn'].config._attn_implementation)

        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.processor = self.text_tokenizer
        tokenizer = self.text_tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left" # we truncate elder history than recent one
        tokenizer.padding_side = "left" # batch generation needs left padding

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
        self.max_partition = 4

    def format_chat(self, text, image_list, system_prompt):
        return text

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"]
        if self.max_image_num is not None:
            text, image_list = truncate_images(text, image_list, self.max_image_num)

        query = self.format_chat(text, image_list, data["system_template"])
        image_list = [Image.open(image).convert('RGB') for image in image_list]

        if self.image_resize is not None:
            from .model_utils import resize_image
            image_list = resize_image(image_list, self.image_resize)

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, image_list,
                                                                       max_partition=self.max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype)

        return {"text": prompt, "input_ids": input_ids,
        "attention_mask": attention_mask, "pixel_values": pixel_values}

    def safe_decode(self, input_ids, skip_special_tokens=False):
        safe_ids = input_ids.clone()
        safe_ids[safe_ids < 0] = self.text_tokenizer.pad_token_id
        return self.text_tokenizer.decode(safe_ids, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def generate(self, inputs=None, prompt=None, **kwargs):
        input_ids = inputs["input_ids"].to(device=self.model.device)
        attention_mask = inputs["attention_mask"].to(device=self.model.device)
        pixel_values = inputs["pixel_values"].to(device=self.visual_tokenizer.device)
        input_len = input_ids.size(1)
        outputs = self.model.generate(
            input_ids,
            pixel_values=[pixel_values],
            attention_mask=attention_mask,
            max_new_tokens=self.generation_max_length,
            min_new_tokens=self.generation_min_length,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            top_k = None,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.text_tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        text = self.processor.decode(outputs['sequences'][0], skip_special_tokens=True)

        if input_len > 1500:
            save_prompt = self.safe_decode(inputs["input_ids"][0][:500]) + " <skip> " + self.safe_decode(
                inputs["input_ids"][0][-500:])
        else:
            save_prompt = self.safe_decode(inputs["input_ids"][0])
        return {
            "output": text,
            "input_len": input_len,
            "output_len": outputs['sequences'].size(1),
            "input_text": save_prompt,
        }