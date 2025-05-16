import torch


def load_LLM(args):
    kwargs = {}
    if "gpt" in args.model_name_or_path or "claude" in args.model_name_or_path: # we use gpt compatible api
        from .openai_model import OpenAIModel
        model_cls = OpenAIModel
    elif "gemini" in args.model_name_or_path:
        from .gemini import GeminiModel
        model_cls = GeminiModel
    else:
        # HF models
        lower_model_name = args.model_name_or_path.lower().split("/")[-1]
        if "qwen2-vl" in lower_model_name or "qwen2.5-vl" in lower_model_name or "qvq-" in lower_model_name:
            from .qwen2_vl import Qwen2VLModel
            model_cls = Qwen2VLModel
        elif "qwen-vl" in lower_model_name: # too many tokens for image
            from .qwen_vl import QwenVLModel
            model_cls = QwenVLModel
        elif "llama-3.2" in lower_model_name: # too many tokens for image
            from .mllama import MLlamaModel
            model_cls = MLlamaModel
        elif "deepseek-vl2" in lower_model_name: # hard code integration
            from .deepseek_vl2 import DeepseekVL2Model
            model_cls = DeepseekVL2Model
        elif "idefics" in lower_model_name:
            from .idefics import IdeficsModel
            model_cls = IdeficsModel
            if args.do_image_splitting != "None":
                # assert "idefics2" in lower_model_name
                kwargs["do_image_splitting"] = args.do_image_splitting == "True"
            if args.vision_batch_size is not None:
                kwargs["vision_batch_size"] = args.vision_batch_size
        elif "phi-4" in lower_model_name:
            from .phi4 import Phi4Model
            model_cls = Phi4Model
        elif "phi-3" in lower_model_name:
            from .phi3 import Phi3Model
            model_cls = Phi3Model
        elif "v2pe" in lower_model_name:
            from .internv2pe import InternV2PEModel
            if args.vision_batch_size is not None:
                kwargs["vision_batch_size"] = args.vision_batch_size
            if args.v2pe_step is not None:
                kwargs["v2pe_step"] = args.v2pe_step
            model_cls = InternV2PEModel
        elif "internvl" in lower_model_name:
            from .internvl import InternVLModel
            if args.vision_batch_size is not None:
                kwargs["vision_batch_size"] = args.vision_batch_size
            model_cls = InternVLModel
        elif "pixtral" in lower_model_name:
            from .pixtral import PixtralModel
            if args.vision_batch_size is not None:
                kwargs["vision_batch_size"] = args.vision_batch_size
            model_cls = PixtralModel
        elif "nvila" in lower_model_name:
            from .nvila import NVILAModel
            model_cls = NVILAModel
        elif "llava-onevision" in lower_model_name: # too many tokens for image
            from .llava_onevision import LlavaOneVisionModel
            model_cls = LlavaOneVisionModel
        elif "gemma-3" in lower_model_name:
            from .gemma3 import Gemma3VLModel
            model_cls = Gemma3VLModel
        elif "minicpm" in lower_model_name: # only support one GPU inference
            from .minicpm import MiniCPMModel
            model_cls = MiniCPMModel
        elif "ovis2" in lower_model_name:
            from .ovis2 import Ovis2Model
            if args.vision_batch_size is not None:
                kwargs["vision_batch_size"] = args.vision_batch_size
            model_cls = Ovis2Model
        elif "qwen2.5" in lower_model_name:
            # we already checked the qwen2.5-vl and qwen2-vl above, this is for qwen2.5 LLMs
            # Not that TextOnlyModel support most text-only LLMs, not just Qwen2.5 LLMs
            from .text_only_model import TextOnlyModel
            model_cls = TextOnlyModel

        elif "mplug-owl3" in lower_model_name: # too many tokens for images
            from .mplug_owl3 import mPLUGOwl3Model
            model_cls = mPLUGOwl3Model

        if args.no_torch_compile:
            kwargs["torch_compile"] = False
        if args.no_bf16:
            kwargs["torch_dtype"] = torch.float16
        if args.load_in_8bit:
            kwargs["load_in_8bit"] = True
        if args.rope_theta is not None:
            kwargs["rope_theta"] = args.rope_theta
        if args.use_yarn:
            kwargs["use_yarn"] = True
        if args.offload_state_dict:
            kwargs["offload_state_dict"] = True
        if args.do_prefill:
            kwargs["do_prefill"] = args.do_prefill
        if args.attn_implementation is not None:
            kwargs["attn_implementation"] = args.attn_implementation
    if args.image_resize:
        kwargs["image_resize"] = args.image_resize
    if args.max_image_num is not None:
        kwargs["max_image_num"] = args.max_image_num
    if args.max_image_size is not None:
        kwargs["max_image_size"] = args.max_image_size
    if args.api_sleep is not None:
        kwargs["api_sleep"] = args.api_sleep
    if args.image_detail != "auto":
        kwargs["image_detail"] = args.image_detail

    model = model_cls(
        args.model_name_or_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.input_max_length,
        generation_max_length=args.generation_max_length,
        generation_min_length=args.generation_min_length,
        do_sample=args.do_sample,
        stop_newline=args.stop_newline,
        use_chat_template=args.use_chat_template,
        **kwargs,
    )

    return model