import argparse
import yaml
import ast
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluation on downstream tasks")
    parser.add_argument("--config", type=str, default=None, help="path to config file")

    # model setting
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", help="whether to use vllm engine")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Implementation of self-attention. None means using the default (flash_attention_2 for most models).")

    # data paths
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--test_file_root", type=str, default=None)
    parser.add_argument("--image_file_root", type=str, default=None)
    parser.add_argument("--test_files", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the predictions")
    parser.add_argument("--overwrite", action="store_true", help="whether to the saved file")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--preprocessing_num_workers", type=int, default=8)

    # evaluation settings
    parser.add_argument("--input_max_length", type=str, default='8192', help="the maximum number of tokens of the input, we truncate the end of the context; can be separated by comma to match the specified datasets")
    parser.add_argument("--test_length", type=str, default="4,8,16,32,64,128", help="list the length to be tested.")

    # generation settings
    parser.add_argument("--do_sample", type=ast.literal_eval, choices=[True, False], default=False, help="whether to use sampling (false is greedy), overwrites temperature")
    parser.add_argument("--generation_max_length", type=str, default='10', help="max number of tokens to generate, can be separated by comma to match the specified datasets")
    parser.add_argument("--generation_min_length", type=int, default=0, help="min number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")
    parser.add_argument("--stop_newline", type=ast.literal_eval, choices=[True, False], default=False, help="whether to stop generation at newline")
    parser.add_argument("--do_prefill", action="store_true", help="prefill the context to save memory")

    # model specific settings
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bf16 and use fp32")
    parser.add_argument("--load_in_8bit", action="store_true", help="int8 mode")
    parser.add_argument("--no_torch_compile", action="store_true", help="disable cuda")
    parser.add_argument("--use_chat_template", type=ast.literal_eval, choices=[True, False], default=True, help="whether to use chat template")
    parser.add_argument("--rope_theta", type=int, default=None, help="override rope theta")
    parser.add_argument("--use_yarn", action="store_true", help="yarn extension")
    parser.add_argument("--do_image_splitting", type=str, choices=["True", "False", "None"], default="None", help="whether to use image splitting for Idefics2 and Mantis (True, False, or None to use model default)")
    parser.add_argument("--offload_state_dict", action="store_true", help="model with offload")
    parser.add_argument("--image_resize", type=float, default=None, help="Image scaling factor, where 1.0 means original size and 0.5 means half the original size")
    parser.add_argument("--max_image_num", type=int, default=None, help="the max image number for models with dynamic cropping (e.g., internvl1.5/2/2.5, phi3/3.5)")
    parser.add_argument("--vision_batch_size", type=int, default=None, help="the batch size for Pixtral's and Ovis2's vision tower since its implementation has O(N^2) memory cost (N is the image number)")
    parser.add_argument("--api_sleep", type=int, default=None, help="the sleep time for API models after each call")
    parser.add_argument("--max_image_size", type=int, default=None, help="Max image size for Gemini to prevent over resizing and splitting")
    parser.add_argument("--image_detail", type=str, choices=["high", "low", "auto"], default="auto", help="Image detail for OpenAI models")
    parser.add_argument("--batch_size", type=int, default=4, help="inference batch size. This is only effective for OpenAI models now!")
    parser.add_argument("--v2pe_step", type=int, default=64, help="the increment size for visual tokens in V2PE")

    # misc
    parser.add_argument("--debug", action="store_true", help="for debugging")
    parser.add_argument("--count_tokens", action="store_true", help="instead of running generation, just count the number of tokens (only for HF models not API)")
    parser.add_argument("--dry_run", action="store_true", help="Test the data loading speed.")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"output/{os.path.basename(args.model_name_or_path)}"

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    return args