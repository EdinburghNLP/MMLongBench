import os
from transformers import set_seed
from collections import defaultdict
import json
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from arguments import parse_arguments
from vlm_model import load_LLM
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from data import (
    load_data, 
    TestItemDataset,
)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_test(args, model, dataset, test_file):
    logger.info(f"running test on {dataset} with test {test_file}")

    test_name = os.path.splitext(os.path.basename(test_file))[0]
    output_path = os.path.join(args.output_dir, f"{dataset}_{test_name}_in{args.input_max_length}_size{args.max_test_samples}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json")
    print("output path:", output_path)
    if os.path.exists(output_path) and not args.overwrite and not args.debug:
        logger.info(f"{output_path} already exists, skipping...")
        return output_path

    set_seed(args.seed)
    data = load_data(args, dataset, test_file)

    if args.dry_run:
        logger.info(f"Dry run mode, loaded {len(data['data'])} samples from {dataset}")
        return None
    else:
        logger.info(f"loaded {len(data['data'])} samples from {dataset}")

    dataloader = DataLoader(
        TestItemDataset(data, model, model.processor),
        batch_size=1, 
        shuffle=False, 
        collate_fn=lambda x: x,
        num_workers=args.num_workers if not args.debug else 0,
    )

    assert hasattr(model, "api_model") and model.api_model, "Wrong model type, not api models"
    cache_path = output_path + ".cache"
    if os.path.exists(cache_path):
        with open(cache_path) as fin:
            results = [json.loads(line) for line in fin.readlines()]
    else:
        results = []
    if args.overwrite:
        fout = open(cache_path, "w")
    else:
        fout = open(cache_path, "a")

    # recover metrics
    metrics = defaultdict(list)
    if results:
        tmp_mets, _ = data['post_process'](results[0], data["data"][0])
        for k, v in tmp_mets.items():
            metrics[k] = [r[k] for r in results if r["parsed_output"] != ""]
        metrics["input_len"] = [r["input_len"] for r in results if r["parsed_output"] != ""]
        metrics["output_len"] = [r["output_len"] for r in results if r["parsed_output"] != ""]

    def process_single_item(idx):
        test_item = data["data"][idx]
        inputs, input_text = dataloader.dataset[idx]

        output = model.generate(inputs=inputs)

        prepend_text = data["system_template"].format(**test_item)
        output["output"] = prepend_text + output["output"]

        mets, others = data['post_process'](output, test_item)
        output.update({**others, **mets})

        result = {**test_item, **output}
        result.pop("context", None)
        result.pop("input_ids", None)

        if input_text is None:
            input_text = result['input_text']

        return idx, result, mets, output["input_len"], output["output_len"], input_text

    start_time = time.time()

    remaining_indices = list(range(len(results), len(data["data"])))
    remaining_samples = len(remaining_indices)
    logger.info(f"Processing {remaining_samples} remaining samples with {args.batch_size} workers")

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        try:
            for start_idx in tqdm(range(0, remaining_samples, args.batch_size)):
                current_indices = remaining_indices[start_idx: start_idx + args.batch_size]
                print(current_indices)
                futures = [executor.submit(process_single_item, i) for i in current_indices]
                for future in futures:
                    result = future.result()

                    idx, result_dict, mets, input_len, output_len, input_text = result

                    if result_dict["parsed_output"] != "":
                        for k, v in mets.items():
                            metrics[k].append(v)
                        metrics["input_len"].append(input_len)
                        metrics["output_len"].append(output_len)
                    else:
                        logger.info(f"skipping example {idx + 1} because the model returned empty string")

                    results.append(result_dict)
                    fout.write(json.dumps(result_dict) + "\n")
                    fout.flush()

                    if idx < 2 or args.debug:
                        logger.info(f"Example {idx + 1}: ")
                        logger.info(f"Decoder inputs:\n{input_text}\n")
                        logger.info(f"Input length: {input_len}")
                        logger.info(f"Question: {result_dict['question'] if 'question' in result_dict else ''}")
                        logger.info(f"Answer: {result_dict['answer'] if 'answer' in result_dict else ''}")
                        logger.info(f"Output: {result_dict['output']}")
                        logger.info(f"Parsed output: {result_dict['parsed_output']}")
        finally:
            if 'futures' in locals():
                for future in futures:
                    if not future.done():
                        future.cancel()
            if not fout.closed:
                fout.close()

    end_time = time.time()
    mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    logger.info(f"Memory usage: {mem_usage/1000**3:.02f} GB")
    logger.info(f"Throughput: {len(results) / (end_time - start_time):.02f} samples/s")

    if args.count_tokens:
        logger.info(f"----{dataset}----\nAverage input length: {np.mean(metrics['input_len']):.02f}, std input length: {np.std(metrics['input_len']):.02f}, max input length: {max(metrics['input_len'])}, min input length: {min(metrics['input_len'])}\n----returning----")
        return output_path

    if len(results) == 0:
        logger.error("No results to evaluate, something went wrong, returning...")
        return output_path

    averaged_metrics = {k: np.mean(v)*(100 if "_len" not in k else 1) for k, v in metrics.items()}

    logger.info("Averaged metrics:")
    for k, v in averaged_metrics.items():
        logger.info(f"{k}: {v:.02f}")

    # for k, v in metrics.items():
    #     assert len(results) == len(v) and len(v) == len(data["data"]), f"Wrong metrics {k} collections"

    output = {
        "args": args.__dict__,
        "data": [r for r in results if r["parsed_output"] != ""],
        "metrics": metrics,
        "averaged_metrics": averaged_metrics,
        "memory_usage": mem_usage,
        "throughput": len(results) / (end_time - start_time),
    }

    if args.output_dir is not None:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
        with open(output_path + ".score", "w") as f:
            json.dump(output["averaged_metrics"], f, indent=4)
        logger.info(f"done, results are written to {output_path}")

    return output_path


def main():
    args = parse_arguments()

    logger.info(f"Arguments: {args}")
    assert args.model_name_or_path is not None
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_sample:
        if args.temperature != 0.0:
            logger.warning("do_sample is set to false but temperature is not 0, do_sample will overwrite temperature")

    model = load_LLM(args)

    datasets = args.datasets.split(",")
    test_files = args.test_files.split(",")
    max_lengths = ([int(args.input_max_length)] * len(datasets)) if isinstance(args.input_max_length, int) or len(args.input_max_length.split(",")) == 1 else [int(l) for l in args.input_max_length.split(",")]
    gen_lengths = ([int(args.generation_max_length)] * len(datasets)) if isinstance(args.generation_max_length, int) or len(args.generation_max_length.split(",")) == 1 else [int(l) for l in args.generation_max_length.split(",")]
    assert len(test_files) == len(max_lengths)
    test_length_list = [int(l) * 1024 for l in args.test_length.split(",")]

    for dataset, test_file, max_length, gen_length in zip(datasets, test_files, max_lengths, gen_lengths):
        if max_length not in test_length_list:
            continue
        args.datasets = dataset
        args.test_files = test_file
        args.input_max_length = max_length
        args.generation_max_length = gen_length
        model.max_length = max_length
        model.generation_max_length = gen_length

        try: 
            run_test(args, model, dataset, test_file)
        except Exception as e:
            # in case we run into some kind of error 
            logger.exception(e)
            logger.error(f"Error in {dataset}, continuing...")
            if args.debug:
                raise e

if __name__ == "__main__":
    main()