#!/usr/bin/env python3
import os
import subprocess
import time
from multiprocessing import Pool, Manager
from itertools import product
import logging
import argparse
import queue

result_base_path = "/workspace/zw/data_dir/mmlb_result"
def setup_logger(args):
    base_output_dir = f"{result_base_path}/{args.model_name.split('/')[-1]}"
    if args.use_yarn:
        base_output_dir = f"{result_base_path}/{args.model_name.split('/')[-1]}_yarn"
    if args.v2pe_step:
        base_output_dir = f"{result_base_path}/{args.model_name.split('/')[-1]}_step{args.v2pe_step}"

    os.makedirs(base_output_dir, exist_ok=True)
    log_file = os.path.join(base_output_dir, f"eval_log.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger


def worker(args):
    (gpu_group, model_name, task_queue, results,
     logger, global_args) = args

    while True:
        try:
            task, length = task_queue.get(block=False)
            if global_args.use_yarn:
                output_dir = f"{result_base_path}/{model_name.split('/')[-1]}_yarn"
            else:
                output_dir = f"{result_base_path}/{model_name.split('/')[-1]}"

            cmd = f"CUDA_VISIBLE_DEVICES={gpu_group} python eval.py --config configs/{task}_all.yaml " \
                  f"--model_name_or_path {model_name} " \
                  f"--output_dir {output_dir} " \
                  f"--test_file_root /workspace/zw/data_dir/mmlb_data " \
                  f"--image_file_root /workspace/zw/data_dir/mmlb_image " \
                  f"--num_workers {global_args.num_workers} --test_length {length}"
                # TODO replace test_file_root and image_file_root with our own path

            if global_args.image_resize is not None:
                cmd += f" --image_resize {global_args.image_resize}"
            if global_args.do_image_splitting != "None":
                cmd += f" --do_image_splitting {global_args.do_image_splitting}"
            if global_args.max_image_num is not None:
                cmd += f" --max_image_num {global_args.max_image_num}"
            if global_args.do_prefill:
                cmd += f" --do_prefill"
            if global_args.no_bf16:
                cmd += f" --no_bf16"
            if global_args.load_in_8bit:
                cmd += f" --load_in_8bit"
            if global_args.use_yarn:
                cmd += f" --use_yarn"
            if global_args.vision_batch_size is not None:
                cmd += f" --vision_batch_size {global_args.vision_batch_size}"
            if global_args.v2pe_step is not None:
                cmd += f" --v2pe_step {global_args.v2pe_step}"

            task_output_dir = os.path.join(output_dir, f"{task}_{length}")
            os.makedirs(task_output_dir, exist_ok=True)
            error_log_path = os.path.join(task_output_dir, "error.log")
            stdout_log_path = os.path.join(task_output_dir, "stdout.log")

            logger.info(f"GPU {gpu_group}: Started {task} with length {length}\nRunning command: {cmd}")
            start_time = time.time()

            with open(stdout_log_path, 'w') as stdout_file, open(error_log_path, 'w') as stderr_file:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True
                )
                return_code = process.wait()

            if return_code != 0:
                logger.info(f"GPU {gpu_group}: Error in {task} with length {length}, code: {return_code}")
                success = False
            else:
                elapsed_time = time.time() - start_time
                logger.info(f"GPU {gpu_group}: Completed {task} with length {length} in {elapsed_time:.2f}s")
                success = True

            results.append((task, length, success))

        except Exception as e:
            if isinstance(e, queue.Empty) or 'Empty' in str(type(e)):
                logger.info(f"GPU {gpu_group}: No more tasks, exiting")
                break
            else:
                logger.info(f"GPU {gpu_group}: Exception - {str(e)}")
                time.sleep(1)  # avoiding high GPU utilization

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--task_list", type=str, default="vrag,vh,mm_niah_text,mm_niah_image,icl,summ,docqa")
    parser.add_argument("--length_list", type=str, default="8,16,32,64,128")
    parser.add_argument("--gpu_list", type=str, default="1,2,3,4")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_group_size", type=int, default=1)
    parser.add_argument("--image_resize", type=float, default=None)
    parser.add_argument("--do_image_splitting", type=str, choices=["True", "False", "None"], default="None")
    parser.add_argument("--max_image_num", type=int, default=None)
    parser.add_argument("--do_prefill", action="store_true", help="prefill the context to save memory")
    parser.add_argument("--no_bf16", action="store_true", help="use fp16")
    parser.add_argument("--load_in_8bit", action="store_true", help="use bnb int8")
    parser.add_argument("--use_yarn", action="store_true", help="use yarn for qwen2.5-vl")
    parser.add_argument("--vision_batch_size", type=int, default=None)
    parser.add_argument("--v2pe_step", type=int, default=None)
    args = parser.parse_args()

    task_list = args.task_list.split(",")
    length_list = [int(l) for l in args.length_list.split(",") if l] # reverse order, better to finish longer tasks first
    gpu_list = [i for i in args.gpu_list.split(",") if i]
    length_list = sorted(length_list, reverse=True)

    logger = setup_logger(args)
    logger.info(str(args))

    with Manager() as manager:
        task_queue = manager.Queue()
        results = manager.list()

        total_tasks = 0
        logger.info("Task list:")
        for length, task in product(length_list, task_list):
            task_queue.put((task, length))
            logger.info(f"{total_tasks}. {task}-{length}")
            total_tasks += 1
        logger.info(f"Total tasks: {total_tasks}")

        assert len(gpu_list) % args.gpu_group_size == 0
        gpu_group_list = [gpu_list[i: i + args.gpu_group_size]
                          for i in range(0, len(gpu_list), args.gpu_group_size)]
        gpu_group_list = [",".join(gpu_group) for gpu_group in gpu_group_list]

        args_list = [(gpu_group, args.model_name, task_queue,
                      results, logger, args) for gpu_group in gpu_group_list]

        start_time = time.time()
        with Pool(processes=len(gpu_group_list)) as pool:
            pool.map(worker, args_list)

        total_time = time.time() - start_time

        success_count = sum(1 for _, _, success in results if success)
        logger.info(f"Completed: {success_count}/{total_tasks} tasks in {total_time:.2f} seconds")

        if success_count < total_tasks:
            logger.info("Failed tasks:")
            for task, length, success in results:
                if not success:
                    logger.info(f" Failed Config: {task} (length {length})")


if __name__ == "__main__":
    main()