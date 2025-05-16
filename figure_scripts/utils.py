import os
import re
import sys
import json
import math
import yaml
from collections import defaultdict
from dataclasses import dataclass, asdict
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoTokenizer, LlamaConfig
import io
from io import StringIO

# Import the project root first
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_path))

# THIS IS THE KEY ARGUMENT
result_dir = "/home/zhaowei.wang/data_dir/mmlb_result" # "/Users/wangzhaowei/Downloads/mmlb_result" "/home/zhaowei.wang/data_dir/mmlb_result"

dataset_to_metrics = {
    "infoseek": "sub_em",
    "viquae": "sub_em",

    "vh_single": "acc",
    "vh_multi": "acc",
    "mm_niah_retrieval-text": "sub_em",
    "mm_niah_counting-text": "soft_acc",
    "mm_niah_reasoning-text": "sub_em",
    "mm_niah_retrieval-image": "mc_acc",
    "mm_niah_counting-image": "soft_acc",
    "mm_niah_reasoning-image": "mc_acc",

    "cars196": "cls_acc",
    "food101": "cls_acc",
    "inat2021": "cls_acc",
    "sun397": "cls_acc",

    "gov-report": "gpt4-flu-f1", # , "rougeLsum_f1"]
    "multi-lexsum": "gpt4-flu-f1", # "rougeLsum_f1"],

    "longdocurl": "doc_qa",
    "mmlongdoc": "doc_qa",
    "slidevqa": "doc_qa",
}
dataset_to_metrics = {k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()}

@dataclass
class arguments:
    input_max_length: int = None
    generation_max_length: int = None
    generation_min_length: int = 0
    max_test_samples: int = None
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    use_chat_template: bool = None
    seed: int = 42
    test_name: str = None
    dataset: str = None
    output_dir: str = None

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_path(self):
        path = os.path.join(self.output_dir,
                            "{args.dataset}_{args.test_name}_in{args.input_max_length}_size{args.max_test_samples}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}.json".format(
                                args=self))

        if self.dataset in {"multi-lexsum", "gov-report"}: # we don't fall back to rougeLSum even if there is no gpt4o scores
            return path.replace(".json", "-gpt4eval_o.json")


        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None

    def get_averaged_metric(self):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            return None
        with open(path) as f:
            results = json.load(f)

        _, metric = self.get_metric_name()
        if path.endswith(".score"):
            if any([m not in results for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
        else:
            if any([m not in results["averaged_metrics"] for m in metric]):
                print("metric doesn't exist")
                return None
            s = {m: results['averaged_metrics'][m] for m in metric}

        # s = {m: v * (100 if m == "gpt4-f1" else 1) for m, v in s.items()}

        # flip the binary_acc
        if "acc" in s:
            if s["acc"] < 50:
                curr_acc = s["acc"]
                print(f"Flipping binnary acc from {curr_acc} to {100 - curr_acc}")
                s["acc"] = 100 - s["acc"]

        print("found scores:", s)
        return s

    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", '')
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)

        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        # flip acc
        if metric == "acc":
            print(f"Flipping binary acc")
            dfs['metric'] = dfs['metric'].apply(lambda x: 1 - x if x < 0.5 else x)

        return dfs.to_dict("records")


config_files = ["configs/vrag_all.yaml", "configs/vh_all.yaml", "configs/mm_niah_text_all.yaml",
                "configs/mm_niah_image_all.yaml", "configs/icl_all.yaml", "configs/summ_all.yaml",
                "configs/docqa_all.yaml"]

config_files = [os.path.join(project_root, config) for config in config_files]

print(os.getcwd())
dataset_configs = []
for file in config_files:
    c = yaml.safe_load(open(file))

    if isinstance(c["generation_max_length"], int):
        c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
    for d, t, l, g in zip(
            c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','),
            c['generation_max_length'].split(',')):
        dataset_configs.append(
            {"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]), "input_max_length": int(l),
             "generation_max_length": int(g), "max_test_samples": c['max_test_samples'],
             'use_chat_template': c['use_chat_template']})
print(dataset_configs)

######################## VARIABLE DEFINITION START ########################

models_configs = [
    {"model": "gpt-4o-2024-11-20", "use_chat_template": True, "training_length": 128000},
    {"model": "claude-3-7-sonnet-20250219", "use_chat_template": True, "training_length": 200000},
    {"model": "gemini-2.0-flash-001", "use_chat_template": True, "training_length": 1048576},
    {"model": 'gemini-2.0-flash-thinking-exp-01-21', "use_chat_template": True, "training_length": 1048576},
    {"model": 'gemini-2.5-flash-preview-04-17', "use_chat_template": True, "training_length": 1048576},
    {"model": "gemini-2.5-pro-preview-03-25", "use_chat_template": True, "training_length": 1048576},

    # Qwen2.5-VL and Qwen2-VL official repo shows 32K
    {"model": "Qwen2-VL-2B-Instruct", "use_chat_template": True, "training_length": 32768},
    {"model": "Qwen2-VL-7B-Instruct", "use_chat_template": True, "training_length": 32768},
    {"model": 'Qwen2-VL-72B-Instruct-AWQ', "use_chat_template": True, "training_length": 32768},

    # {"model": 'QVQ-72B-Preview-AWQ', "use_chat_template": True, "training_length": 32768},

    {"model": "Qwen2.5-VL-3B-Instruct", "use_chat_template": True, "training_length": 32768},
    {"model": "Qwen2.5-VL-7B-Instruct", "use_chat_template": True, "training_length": 32768},
    {"model": 'Qwen2.5-VL-32B-Instruct', "use_chat_template": True, "training_length": 32768},
    {"model": 'Qwen2.5-VL-72B-Instruct-AWQ', "use_chat_template": True, "training_length": 32768},

    # see here: https://github.com/OpenGVLab/InternVL/issues/272
    {"model": "InternVL2-1B", "use_chat_template": True, "training_length": 8192},
    {"model": "InternVL2-2B", "use_chat_template": True, "training_length": 8192},
    {"model": "InternVL2-4B", "use_chat_template": True, "training_length": 8192},
    {"model": "InternVL2-8B", "use_chat_template": True, "training_length": 8192},

    # https://huggingface.co/OpenGVLab/InternVL2_5-2B#progressive-scaling-strategy model card shows length
    {"model": "InternVL2_5-1B", "use_chat_template": True, "training_length": 16384},
    {"model": "InternVL2_5-2B", "use_chat_template": True, "training_length": 16384},
    {"model": "InternVL2_5-4B", "use_chat_template": True, "training_length": 16384},
    {"model": "InternVL2_5-8B", "use_chat_template": True, "training_length": 16384},
    {"model": "InternVL2_5-26B", "use_chat_template": True, "training_length": 16384},

    # In the paper https://arxiv.org/pdf/2504.10479: To support sequences of up to 32K tokens
    {"model": "InternVL3-1B", "use_chat_template": True, "training_length": 32768},
    {"model": "InternVL3-2B", "use_chat_template": True, "training_length": 32768},
    {"model": "InternVL3-8B", "use_chat_template": True, "training_length": 32768},
    {"model": "InternVL3-14B", "use_chat_template": True, "training_length": 32768},
    {"model": "InternVL3-38B", "use_chat_template": True, "training_length": 32768},

    # found in "Usage" part of model card https://huggingface.co/AIDC-AI/Ovis2-1B#usage and config https://huggingface.co/AIDC-AI/Ovis2-1B/blob/main/config.json#L52
    {"model": "Ovis2-1B", "use_chat_template": True, "training_length": 32768},
    {"model": "Ovis2-2B", "use_chat_template": True, "training_length": 32768},
    {"model": "Ovis2-4B", "use_chat_template": True, "training_length": 32768},
    {"model": "Ovis2-8B", "use_chat_template": True, "training_length": 32768},
    {"model": "Ovis2-16B", "use_chat_template": True, "training_length": 32768},
    {"model": "Ovis2-34B", "use_chat_template": True, "training_length": 32768},

    # the official blog and model cards
    {"model": "gemma-3-4b-it", "use_chat_template": True, "training_length": 131072},
    {"model": "gemma-3-12b-it", "use_chat_template": True, "training_length": 131072},
    {"model": "gemma-3-27b-it", "use_chat_template": True, "training_length": 131072},

    # idefics2 paper: OBELICS represents 45% ... a maximum sequence length of 2’048, and the base model (Mistral-Inst v0.1) supports 8K
    {"model": "idefics2-8b", "use_chat_template": True, "training_length": 8192},
    {"model": "idefics2-8b-chatty", "use_chat_template": True, "training_length": 8192},
    {"model": "Mantis-8B-Idefics2", "use_chat_template": True, "training_length": 8192},
    # Table 3 says 10K in Idefics3 paper (https://arxiv.org/pdf/2408.12637)
    {"model": "Idefics3-8B-Llama3", "use_chat_template": True, "training_length": 10240},

    # model cards at HF
    {"model": "Phi-3-vision-128k-instruct", "use_chat_template": True, "training_length": 131072},
    {"model": "Phi-3.5-vision-instruct", "use_chat_template": True, "training_length": 131072},
    {"model": "Phi-4-multimodal-instruct", "use_chat_template": True, "training_length": 131072},

    # model cards at HF
    {"model": "NVILA-Lite-2B-hf-preview", "use_chat_template": True, "training_length": 32768},
    {"model": "NVILA-Lite-8B-hf-preview", "use_chat_template": True, "training_length": 32768},

    # Official blog (https://mistral.ai/news/pixtral-12b) says 128K, and the model config
    {"model": "pixtral-12b", "use_chat_template": True, "training_length": 131072},
]

for model in models_configs:
    model["output_dir"] = f"{result_dir}/{model['model']}"


# pretty names for the models
model_name_replace = {
    'gpt-4o-2024-11-20': 'GPT-4o',
    'claude-3-7-sonnet-20250219': 'Claude-3.7-Sonnet',
    'gemini-2.0-flash-001': 'Gemini-2.0-Flash',
    'gemini-2.0-flash-thinking-exp-01-21': 'Gemini-2.0-Flash-T',
    'gemini-2.5-flash-preview-04-17': 'Gemini-2.5-Flash',
    'gemini-2.5-pro-preview-03-25': 'Gemini-2.5-Pro',

    'Qwen2-VL-2B-Instruct': 'Qwen2-VL-2B-Inst',
    'Qwen2-VL-7B-Instruct': 'Qwen2-VL-7B-Inst',
    'Qwen2-VL-72B-Instruct-AWQ': 'Qwen2-VL-72B-Inst',

    # 'QVQ-72B-Preview-AWQ': 'QVQ-72B',

    'Qwen2.5-VL-3B-Instruct': 'Qwen2.5-VL-3B-Inst',
    'Qwen2.5-VL-7B-Instruct': 'Qwen2.5-VL-7B-Inst',
    'Qwen2.5-VL-32B-Instruct': 'Qwen2.5-VL-32B-Inst',
    'Qwen2.5-VL-72B-Instruct-AWQ': 'Qwen2.5-VL-72B-Inst',

    "InternVL2-1B": "InternVL2-1B",
    "InternVL2-2B": "InternVL2-2B",
    "InternVL2-4B": "InternVL2-4B",
    "InternVL2-8B": "InternVL2-8B",

    "InternVL2_5-1B": "InternVL2.5-1B",
    "InternVL2_5-2B": "InternVL2.5-2B",
    "InternVL2_5-4B": "InternVL2.5-4B",
    "InternVL2_5-8B": "InternVL2.5-8B",
    "InternVL2_5-26B": "InternVL2.5-26B",

    "InternVL3-1B": "InternVL3-1B",
    "InternVL3-2B": "InternVL3-2B",
    "InternVL3-8B": "InternVL3-8B",
    "InternVL3-14B": "InternVL3-14B",
    "InternVL3-38B": "InternVL3-38B",

    "Ovis2-1B": "Ovis2-1B",
    "Ovis2-2B": "Ovis2-2B",
    "Ovis2-4B": "Ovis2-4B",
    "Ovis2-8B": "Ovis2-8B",
    "Ovis2-16B": "Ovis2-16B",
    "Ovis2-34B": "Ovis2-34B",

    "gemma-3-4b-it": "Gemma3-4B",
    "gemma-3-12b-it": "Gemma3-12B",
    "gemma-3-27b-it": "Gemma3-27B",

    "idefics2-8b": "Idefics2-8B",
    "idefics2-8b-chatty": "Idefics2-8B-C",
    "Mantis-8B-Idefics2": "Mantis-Idefics2",
    "Idefics3-8B-Llama3": "Idefics3-8B",

    "Phi-3-vision-128k-instruct": "Phi-3-Vision",
    "Phi-3.5-vision-instruct": "Phi-3.5-Vision",
    "Phi-4-multimodal-instruct": "Phi-4-Multimodal",

    "NVILA-Lite-2B-hf-preview": "NVILA-Lite-2B",
    "NVILA-Lite-8B-hf-preview": "NVILA-Lite-8B",

    "pixtral-12b": "Pixtral-12B",

}

dataset_name_replace = {
    "model": "Model",

    # RAG
    'infoseek sub_em': "InfoSeek",
    'viquae sub_em': "ViQuAE",

    # Synthetic/RULER
    'vh_single acc': "VH-Single",
    'vh_multi acc': "VH-Multi",
    'mm_niah_retrieval-text sub_em': 'MM-NIAH-Ret (T)',
    'mm_niah_counting-text soft_acc': "MM-NIAH-Count (T)",
    'mm_niah_reasoning-text sub_em': "MM-NIAH-Reason (T)",
    'mm_niah_retrieval-image mc_acc': "MM-NIAH-Ret (I)",
    'mm_niah_counting-image soft_acc': "MM-NIAH-Count (I)",
    'mm_niah_reasoning-image mc_acc': "MM-NIAH-Reason (I)",
    'mm_niah_retrieval': 'MM-NIAH-Ret',
    'mm_niah_counting': "MM-NIAH-Count",
    "mm_niah_reasoning": "MM-NIAH-Reason",

    # ICL
    'cars196 cls_acc': "Stanford Cars",
    'food101 cls_acc': "Food101",
    'sun397 cls_acc': "SUN397",
    'inat2021 cls_acc': "Inat2021",

    # Summ
    'gov-report gpt4-flu-f1': "GovReport",
    'multi-lexsum gpt4-flu-f1': "Multi-LexSum",

    # DocQA
    'mmlongdoc doc_qa': "MMLongBench-Doc",
    'longdocurl doc_qa': "LongDocURL",
    'slidevqa doc_qa': "SlideVQA",
}

custom_avgs = {
    "VRAG": ['infoseek sub_em', 'viquae sub_em'],
    "mm_niah_retrieval": ['mm_niah_retrieval-text sub_em', 'mm_niah_retrieval-image mc_acc'],
    "mm_niah_counting": ["mm_niah_counting-text soft_acc", "mm_niah_counting-image soft_acc"],
    "mm_niah_reasoning": ["mm_niah_reasoning-text sub_em", "mm_niah_reasoning-image mc_acc"],
    "NIAH": ["vh_single acc", "vh_multi acc", "mm_niah_retrieval", "mm_niah_counting", "mm_niah_reasoning"],
    "ICL": ['cars196 cls_acc', 'food101 cls_acc', 'inat2021 cls_acc', 'sun397 cls_acc'],
    "Summ": ['gov-report gpt4-flu-f1', 'multi-lexsum gpt4-flu-f1'], # ['gov-report rougeLsum_f1', 'multi-lexsum rougeLsum_f1'],
    "DocVQA": ['longdocurl doc_qa', 'mmlongdoc doc_qa', 'slidevqa doc_qa'],
    "Ours": ['VRAG', 'NIAH', 'ICL', 'Summ', 'DocVQA'],
}

full_table_models = [
    "GPT-4o",
    "Claude-3.7-Sonnet",
    "Gemini-2.0-Flash",
    'Gemini-2.0-Flash-T',
    'Gemini-2.5-Flash',
    "Gemini-2.5-Pro",

    "Qwen2-VL-2B-Inst",
    "Qwen2-VL-7B-Inst",
    'Qwen2-VL-72B-Inst',
    # 'QVQ-72B',
    "Qwen2.5-VL-3B-Inst",
    "Qwen2.5-VL-7B-Inst",
    'Qwen2.5-VL-32B-Inst',
    'Qwen2.5-VL-72B-Inst',

    "InternVL2-1B",
    "InternVL2-2B",
    "InternVL2-4B",
    "InternVL2-8B",
    "InternVL2.5-1B",
    "InternVL2.5-2B",
    "InternVL2.5-4B",
    "InternVL2.5-8B",
    "InternVL2.5-26B",
    "InternVL3-1B",
    "InternVL3-2B",
    "InternVL3-8B",
    "InternVL3-14B",
    "InternVL3-38B",

    "Ovis2-1B",
    "Ovis2-2B",
    "Ovis2-4B",
    "Ovis2-8B",
    "Ovis2-16B",
    "Ovis2-34B",

    "Gemma3-4B",
    "Gemma3-12B",
    "Gemma3-27B",

    "Idefics2-8B",
    "Idefics2-8B-C",
    "Mantis-Idefics2",
    "Idefics3-8B",

    "Phi-3-Vision",
    "Phi-3.5-Vision",
    "Phi-4-Multimodal",

    "NVILA-Lite-2B",
    "NVILA-Lite-8B",

    "Pixtral-12B",
]

main_table_models = [
    "GPT-4o",
    "Claude-3.7-Sonnet",
    "Gemini-2.0-Flash",
    'Gemini-2.0-Flash-T',
    'Gemini-2.5-Flash',
    "Gemini-2.5-Pro",

    'Qwen2-VL-72B-Inst',
    # 'QVQ-72B',
    "Qwen2.5-VL-7B-Inst",
    'Qwen2.5-VL-32B-Inst',
    'Qwen2.5-VL-72B-Inst',

    "InternVL2.5-26B",
    "InternVL3-8B",
    "InternVL3-14B",
    "InternVL3-38B",

    "Ovis2-8B",
    "Ovis2-16B",
    "Ovis2-34B",

    "Gemma3-12B",
    "Gemma3-27B",

    "Idefics3-8B",
    "Phi-4-Multimodal",
    "NVILA-Lite-8B",
    "Pixtral-12B",
]

######################## VARIABLE DEFINITION END ########################

result_exclusions = {'claude-3-7-sonnet-20250219':
        {'mm_niah_retrieval-text sub_em': [65536, 131072],
         'mm_niah_retrieval-image mc_acc': [65536, 131072],
         "mm_niah_counting-text soft_acc": [65536, 131072],
         "mm_niah_counting-image soft_acc": [65536, 131072],
         "mm_niah_reasoning-text sub_em": [65536, 131072],
         "mm_niah_reasoning-image mc_acc": [65536, 131072],
         'vh_single acc': [65536, 131072],
         'vh_multi acc': [65536, 131072],
         'cars196 cls_acc': [32768, 65536, 131072],
         'food101 cls_acc': [32768, 65536, 131072],
         'sun397 cls_acc': [32768, 65536, 131072],
         'inat2021 cls_acc': [32768, 65536, 131072],
         'mmlongdoc doc_qa': [131072],
         'longdocurl doc_qa': [131072],
         'slidevqa doc_qa': [131072]}
    }

### collect results
dfs = []

for m_idx, model in enumerate(models_configs):
    args = arguments()
    for dataset in dataset_configs:
        args.update(dataset)
        args.update(model)

        # parse the metrics
        metric = args.get_averaged_metric()
        dsimple, mnames = args.get_metric_name()

        if metric is None:
            print("failed:", args.get_path()) # will be np.nan when using DataFrame
            continue
        for k, m in metric.items():
            dfs.append({**asdict(args), **model,
                        "metric name": k, "metric": m,
                        "dataset_simple": dsimple + " " + k,
                        "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                        })

all_df = pd.DataFrame(dfs)

## preprocessing for all the following codes
model_lengths = {m['model']: m['training_length'] for m in models_configs}
l_maps = {128000: "128k", 131072: '128k', 200000: '200k', 1048576: '1m', 1010000: '1m', 2097152: '2m', 8192: '8k', 80000: '80k', 10240: '10k', 16384: '16k', 32768: '32k', 262144: '256k', 65536: '64k', 524288: '512k',}
model_lengths = {model_name_replace[k]: l_maps[v] for k, v in model_lengths.items()}


def process_df(lf_df, chosen_models=None, chosen_datasets=None):
    # convert into longform dataframe
    # chosen models should be the name before replacement, set to None to keep everything
    # same for chosen datasets
    if chosen_models is not None:
        lf_df = lf_df[lf_df.model.isin(chosen_models)]
    lf_df = lf_df.replace(model_name_replace)

    lf_df = lf_df.pivot_table(index=["input_max_length", "model"], columns="dataset_simple", values="metric", sort=False)
    lf_df = lf_df.reset_index()
    # now each row is a model at a length, and later columns each corresponds to a datapoint

    # we can add some additional aggregates here
    for k, v in custom_avgs.items():
        lf_df[k] = lf_df[v].mean(skipna=False, axis=1)

    # the first two columns are model and input lengths
    lf_df = lf_df[['model', 'input_max_length'] + (chosen_datasets if chosen_datasets is not None else lf_df.columns.tolist()[2:])]
    lf_df = lf_df.rename(columns=dataset_name_replace)
    return lf_df
