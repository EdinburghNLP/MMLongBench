import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(file_path))
# sys.path.append(os.path.join(project_root, "figure_scripts"))
from arguments import parse_arguments
from datasets import load_dataset
import random
from tqdm import tqdm


def load_vrag(args, path, max_test_samples=None):
    """
    Load the data for vision rag
    """
    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    def update(sample):
        return {"image_list": [sample["image"]]}

    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return data


def load_visual_haystack(args, path, max_test_samples=None):
    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    def update(sample):
        return {"image_list": sample["ctxs"]}

    data = data.map(update, num_proc=args.preprocessing_num_workers)

    return data


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}
def load_mm_niah_text(args, path, max_test_samples=None):
    path = os.path.join(args.test_file_root, path)
    with open(path) as fin:
        data = [json.loads(line) for line in tqdm(fin.readlines())]

    return data


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}"
def load_mm_niah_image_count(args, path, max_test_samples=None):
    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    def update(sample):
        image_list = sample["image_list"] + sample["needle_image_list"]
        return {"image_list": image_list}

    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return data


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}"
# used for multiple choices problems in mm-niah-image
def load_mm_niah_image_mc(args, path, max_test_samples=None):
    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    def update(sample):
        image_list = sample["image_list"] + sample["choices_image"]
        return {"image_list": image_list}

    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return data


def load_icl_er(args, path, max_test_samples=None):
    path = os.path.join(args.test_file_root, path)
    with open(path) as fin:
        data = json.load(fin)

    exemplar_list_by_domain = {domain: data_dict["exemplar_list"] for domain, data_dict in data.items()}
    exemplar_list_by_domain = {domain: [[item["image"] for sublist in exemplar for item in sublist] for exemplar in exemplar_list] for domain, exemplar_list in exemplar_list_by_domain.items()}

    test_examples = [{"domain": domain, "example": example} for domain, data_dict in data.items() for example in
                     data_dict["test_example"]]

    def update(sample):
        domain, example = sample["domain"], sample["example"]
        exemplar_list = exemplar_list_by_domain[domain]

        sampled_exemplar = random.choice(exemplar_list)
        image_list = sampled_exemplar + [example["image"]]

        return {"image_list": image_list}

    from datasets import Dataset as HFDataset
    test_examples = HFDataset.from_list(test_examples)
    test_examples = test_examples.map(update, num_proc=args.preprocessing_num_workers)

    return test_examples


def load_gov_report(args, path, max_test_samples=None):
    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    return data


def load_multi_lexsum(args, path, max_test_samples=None):
    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]
    return data


def load_doc_qa(args, path, max_test_samples=None):
    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    def update(sample):
        image_list = sample["page_list"]
        return {"image_list": image_list}

    data = data.map(update, num_proc=args.preprocessing_num_workers)

    return data


def load_data(args, dataset, path=None):
    if "infoseek" in dataset or "viquae" in dataset:
        data = load_vrag(args, path, max_test_samples=args.max_test_samples)
    elif "vh_single" in dataset or "vh_multi" in dataset:
        data = load_visual_haystack(args, path, max_test_samples=args.max_test_samples)
    elif "mm_niah" in dataset:
        if "text" in path:
            data = load_mm_niah_text(args, path, max_test_samples=args.max_test_samples)
        else:
            if "retrieval" in path or "reasoning" in path:
                data = load_mm_niah_image_mc(args, path, max_test_samples=args.max_test_samples)
            else:
                data = load_mm_niah_image_count(args, path, max_test_samples=args.max_test_samples)
    elif any(key in dataset for key in ["cars196", "food101", "inat2021", "sun397"]):
        data = load_icl_er(args, path, max_test_samples=args.max_test_samples)
    elif "gov-report" in dataset:
        data = load_gov_report(args, path, max_test_samples=args.max_test_samples)
    elif "lexsum" in dataset:
        data = load_multi_lexsum(args, path, max_test_samples=args.max_test_samples)
    elif any(key in dataset for key in ["longdocurl", "mmlongdoc", "slidevqa"]):
        data = load_doc_qa(args, path, max_test_samples=args.max_test_samples)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return data

args = parse_arguments()

config_files = ["configs/vrag_all.yaml", "configs/vh_all.yaml", "configs/mm_niah_text_all.yaml", "configs/mm_niah_image_all.yaml",
                "configs/icl_all.yaml", "configs/summ_all.yaml", "configs/docqa_all.yaml"]

config_files = [os.path.join(project_root, config) for config in config_files]

dataset_configs = []
for file in config_files:
    c = yaml.safe_load(open(file))

    if isinstance(c["generation_max_length"], int):
        c["generation_max_length"] = ",".join([str(c["generation_max_length"])] * len(c["datasets"].split(",")))
    for d, t, l, g in zip(
            c['datasets'].split(','), c['test_files'].split(','), c['input_max_length'].split(','),
            c['generation_max_length'].split(',')):
        dataset_configs.append(
            {"dataset": d, "test_file": t, "input_max_length": int(l),
             "generation_max_length": int(g), "max_test_samples": c['max_test_samples'],
             'use_chat_template': c['use_chat_template']})


def process_single_config(curr_config):
    data = load_data(args, curr_config["dataset"], curr_config["test_file"])
    image_len_list = [len(d["image_list"]) for d in data]
    return {"dataset": curr_config["dataset"],
            "input_max_length": curr_config["input_max_length"],
            "mean_length": sum(image_len_list) / len(image_len_list),
            "max_length": max(image_len_list),
            "min_length": min(image_len_list),
            "std_length": np.std(image_len_list, ddof=1) if len(image_len_list) > 1 else 0}

# with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
#future_results = executor.map(process_single_config, dataset_configs)
all_len_results = []
for config in dataset_configs:
    print(config["dataset"], config["input_max_length"])
    all_len_results.append(process_single_config(config))

all_len_df = pd.DataFrame(all_len_results)


dataset_name_replace = {
    # RAG
    'infoseek': "InfoSeek",
    'viquae': "ViQuAE",

    # Synthetic/RULER
    'vh_single': "VH-Single",
    'vh_multi': "VH-Multi",
    'mm_niah_retrieval-text': 'MM-NIAH-Ret (T)',
    'mm_niah_counting-text': "MM-NIAH-Count (T)",
    'mm_niah_reasoning-text': "MM-NIAH-Reason (T)",
    'mm_niah_retrieval-image': "MM-NIAH-Ret (I)",
    'mm_niah_counting-image': "MM-NIAH-Count (I)",
    'mm_niah_reasoning-image': "MM-NIAH-Reason (I)",

    # ICL
    'cars196': "Stanford Cars",
    'food101': "Food101",
    'sun397': "SUN397",
    'inat2021': "Inat2021",

    # Summ
    'gov-report': "GovReport",
    'multi-lexsum': "Multi-LexSum",

    # DocQA
    'mmlongdoc': "MMLongBench-Doc",
    'longdocurl': "LongDocURL",
    'slidevqa': "SlideVQA",
}

column_rename_map = {
    8192: "8K",
    16384: "16K",
    32768: "32K",
    65536: "64K",
    131072: "128K"
}

pivoted_df = all_len_df.pivot_table(
    index='dataset',
    columns='input_max_length',
    values=['mean_length', 'std_length']
)

pivoted_df = pivoted_df.rename(index=dataset_name_replace)
pivoted_df = pivoted_df.reindex([v for k, v in dataset_name_replace.items()])
pivoted_df = pivoted_df.rename(columns=column_rename_map)
pivoted_df.columns.name = "Data Length"
pivoted_df.index.name = ""


formatted_df = pd.DataFrame(index=pivoted_df.index, columns=['8K', '16K', '32K', '64K', '128K'])

for col_len in pivoted_df.columns.levels[1]:
    mean_col = ('mean_length', col_len)
    std_col = ('std_length', col_len)

    assert mean_col in pivoted_df.columns and std_col in pivoted_df.columns
    formatted_df[col_len] = '-' # Or some placeholder

    for idx in pivoted_df.index: # e.g., 'InfoSeek', 'VH-Single'
        mean_val = pivoted_df.loc[idx, mean_col]
        std_val = pivoted_df.loc[idx, std_col]

        if pd.isna(mean_val):
            formatted_df.loc[idx, col_len] = '-' # Use na_rep for NaN mean
        elif pd.isna(std_val):
            # Just show mean if std is NaN or zero
            formatted_df.loc[idx, col_len] = f"{mean_val:.1f}"
        else:
            # Format as mean_{std} using LaTeX subscript
            formatted_df.loc[idx, col_len] = f"{mean_val:.1f}$_{{{std_val:.1f}}}$"

num_value_columns = len(formatted_df.columns)
col_format = "l" + "r" * num_value_columns

latex_string = formatted_df.to_latex(
    buf=None,
    column_format=col_format,
    header=True,
    index=True,
    na_rep='-',
    escape=False,
    caption="Average number of images per example (Mean$_{Std Dev}$) by dataset and input length",
    label="tab:mean_std_image_number",
    position='t!'
)

print("\nGenerated LaTeX Code with Subscripts:")
print(latex_string)





