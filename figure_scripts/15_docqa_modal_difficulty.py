from utils import *
import matplotlib

#
config_files = ["configs/text_docqa_all.yaml", "configs/docqa_all.yaml"]

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
        if "mmlongdoc" in t:
            dataset_configs.append(
                {"dataset": d, "test_name": os.path.basename(os.path.splitext(t)[0]), "input_max_length": int(l),
                 "generation_max_length": int(g), "max_test_samples": c['max_test_samples'],
                 'use_chat_template': c['use_chat_template']})
print(dataset_configs)

models_configs = [
    {"model": "Qwen2.5-VL-3B-Instruct", "use_chat_template": True, "training_length": 32768},
    {"model": "Qwen2.5-VL-7B-Instruct", "use_chat_template": True, "training_length": 32768},
    {"model": 'Qwen2.5-VL-32B-Instruct', "use_chat_template": True, "training_length": 32768},

    {"model": "Qwen2.5-3B-Instruct_yarn", "use_chat_template": True, "training_length": 131072},
    {"model": "Qwen2.5-7B-Instruct_yarn", "use_chat_template": True, "training_length": 131072},
    {"model": 'Qwen2.5-32B-Instruct_yarn', "use_chat_template": True, "training_length": 131072},

    {"model": "gemma-3-4b-it", "use_chat_template": True, "training_length": 131072},
    {"model": "gemma-3-12b-it", "use_chat_template": True, "training_length": 131072},
    {"model": "gemma-3-27b-it", "use_chat_template": True, "training_length": 131072},
]
result_dir = "/home/zhaowei.wang/data_dir/mmlb_result_backup/analysis_models"
for model in models_configs:
    model["output_dir"] = f"{result_dir}/{model['model']}"

new_dfs = []
metrics = ["doc_qa", "text_score", "mm_score"]
dataset_to_metrics["text_doc"] = metrics
dataset_to_metrics["mmlongdoc"] = metrics

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
            new_dfs.append({**asdict(args), **model,
                        "metric name": k, "metric": m,
                        "dataset_simple": dsimple + " " + k,
                        "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                        })

new_df = pd.DataFrame(new_dfs)

import utils
chosen_models = [mc["model"] for mc in models_configs]
utils.custom_avgs = {}
utils.dataset_name_replace = {}
all_doc_df = process_df(new_df, chosen_models=chosen_models)


text_doc_metric_list = [f"text_doc {m}" for m in metrics]
text_doc_df = all_doc_df[['model', 'input_max_length'] + text_doc_metric_list].copy()
text_doc_df['model'] = text_doc_df['model'] + '_text'
for curr_metric in metrics:
    text_doc_df.rename(columns={f'text_doc {curr_metric}': f'mmlongdoc {curr_metric}'}, inplace=True)

doc_metric_list = [f"mmlongdoc {m}" for m in metrics]
doc_df = all_doc_df[['model', 'input_max_length'] + doc_metric_list].copy()
doc_df = doc_df[~doc_df['model'].str.contains('yarn')]

lf_df = pd.concat([text_doc_df, doc_df])
lf_df.reset_index(drop=True, inplace=True)

# plot specific ones in a row, for formatting in the paper
length_datasets = doc_metric_list
title_list = ["MMLB-Doc (All)", "Text-Pure Cases", "Vision-Needed Cases"] # "Layout/Table/Chart/Image"]

ncols = 3
nrows = (len(length_datasets) - 1) // ncols + 1

custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#4A895B'])
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=False)
fig.set_size_inches((20, nrows * 5))
plt.rc('axes', unicode_minus=False)
plt.rcParams.update({'axes.unicode_minus': False})

base_index_order = [
    'Qwen2.5-VL-7B-Inst', 'Qwen2.5-VL-7B-Inst_text', 'Qwen2.5-7B-Instruct_yarn_text',
    'Qwen2.5-VL-32B-Inst', 'Qwen2.5-VL-32B-Inst_text', 'Qwen2.5-32B-Instruct_yarn_text',
    "Gemma3-27B", "Gemma3-27B_text",
] # "Gemma3-12B", "Gemma3-12B_triviaqa",

for i, dataset in enumerate(length_datasets):
    if nrows > 1:
        a = ax[i // ncols][i % ncols]
    elif ncols > 1:
        a = ax[i]
    else:
        a = ax

    tdf = lf_df[lf_df.input_max_length > 4096]
    tdf = tdf.pivot_table(index="model", columns="input_max_length", values=dataset)
    tdf = tdf.reindex(base_index_order)

    sns_g = sns.heatmap(
        tdf, annot=True, cmap=custom_cmap, fmt=".1f", yticklabels=True,
        ax=a, annot_kws={"fontsize": 22},
        cbar=False
    )
    sns_g.set_title(title_list[i], fontsize=28)

    sns_g.set_ylabel("")
    sns_g.set_xlabel("")
    # "Gemma3-12B", '$\\diamond$ w/ name',
    written_index = ['Qwen2.5-VL-7B', '$\\diamond$ w/ OCR', '$\\diamond$ w/ LLM',
                     'Qwen2.5-VL-32B', '$\\diamond$ w/ OCR', '$\\diamond$ w/ LLM',
                     "Gemma3-27B", '$\\diamond$ w/ OCR',]
    sns_g.set_yticklabels(written_index, size=26)
    xticks_map = {"8192": '8k', "16384": '16k', "32768": '32k', "65536":'64k', "131072":'128k'}
    sns_g.set_xticklabels([xticks_map[st.get_text()] for st in sns_g.get_xticklabels()], size=26)

    # idx, start, end
    a.hlines([3, 6, 8], 0, 6, color="0.95", linestyle="-", linewidth=3)

[fig.delaxes(a) for a in ax.flatten() if not a.has_data()]

plt.tight_layout()
plt.subplots_adjust(wspace=0.20)
file_path = os.path.join(project_root, f"figures/15_text_doc_difficulty.pdf")
plt.savefig(file_path, dpi=500, format="pdf")
plt.show()