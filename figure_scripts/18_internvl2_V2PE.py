from utils import *

dataset_to_metrics["gov-report"] = ["rougeLsum_f1"]
dataset_to_metrics["multi-lexsum"] = ["rougeLsum_f1"]
custom_avgs["Summ"] = ['gov-report rougeLsum_f1', 'multi-lexsum rougeLsum_f1']

models_configs = [
    {"model": "InternVL2-2B", "use_chat_template": True, "training_length": 8192},
    {"model": "V2PE-256K_16", "use_chat_template": True, "training_length": 8192},
    {"model": "V2PE-256K", "use_chat_template": True, "training_length": 8192},
    {"model": "V2PE-256K_256", "use_chat_template": True, "training_length": 8192},
]

for model in models_configs:
    model["output_dir"] = f"/home/zhaowei.wang/data_dir/mmlb_result_backup/analysis_models/{model['model']}"

models_configs[0]["output_dir"] = f"/home/zhaowei.wang/data_dir/mmlb_result/InternVL2-2B"

model_name_replace.update({
    "V2PE-256K_16": "V2PE (16)",
    "V2PE-256K": "V2PE (64)",
    "V2PE-256K_256": "V2PE (256)"
})

curr_table_models = [
    "InternVL2-2B",
    "V2PE (16)",
    "V2PE (64)",
    "V2PE (256)",
]

new_dfs = []

for m_idx, model in enumerate(models_configs):
    args = arguments()
    for dataset in dataset_configs:
        args.update(dataset)
        args.update(model)

        # parse the metrics
        if dataset["dataset"] in {"multi-lexsum", "gov-report"}:
            path = args.get_path()
            path = path.replace("-gpt4eval_o.json", ".json") + ".score"
            with open(path) as f:
                results = json.load(f)
            metric = {"rougeLsum_f1": results["rougeLsum_f1"]}
        else:
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

all_dfs = new_dfs
all_df = pd.DataFrame(all_dfs)

# Figure: positional extrploation methods
main_table_datasets = [
    "VRAG",
    "NIAH",
    'ICL',
    "Summ",
    "DocVQA",
    "Ours"
]

# plot specific ones in a row, for formatting in the paper
lf_df = process_df(all_df)
length_datasets = main_table_datasets

ncols = 3
nrows = (len(length_datasets) - 1) // ncols + 1
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#4A895B'])

fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=False)
fig.set_size_inches((ncols * 8, 7)) # helmet has 45 models for 40 height, we have 46 models

plt.rc('axes', unicode_minus=False)
plt.rcParams.update({'axes.unicode_minus': False})

for i, dataset in enumerate(length_datasets):
    if nrows > 1:
        a = ax[i // ncols][i % ncols]
    else:
        a = ax[i]

    new_index = curr_table_models

    tdf = lf_df[lf_df.input_max_length > 4096]
    tdf = tdf.pivot_table(index="Model", columns="input_max_length", values=dataset)
    tdf = tdf.reindex(new_index)

    # process the scores
    annot_matrix = tdf.copy()
    tdf = tdf.applymap(lambda x: x if not pd.isna(x) else 0)
    annot_matrix = annot_matrix.applymap(lambda x: "N/A" if pd.isna(x) else f"{x:.1f}")

    sns_g = sns.heatmap(
        tdf, annot=annot_matrix, cmap=custom_cmap, fmt="", yticklabels=True,
        ax=a, annot_kws={"fontsize": 23.5},
        cbar=False
    )
    sns_g.set_title(dataset if dataset != "Ours" else "Avg.", fontsize=34)

    sns_g.set_ylabel("")
    sns_g.set_xlabel("")

    new_index = [x.replace("-Inst", '') for x in new_index]
    new_index = ["$\\diamond$ w/ Yarn" if "(Y)" in x else x for x in new_index ]

    sns_g.set_yticklabels(new_index, size=28)

    xticks = ['8k', '16k', '32k', '64k', '128k']
    sns_g.set_xticklabels(xticks, size=28)

    # for idx in [6, 13, 27, 33, 36, 40, 43, 45]:
    #     a.axhline(y=idx, color="white", linestyle="-", linewidth=4)

[fig.delaxes(a) for a in ax.flatten() if not a.has_data()]

plt.tight_layout()
plt.subplots_adjust(left=0.17, wspace=0.15)
figure_path = os.path.join(project_root, f"figures/18_internvl2_v2pe.pdf")
plt.savefig(figure_path, dpi=500, format="pdf")
plt.show()
