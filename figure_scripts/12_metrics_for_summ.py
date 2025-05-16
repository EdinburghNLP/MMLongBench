from utils import *

# plot specific ones in a row, for formatting in the paper
lf_df = process_df(all_df)
dataset_names = ["GovReport", "Multi-LexSum"]
length_datasets = {"gov-report rougeLsum_f1": "ROUGE-L",
                   "GovReport": "GPT-4o Eval",
                   "multi-lexsum rougeLsum_f1": "ROUGE-L",
                   "Multi-LexSum": "GPT-4o Eval"}

summ_dataset_rouge = {"gov-report": "rougeLsum_f1",
    "multi-lexsum": "rougeLsum_f1"}

dataset_to_metrics["gov-report"] = ["rougeLsum_f1"]
dataset_to_metrics["multi-lexsum"] = ["rougeLsum_f1"]

rouge_dfs = []

for m_idx, model in enumerate(models_configs):
    args = arguments()
    for dataset in dataset_configs:
        if dataset["dataset"] not in summ_dataset_rouge:
            continue
        args.update(dataset)
        args.update(model)

        # parse the metrics
        metric = args.get_averaged_metric()
        dsimple, mnames = args.get_metric_name()

        if metric is None:
            print("failed:", args.get_path()) # will be np.nan when using DataFrame
            continue
        for k, m in metric.items():
            rouge_dfs.append({**asdict(args), **model,
                        "metric name": k, "metric": m,
                        "dataset_simple": dsimple + " " + k,
                        "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}"
                        })

rouge_df = pd.DataFrame(rouge_dfs)
import utils
utils.custom_avgs = {}
rouge_df = process_df(rouge_df)

lf_df = pd.merge(
    lf_df,
    rouge_df,
    on=["input_max_length", "Model"],
    how="outer",
    suffixes=('', '_rouge')
)
# duplicated_cols = lf_df.columns.duplicated()
# lf_df = lf_df.loc[:, ~duplicated_cols]

new_index = [
    'GPT-4o',
    'InternVL2.5-4B', 'InternVL2.5-8B',
    'InternVL3-2B', 'InternVL3-14B',
    'Gemma3-4B', 'Gemma3-12B', 'Gemma3-27B',
]

ncols = 4
nrows = (len(length_datasets) - 1) // ncols + 1

fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=False)
fig.set_size_inches((ncols * 6, nrows * 6))
plt.subplots_adjust(left=0.15, top=0.83, right=0.99, wspace=0.07)

title_y_pos = 0.95
left_pos = (ax[0].get_position().x0 + ax[1].get_position().x1) / 2
right_pos = (ax[2].get_position().x0 + ax[3].get_position().x1) / 2
fig.text(left_pos, title_y_pos, 'GovReport', fontsize=30, ha='center', va='center', fontweight='bold')
fig.text(right_pos, title_y_pos, 'Multi-LexSum', fontsize=30, ha='center', va='center', fontweight='bold')

line_y = title_y_pos - 0.03
axes_row = ax if nrows == 1 else ax[0, :]
transform = fig.transFigure
fig.add_artist(plt.Line2D([axes_row[0].get_position().x0, axes_row[1].get_position().x1],
                              [line_y, line_y], color='black', linewidth=1.0, transform=transform))
fig.add_artist(plt.Line2D([axes_row[2].get_position().x0, axes_row[3].get_position().x1],
                              [line_y, line_y], color='black', linewidth=1.0, transform=transform))




cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#ed4d6e", '#DD9380', '#DEA683', '#CFCC86', "#0CD79F"])

group_min_list = [float('inf')] * len(length_datasets)
group_max_list = [float('-inf')] * len(length_datasets)

for d_idx, dataset_col in enumerate(length_datasets.keys()):
    tdf = lf_df[lf_df.input_max_length > 4096]
    tdf = tdf.pivot_table(index="Model", columns="input_max_length", values=dataset_col)
    tdf = tdf.reindex(new_index)

    current_min = tdf.min().min()
    current_max = tdf.max().max()

    group_min_list[d_idx] = min(group_min_list[d_idx], current_min)
    group_max_list[d_idx] = max(group_max_list[d_idx], current_max)

group_size = 2
for g_idx in range(0, len(length_datasets), group_size):
    group_min = min(group_min_list[g_idx: g_idx + group_size])
    group_max = max(group_max_list[g_idx: g_idx + group_size])
    for _ in range(g_idx, g_idx + group_size):
        group_min_list[_] = group_min
        group_max_list[_] = group_max

for i, (dataset_col, dataset_title) in enumerate(length_datasets.items()):
    if nrows > 1:
        a = ax[i // ncols][i % ncols]
    else:
        a = ax[i]

    tdf = lf_df[lf_df.input_max_length > 4096]
    tdf = tdf.pivot_table(index="Model", columns="input_max_length", values=dataset_col)
    tdf = tdf.reindex(new_index)

    import matplotlib.colors as mcolors
    sns_g = sns.heatmap(
        tdf, annot=True, cmap=cmap, fmt=".1f", yticklabels=True,
        ax=a, annot_kws={"fontsize": 21}, vmax=group_max_list[i], vmin=group_min_list[i],
        cbar=False, norm=mcolors.PowerNorm(gamma=0.5, vmin=group_min_list[i], vmax=group_max_list[i])
    )
    sns_g.set_title(dataset_title, fontsize=30)

    sns_g.set_ylabel("")
    sns_g.set_xlabel("")

    written_index = [x.replace("-Inst", '') for x in tdf.index.tolist()]

    sns_g.set_yticklabels(written_index, size=29)
    xticks = ['8k', '16k', '32k', '64k', '128k']
    sns_g.set_xticklabels(xticks, size=29)
    for idx in [1, 3, 5]:
        a.axhline(y=idx, color="white", linestyle="-", linewidth=4)

[fig.delaxes(a) for a in ax.flatten() if not a.has_data()]

# plt.tight_layout()
figure_path = os.path.join(project_root, f"figures/12_results_length_model_eval.pdf")
plt.savefig(figure_path, dpi=500, format="pdf")
plt.show()