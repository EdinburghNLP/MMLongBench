from utils import *

main_table_datasets = [
    "VRAG",
    "NIAH",
    'ICL',
    "Summ",
    "DocVQA",
    "Ours"
]

lf_df = process_df(all_df)
length_datasets = main_table_datasets

ncols = 3
nrows = (len(length_datasets) - 1) // ncols + 1

custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#4A895B'])
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=False)
fig.set_size_inches((ncols * 8, 26)) # helmet has 22 models for 26 height, we have 23 models

plt.rc('axes', unicode_minus=False)
plt.rcParams.update({'axes.unicode_minus': False})

for i, dataset in enumerate(length_datasets):
    if nrows > 1:
        a = ax[i // ncols][i % ncols]
    else:
        a = ax[i]

    new_index = main_table_models

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
    sns_g.set_yticklabels(new_index, size=28)

    xticks = ['8k', '16k', '32k', '64k', '128k']
    sns_g.set_xticklabels(xticks, size=28)

    for idx in [6, 10, 14, 17, 19]:
        a.axhline(y=idx, color="white", linestyle="-", linewidth=4)

[fig.delaxes(a) for a in ax.flatten() if not a.has_data()]


plt.tight_layout()
plt.subplots_adjust(left=0.17, wspace=0.15)
plt.savefig(os.path.join(project_root, f"figures/1_results_length_main.pdf"), dpi=500, format="pdf")
plt.show()