from utils import *
import matplotlib

# plot specific ones in a row, for formatting in the paper
lf_df = process_df(all_df)
length_datasets = ["ICL", "VRAG", "NIAH", "DocVQA"]

ncols = 4
nrows = (len(length_datasets) - 1) // ncols + 1

custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#4A895B'])
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=False)
fig.set_size_inches((ncols * 6, nrows * 5))
plt.rc('axes', unicode_minus=False)
plt.rcParams.update({'axes.unicode_minus': False})

diff_pairs = [
        ('InternVL3-38B', 'InternVL3-8B', 'Diff (8B$\\rightarrow$38B)'),
        ('Gemma3-27B', 'Gemma3-12B', 'Diff (12B$\\rightarrow$27B)')
    ]

base_index_order = [
    "GPT-4o", "Gemini-2.5-Pro",
    'InternVL3-8B', 'InternVL3-38B',
    "Gemma3-12B", "Gemma3-27B",
]

for i, dataset in enumerate(length_datasets):
    if nrows > 1:
        a = ax[i // ncols][i % ncols]
    else:
        a = ax[i]

    tdf = lf_df[lf_df.input_max_length > 4096]
    tdf = tdf.pivot_table(index="Model", columns="input_max_length", values=dataset)
    tdf = tdf.reindex(base_index_order)

    final_index_order = list(base_index_order)
    diff_data_to_add = {}

    for model_large, model_small, diff_name in diff_pairs:
        if model_large in tdf.index and model_small in tdf.index:
            diff_series = tdf.loc[model_large] - tdf.loc[model_small]
            diff_data_to_add[diff_name] = diff_series

            # insert the diff
            insert_pos = final_index_order.index(model_large) + 1
            final_index_order.insert(insert_pos, diff_name)

    tdf = tdf.reindex(final_index_order)

    for diff_name, diff_series in diff_data_to_add.items():
        tdf.loc[diff_name] = diff_series

    annot_matrix = tdf.copy()
    for diff_name in diff_data_to_add.keys():
        annot_matrix.loc[diff_name] = annot_matrix.loc[diff_name].apply(lambda x: f"{x:+.1f}")
    for idx in tdf.index:
        if idx not in diff_data_to_add:
            annot_matrix.loc[idx] = annot_matrix.loc[idx].apply(lambda x: f"{x:.1f}")

    sns_g = sns.heatmap(
        tdf, annot=annot_matrix, cmap=custom_cmap, fmt="", yticklabels=True,
        ax=a, annot_kws={"fontsize": 22},
        cbar=False
    )
    sns_g.set_title(dataset, fontsize=34)

    sns_g.set_ylabel("")
    sns_g.set_xlabel("")

    #     a.set_yticklabels(ax[0].get_yticklabels() if (i%ncols) == 0 else [], size = 16)
    #     sns_g.set_yticklabels(sns_g.get_yticklabels(), size = 16)
    #     sns_g.set_xticklabels(sns_g.get_xticklabels(), size = 24)

    sns_g.set_yticklabels(final_index_order, size=26)
    xticks_map = {"8192": '8k', "16384": '16k', "32768": '32k', "65536":'64k', "131072":'128k'}
    sns_g.set_xticklabels([xticks_map[st.get_text()] for st in sns_g.get_xticklabels()], size=28)

    # idx, start, end
    a.hlines([2, 5], 0, 6, color="0.95", linestyle="-", linewidth=3)
#     a.vlines([5, 1, 3], [0, 5, 7], [5, 7, 11], color="bisque", linestyle="--", linewidth=3)

[fig.delaxes(a) for a in ax.flatten() if not a.has_data()]

plt.tight_layout()
file_path = os.path.join(project_root, f"figures/11_results_length_select.pdf")
plt.savefig(file_path, dpi=500, format="pdf")
plt.show()