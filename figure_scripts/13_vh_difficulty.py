from utils import *
import matplotlib

# plot specific ones in a row, for formatting in the paper
lf_df = process_df(all_df)
length_datasets = ["VH-Single", "VH-Multi"]

ncols = 2
nrows = (len(length_datasets) - 1) // ncols + 1

custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#4A895B'])
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=False)
fig.set_size_inches((ncols * 6 + 2, nrows * 5))
plt.rc('axes', unicode_minus=False)
plt.rcParams.update({'axes.unicode_minus': False})

base_index_order = [
    "GPT-4o", "Gemini-2.5-Pro",
    'Qwen2.5-VL-7B-Inst', 'Qwen2.5-VL-32B-Inst', 'Qwen2.5-VL-72B-Inst',
    "Gemma3-4B", "Gemma3-12B", "Gemma3-27B",
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

    sns_g = sns.heatmap(
        tdf, annot=True, cmap=custom_cmap, fmt=".1f", yticklabels=True,
        ax=a, annot_kws={"fontsize": 22},
        cbar=False
    )
    sns_g.set_title(dataset, fontsize=26)

    sns_g.set_ylabel("")
    sns_g.set_xlabel("")

    written_index = [x.replace("-Inst", '') for x in final_index_order]
    sns_g.set_yticklabels(written_index, size=18, fontweight='bold')
    xticks_map = {"8192": '8k', "16384": '16k', "32768": '32k', "65536":'64k', "131072":'128k'}
    sns_g.set_xticklabels([xticks_map[st.get_text()] for st in sns_g.get_xticklabels()], size=22)

    # idx, start, end
    a.hlines([2, 5], 0, 6, color="0.95", linestyle="-", linewidth=3)

[fig.delaxes(a) for a in ax.flatten() if not a.has_data()]

plt.tight_layout()
plt.subplots_adjust(left=0.18, wspace=0.28)
file_path = os.path.join(project_root, f"figures/13_vh_difficulty.pdf")
plt.savefig(file_path, dpi=500, format="pdf")
plt.show()