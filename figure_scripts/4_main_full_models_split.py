from utils import *


# Figure: Full models


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

tasks_per_row = 3
total_tasks = len(length_datasets)
rows_needed = (total_tasks + tasks_per_row - 1) // tasks_per_row
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#4A895B'])

for row in range(rows_needed):
    start_idx = row * tasks_per_row
    end_idx = min((row + 1) * tasks_per_row, total_tasks)
    current_datasets = length_datasets[start_idx:end_idx]

    fig, ax = plt.subplots(ncols=tasks_per_row, nrows=1, sharey=True, sharex=False)
    fig.set_size_inches((tasks_per_row * 7.5, 20.5)) # helmet has 45 models for height 20, we have 47 models

    plt.rc('axes', unicode_minus=False)
    plt.rcParams.update({'axes.unicode_minus': False})

    for i, dataset in enumerate(current_datasets):
        a = ax[i]

        new_index = full_table_models

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

        for idx in [6, 13, 27, 33, 36, 40, 43, 45]:
            a.axhline(y=idx, color="white", linestyle="-", linewidth=4)

    [fig.delaxes(a) for a in np.atleast_1d(ax).flatten() if not a.has_data()]

    plt.tight_layout()

    figure_path = os.path.join(project_root, f"figures/4_results_length_full_row{row+1}.pdf")
    plt.savefig(figure_path, dpi=500, format="pdf")
    plt.close(fig)


