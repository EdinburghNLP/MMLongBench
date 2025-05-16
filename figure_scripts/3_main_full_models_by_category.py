from utils import *


# plot figures on each dataset by category

category_to_datasets = {
    "VRAG": ["InfoSeek", "ViQuAE"],
    "NIAH": ["VH-Single", "VH-Multi", 'MM-NIAH-Ret (T)', "MM-NIAH-Ret (I)",
               "MM-NIAH-Count (T)", "MM-NIAH-Count (I)", "MM-NIAH-Reason (T)", "MM-NIAH-Reason (I)"],
    "ICL": ["Stanford Cars", "Food101", "SUN397", "Inat2021"],
    "Summ": ["GovReport", "Multi-LexSum"],
    "DocVQA": ["MMLongBench-Doc", "LongDocURL", "SlideVQA"]
}

category_group = [["VRAG", "Summ"], "NIAH", "ICL", "DocVQA"] # ["VRAG", "Summ"], "Recall", "ICL", "DocQA"

# plot specific ones in a row, for formatting in the paper
lf_df = process_df(all_df)
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#4A895B'])

for cur_cate_list in category_group:
    if isinstance(cur_cate_list, str):
        cur_cate_list = [cur_cate_list]
    cur_dataset_list = [dataset_name for cate_name in cur_cate_list for dataset_name in category_to_datasets[cate_name]]
    cur_title = " and ".join(cur_cate_list)
    length_datasets = cur_dataset_list

    ncols = min(4, len(length_datasets))
    nrows = (len(length_datasets) - 1) // ncols + 1
    if ncols == 3:
        col_width = 8
    else:
        col_width = 6

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=False)
    fig.set_size_inches((ncols * col_width, 20 * nrows)) # helmet has 45 models for 20 height, we have 47 models

    plt.rc('axes', unicode_minus=False)
    plt.rcParams.update({'axes.unicode_minus': False})

    for i, dataset in enumerate(length_datasets):
        if nrows > 1:
            a = ax[i // ncols][i % ncols]
        else:
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

    if len(cur_cate_list) > 1:
        fig.tight_layout()
        col_divider = 2
        assert nrows == 1
        left_ax = ax[col_divider - 1]
        right_ax = ax[col_divider]
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()

        line_pos = (left_pos.x1 + right_pos.x0) / 2 + 0.001

        fig.add_artist(plt.Line2D([line_pos, line_pos], [0, 1],
                                  transform=fig.transFigure,
                                  color='black',
                                  linestyle='--',
                                  linewidth=5))

    [fig.delaxes(a) for a in ax.flatten() if not a.has_data()]

    plt.tight_layout()
    if ncols == 3:
        plt.subplots_adjust(left=0.17, wspace=0.30)
    cur_file_title = cur_title.replace(" ", "-")
    figure_path = os.path.join(project_root, f"figures/3_category_{cur_file_title}_length_full.pdf")
    plt.savefig(figure_path, dpi=500, format="pdf")
    plt.show()
