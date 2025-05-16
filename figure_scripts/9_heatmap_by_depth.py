from utils import *

# plot the heatmap by depth for each model, note that this takes quite some time to generate...
# use your dataset name to filter datasets_configs
# the tasks that need depth: viquae, vh_single, mm_niah_retrieval-text, mm_niah_retrieval-image, mm_niah_reasoning-image
dataset_name_list = ["viquae", "vh_single", "mm_niah_retrieval-text", "mm_niah_retrieval-image", "mm_niah_reasoning-image"] #

for dataset_name in dataset_name_list:
    print(f"Processing {dataset_name} now")
    datasets_configs = [config for config in dataset_configs if config["dataset"] == dataset_name]

    depth_dfs = []
    ncols = 6

    for i, model in enumerate(models_configs):
        args = arguments()
        depths = []
        for dataset in datasets_configs:
            args.update(dataset)
            args.update(model)

            depth = args.get_metric_by_depth()
            if depth is None:
                continue
            for d in depth:
                d["input_length"] = args.input_max_length
                d["depth"] = math.ceil(d["depth"] * 10) / 10
            depths += depth
            print('good')
        if len(depths) == 0:
            continue
        depths = pd.DataFrame(depths)
        depth_dfs.append((model, depths))

    fig, ax = plt.subplots(nrows=(len(depth_dfs) - 1) // ncols + 1, ncols=ncols, sharey=False, sharex=False)
    fig.set_size_inches((ncols * 5, len(ax) * 4.25))
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    vmin = min([d[1]['metric'].min() for d in depth_dfs])
    vmax = max([d[1]['metric'].max() for d in depth_dfs])
    for i, (config, depths) in enumerate(tqdm(depth_dfs, "drawing models")):
        a = ax[i // ncols][i % ncols]
        pivot_table = depths.pivot(index="depth", columns="input_length", values="metric")

        need_annot_matrix = []
        for col in [8192, 16384, 32768, 65536, 131072]:
            if col not in pivot_table.columns:
                pivot_table[col] = vmin # use vmin as default
                need_annot_matrix.append(col)

        if need_annot_matrix:
            annot_matrix = pivot_table.copy()
            annot_matrix = annot_matrix.applymap(lambda x: f"{x:.2f}")
            for col in need_annot_matrix:
                annot_matrix[col] = "N/A"
            sns_g = sns.heatmap(
                pivot_table, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"label": "Score"}, ax=a, annot=annot_matrix,
                cbar=False, annot_kws={"fontsize": 16}, fmt="")
        else:
            sns_g = sns.heatmap(
                pivot_table, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"label": "Score"}, ax=a, annot=True,
                cbar=False, annot_kws={"fontsize": 16}
            )
        m = config['model']

        idx = {'8k': 1, '10k': 1, '16k': 2, '32k': 3, '64k': 4, '80k': 4, '128k': 5, '200k': 5, '1m': 5}[model_lengths[model_name_replace[config['model']]]]
        a.axvline(idx, color="white", linestyle="--", linewidth=4)

        num_columns = pivot_table.shape[1]
        a.set_xlim(0, num_columns + 0.5)

        sns_g.set_title(model_name_replace.get(m, m), fontsize=28)

        xticks = {'4096': '4k', '8192': '8k', '16384': '16k', '32768': '32k', '65536': '64k', '131072': '128k'}
        xticks = [xticks[x.get_text()] for x in a.get_xticklabels()]
        a.set_xticklabels(xticks, size=24)

        ytick_labels = pivot_table.index.astype(str).tolist()
        a.set_yticklabels(ytick_labels, size=24, rotation=0)

        if i % ncols == 0:
            a.set_ylabel("Depth", size=28)
        else:
            a.set_ylabel("")
        a.set_xlabel("")

    [fig.delaxes(a) for i, a in enumerate(ax.flatten()) if not a.has_data()]
    plt.tight_layout()
    figure_path = os.path.join(project_root, f"figures/9_depths_{dataset_name}.pdf")
    plt.savefig(figure_path, dpi=300, format="pdf")
    plt.show()