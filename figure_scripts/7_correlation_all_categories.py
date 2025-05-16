from utils import *

# break into different lengths and then plot the pairwise heatmap
datasets = ['VRAG', 'NIAH', 'ICL', 'Summ', 'DocVQA']
lf_df = process_df(all_df)

datasets1, datasets2 = datasets, datasets

cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#ed4d6e", '#DD9380', '#DEA683', '#CFCC86', "#0CD79F"])

all_corr = {}
lengths = [131072]
total_columns = len(datasets) + 1
fig, ax = plt.subplots(figsize=(total_columns, len(datasets) + 0.1), nrows=1, ncols=len(lengths))

for i, (l, d) in enumerate(lf_df.groupby("input_max_length", sort=False)):
    if l not in lengths:
        continue
    i = lengths.index(l)
    spearmans = []
    for d1 in datasets1:
        for d2 in datasets2:
            x = d[d[d1].notnull() & d[d2].notnull()]
            m1 = x[d1]
            m2 = x[d2]

            if len(m1) < 2 and len(m2) < 2:
                continue

            rho, p = stats.spearmanr(m1, m2)
            spearmans.append({"dataset 1": d1, "dataset 2": d2, "correlation": rho})

    all_corr[l] = {"spearman": pd.DataFrame(spearmans)}
    for j, (name, table) in enumerate(all_corr[l].items()):
        hm = table.pivot_table(index="dataset 1", columns="dataset 2", values="correlation", sort=False)

        avg_corr = {}
        std_corr = {}
        for dataset in hm.index:
            corrs = [hm.loc[dataset, col] for col in hm.columns if col != dataset]
            avg_corr[dataset] = np.mean(corrs)
            std_corr[dataset] = np.std(corrs)
        hm['Avg'] = pd.Series(avg_corr)

        annot_matrix = hm.copy()
        for dataset in avg_corr:
            avg = avg_corr[dataset]
            std = std_corr[dataset]
            avg_str = f"{avg:.3f}"[1:]
            std_str = f"{std:.2f}"
            annot_matrix.loc[dataset, "Avg"] = f"${avg_str}_{{{std_str}}}$"

        annot_matrix[datasets] = annot_matrix[datasets].applymap(lambda x: f"{x:.2f}")

        # compress the outlier values for better visualization.
        # otherwise we cannot tell the difference
        max_value = hm[abs(hm - 1) > 1e-6].max().max()
        hm[abs(hm - 1.0) < 1e-6] = min(max_value + 0.03, 1)

        tmp_min_value = hm.min().min()
        min_value = hm[abs(hm - tmp_min_value) > 1e-6].min().min()
        hm[abs(hm - tmp_min_value) < 1e-6] = min(min_value - 0.03, 1)


        a = ax[i] if len(lengths) > 1 else ax
        import matplotlib.colors as mcolors
        sns_g = sns.heatmap(hm, annot=False, ax=a, cbar=False, cmap=cmap, norm=mcolors.PowerNorm(gamma=0.5, vmin=hm.min().min(), vmax=hm.max().max()))
        for i in range(len(annot_matrix.index)):
            for j in range(len(annot_matrix.columns)):
                text_value = annot_matrix.iloc[i, j]
                if annot_matrix.columns[j] == "Avg":
                    a.text(j + 0.5, i + 0.5, text_value,
                           ha="center", va="center", color="black", fontsize=13)
                else:
                    a.text(j + 0.5, i + 0.5, text_value,
                           ha="center", va="center", color="black", fontsize=13.5)

        sns_g.set_ylabel("")
        sns_g.set_xlabel("")

        t = datasets
        sns_g.set_yticklabels(t, size=16)
        sns_g.set_xticklabels(t + ["Avg$_{std}$"], size=16)

        a.tick_params(axis='x', rotation=45)
        a.tick_params(axis='y', rotation=0)

        # a.axvline(x=5, color="white", linestyle="-", linewidth=1.5)

plt.setp(a.get_xticklabels(), ha="right", rotation_mode="anchor")
plt.tight_layout()
figure_path = os.path.join(project_root, "figures/7_correlation_all_category.pdf")
plt.savefig(figure_path, dpi=300, format="pdf")
plt.show()