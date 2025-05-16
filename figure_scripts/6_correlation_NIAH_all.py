from utils import *

# break into different lengths and then plot the pairwise heatmap
lf_df = process_df(all_df)

cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#ed4d6e", '#DD9380', '#DEA683', '#CFCC86', "#0CD79F"])
datasets2 = ['VRAG', 'ICL', 'Summ', 'DocVQA']
datasets1 = ['VH-Single', 'VH-Multi', 'MM-NIAH-Ret (T)', 'MM-NIAH-Ret (I)', 'MM-NIAH-Ret', 'MM-NIAH-Count (T)', 'MM-NIAH-Count (I)', 'MM-NIAH-Count',
             'MM-NIAH-Reason (T)', 'MM-NIAH-Reason (I)', 'MM-NIAH-Reason', 'NIAH']

all_corr = {}
lengths = [131072]
fig, ax = plt.subplots(figsize=(2.5 + 0.7 * (len(datasets2) + 1), 0.7 * len(datasets1)), nrows=1, ncols=len(lengths))

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
        a = ax[i] if len(lengths) > 1 else ax
        hm["Avg"] = hm.mean(axis=1)
        def fmt(x):
            return "0.00" if abs(x) < 1e-2 else f"{x:.2f}"
        annots = np.vectorize(fmt)(hm)

        sns_g = sns.heatmap(hm, annot=annots, ax=a, cbar=False, cmap=cmap, annot_kws={"fontsize": 13.5}, fmt="")

        sns_g.set_ylabel("")
        sns_g.set_xlabel("")

        sns_g.set_yticklabels(datasets1, size=18)
        sns_g.set_xticklabels(sns_g.get_xticklabels(), size=18)

        a.tick_params(axis='x', rotation=45)
        a.tick_params(axis='y', rotation=0)

plt.setp(a.get_xticklabels(), ha="right", rotation_mode="anchor")
plt.tight_layout()
figure_path = os.path.join(project_root, "figures/6_correlation_NIAH_all.pdf")
plt.savefig(figure_path, dpi=300, format="pdf")
plt.show()