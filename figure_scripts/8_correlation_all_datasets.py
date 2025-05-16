from utils import *

# break into different lengths and then plot the pairwise heatmap
lf_df = process_df(all_df)
datasets = ["InfoSeek", "ViQuAE", "VRAG",
            "VH-Single", "VH-Multi", 'MM-NIAH-Ret', "MM-NIAH-Count", "MM-NIAH-Reason", 'NIAH',
            "Stanford Cars", "Food101", "SUN397", "Inat2021", 'ICL',
            "GovReport", "Multi-LexSum", 'Summ',
            "MMLongBench-Doc", "LongDocURL", "SlideVQA", 'DocVQA',
            'Ours']

datasets1, datasets2 = datasets, datasets

cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#ed4d6e", '#DD9380', '#DEA683', '#CFCC86', "#0CD79F"])

all_corr = {}
lengths = [131072]
fig, ax = plt.subplots(figsize=(len(datasets), len(datasets)), nrows=1, ncols=len(lengths))

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
        def fmt(x):
            return "0.00" if abs(x) < 1e-2 else f"{x:.2f}"
        annots = np.vectorize(fmt)(hm)
        sns_g = sns.heatmap(hm, annot=annots, ax=a, cbar=False, cmap=cmap, annot_kws={"fontsize": 19}, fmt="")

        sns_g.set_ylabel("")
        sns_g.set_xlabel("")

        t = datasets
        sns_g.set_yticklabels(t, size=24)
        sns_g.set_xticklabels(t, size=24)

        a.tick_params(axis='x', rotation=45)
        a.tick_params(axis='y', rotation=0)

        for idx in [3, 9, 14, 17, 21]:
            a.axvline(x=idx, color="white", linestyle="-", linewidth=3)
            a.axhline(y=idx, color="white", linestyle="-", linewidth=3)

plt.setp(a.get_xticklabels(), ha="right", rotation_mode="anchor")
plt.tight_layout()
figure_path = os.path.join(project_root, "figures/8_correlation_all_dataset.pdf")
plt.savefig(figure_path, dpi=300, format="pdf")
plt.show()