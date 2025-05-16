from utils import *
import matplotlib.patches as patches

lf_df = process_df(all_df)
dname = "DocVQA"

more_avgs = {
    "Others": ["DocVQA"], # 'VRAG', 'ICL', 'Summ', "DocQA"
}

for k, v in more_avgs.items():
    lf_df[k] = lf_df[v].mean(axis=1)

melted_df = lf_df.melt(id_vars=['input_max_length', "Model", dname])

# plot correlation across datasets and lengths with one specific dataset
tdf = melted_df[melted_df.input_max_length==131072]
# tdf = tdf[tdf.dataset_simple.isin(['NIAH S Essay', 'NIAH MK Needle', 'HotpotQA'])]
g = sns.relplot(
    tdf.rename({"input_max_length": "length", "dataset_simple": "dataset"}, axis=1),
    x="value", y=dname, col="dataset", row="length",
    facet_kws={'sharey': False, 'sharex': False},
    hue="Model", markers=True, legend=False, col_order=['MM-NIAH-Ret', 'MM-NIAH-Count', 'MM-NIAH-Reason'],
    s=100,
)


def annotate(data, **kws):
    data = data[data["value"].notna() & data[dname].notna()]
    ax = plt.gca()
    if len(data) > 1:
        ax.text(.05, .95, 'n={}'.format(len(data)), transform=ax.transAxes, fontsize=15)
        rho, p = stats.spearmanr(data['value'], data[dname])
        ax.text(.05, .88, 'Spearman $\\rho$={:.2f}'.format(rho), transform=ax.transAxes, fontsize=15)
        # ax.text(.05, .81, 'p={:.2g}'.format(p), transform=ax.transAxes, fontsize=15)
        title_info = ax.get_title().split("dataset =")[1].strip()
        if title_info == 'MM-NIAH-Reason':
            x_start, x_end = 0, 40
            y_start, y_end = 0, 60
            x_offset = 18
            y_offset = 2
        elif title_info == 'MM-NIAH-Count':
            x_start, x_end = 0, 30
            y_start, y_end = 0, 60
            x_offset = 18
            y_offset = -1
        elif title_info == 'MM-NIAH-Ret':
            x_start, x_end = 0, 50
            y_start, y_end = 0, 60
            x_offset = 20
            y_offset = -0.5

        width, height = x_end - x_start, y_end - y_start

        rect = patches.Rectangle(
            (x_start, y_start), width, height,
            linewidth=0, edgecolor='none',  facecolor='lightcoral',
            alpha=0.2, zorder=0  # Put it behind data points
        )
        ax.add_patch(rect)

        # 2. Find points within the X range [0, 40]
        points_in_range = data[(data['value'] >= 0) & (data['value'] <= x_end)].copy()
        max_point = points_in_range.loc[points_in_range[dname].idxmax()]
        x_coord = max_point['value']
        y_coord = max_point[dname]

        ax.axhline(y=y_coord, xmin=0, xmax=x_coord / ax.get_xlim()[1],
                   linestyle='--', color='dimgrey', alpha=0.6, linewidth=2)

        ax.annotate(text=f"{y_coord:.1f}",  xy=(x_coord, y_coord),  xytext=(x_coord + x_offset, y_coord + y_offset),
            fontsize=14,  color='dimgrey',  fontweight='normal',
            arrowprops=dict(
                arrowstyle="->",  # Arrow style
                color='dimgrey',  # Arrow color
                lw=1.5,  # Slightly thicker arrow
                connectionstyle = "arc3,rad=0.2"
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc='bisque', ec="black", lw=0.5, alpha=0.6)
        )

    print(ax.get_title())
    dataset = ax.get_title().split("= ")[-1]
    ax.set_title("")
    ax.set_xlabel(dataset, fontsize=22)
    ax.set_ylabel(dname, fontsize=22)
    ax.set_yticklabels(ax.get_yticklabels(), size = 20)
    ax.set_xticklabels(ax.get_xticklabels(), size = 20)

g.map_dataframe(annotate)
# g.fig.suptitle(f"Correlation with {dname}", fontsize=24, y=1.05)
plt.tight_layout()
fname = "figures/10_correlation_recall_others_dist.pdf"
fname = os.path.join(project_root, fname)
print(fname)
plt.savefig(fname, dpi=300, format="pdf")
plt.show()