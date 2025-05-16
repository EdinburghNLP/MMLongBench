import matplotlib.pyplot as plt
import numpy as np


def create_adaptive_pie(ax, data, colors, threshold=0.05, title=None, font_size=12):
    large_data = {}
    small_data = {}

    for key, value in data.items():
        if value >= threshold:
            large_data[key] = value
        else:
            small_data[key] = value

    all_keys = list(data.keys())
    all_values = list(data.values())
    all_colors = [colors[i % len(colors)] for i in range(len(data))]

    wedges, _ = ax.pie(
        all_values,
        labels=None,
        colors=all_colors,
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )

    for i, key in enumerate(all_keys):
        if all_values[i] >= threshold:
            angle = wedges[i].theta1 + (wedges[i].theta2 - wedges[i].theta1) / 2
            angle_rad = np.deg2rad(angle)

            r = 0.6
            x = r * np.cos(angle_rad)
            y = r * np.sin(angle_rad)

            percentage = f"{all_values[i]:.1%}"
            ax.text(x, y, percentage, ha='center', va='center',
                    fontweight='bold', fontsize=font_size)

    for i, key in enumerate(all_keys):
        angle = wedges[i].theta1 + (wedges[i].theta2 - wedges[i].theta1) / 2
        angle_rad = np.deg2rad(angle)

        if all_values[i] < threshold:

            label_text = f"{key}\n({all_values[i]:.1%})"
        else:
            label_text = f"{key}"

        r = 1.1
        x = r * np.cos(angle_rad)
        y = r * np.sin(angle_rad)

        if key == "Pure-text" and "LongDocURL" in title:
            x += 0.2
            y -= 0.4

        ha = "center"
        if angle > 90 and angle < 270:
            ha = "right"
        elif angle < 90 or angle > 270:
            ha = "left"

        ax.text(x, y, label_text, ha=ha, va='center', fontweight='bold', fontsize=font_size)

    if title:
        ax.set_title(title, fontsize=font_size + 2)

    ax.set_aspect('equal')


mmlongdoc_sources = {
    'Pure-text': 0.256,
    'Layout': 0.106,
    'Table': 0.205,
    'Figure': 0.285,
    'Chart': 0.161,
}

mmlongdoc_formats = {
    'String': 0.172,
    'Integer': 0.341,
    'Float': 0.135,
    'List': 0.120,
    'None': 0.232,
}

longdocurl_sources = {
    'Pure-Text': 0.450,
    'Layout': 0.272,
    'Table': 0.372,
    'Figure': 0.208,
    'Others': 0.002
}

longdocurl_formats = {
    'String': 0.261,
    'Integer': 0.341,
    'Float': 0.152,
    'List': 0.239,
    'None': 0.008
}

colors1 = plt.cm.Pastel1(np.linspace(0, 1, 5))
colors2 = plt.cm.Pastel2(np.linspace(0, 1, 5))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
pie_font = 12

create_adaptive_pie(axes[0], mmlongdoc_sources, colors1, threshold=0.10,
                   title='MMLB-Doc Answer Sources', font_size=14)

create_adaptive_pie(axes[1], mmlongdoc_formats, colors2, threshold=0.10,
                   title='MMLB-Doc Answer Format', font_size=14)

create_adaptive_pie(axes[2], longdocurl_sources, colors1, threshold=0.10,
                   title='LongDocURL Answer Sources', font_size=14)

create_adaptive_pie(axes[3], longdocurl_formats, colors2, threshold=0.10,
                   title='LongDocURL Answer Format', font_size=14)
for ax in axes:
    ax.set_aspect('equal')

plt.tight_layout()
project_root = "/home/zhaowei.wang/vl-longbench"
import os
figure_path = os.path.join(project_root, 'figures/16_docqa_dist.pdf')
plt.savefig(figure_path, format='pdf', dpi=300, bbox_inches='tight')
plt.show()