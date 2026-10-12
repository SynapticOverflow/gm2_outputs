import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

C_ENZYME = "#D55E00"
C_DELIV = "#0072B2"

DATA_DIR = "/home/claude/task_results/gm2_lsd_tasks1-6_results"
OUT = "/home/claude/plos_paper/figures"

# Panel A: full 9-parameter GM2 three-way (real Task 1 data)
scenarios = {}
for s in ['as_documented', 'equal_width', 'tier1_promotion']:
    with open(f'{DATA_DIR}/gm2_full14_{s}.json') as f:
        scenarios[s] = json.load(f)

names_raw = scenarios['as_documented']['names']
label_map = {
    "gm2_synth": "Synthesis rate", "k_T4_decay": "Catalytic decay",
    "vmax_brain": "Catalytic capacity", "km_brain": "Michaelis constant",
    "k_T4_payload": "Expression rate", "t4_dose_total": "Total AAV dose",
    "k_T4_entry": "BBB-entry rate", "IC50": "IC50",
    "fus_entry_gain_scale": "FUS permeability gain",
}
delivery_params = {"t4_dose_total", "k_T4_entry", "IC50", "fus_entry_gain_scale"}
keep = [n for n in names_raw if n in label_map]

as_doc = [scenarios['as_documented']['ST'][names_raw.index(n)] for n in keep]
eq_w = [scenarios['equal_width']['ST'][names_raw.index(n)] for n in keep]
t1 = [scenarios['tier1_promotion']['ST'][names_raw.index(n)] for n in keep]
is_delivery = [n in delivery_params for n in keep]
labels = [label_map[n] for n in keep]

order = np.argsort(as_doc)
labels_s = [labels[i] for i in order]
as_doc_s = [as_doc[i] for i in order]
eq_w_s = [eq_w[i] for i in order]
t1_s = [t1[i] for i in order]
is_delivery_s = [is_delivery[i] for i in order]

# Panel B: cross-disease, as-documented (sorted), real data
diseases = ["Tay-Sachs", "Sandhoff", "Pompe", "Krabbe", "GM1", "MPS I",
            "MLD", "CLN2", "Niemann-Pick C", "Fabry"]
dominant_val = [0.241, 0.241, 0.417, 0.225, 0.231, 0.597, 0.597, 0.598, 0.959, 0.991]
entry_st = [0.047, 0.040, 0.170, 0.038, 0.063, 0.009, 0.023, 0.009, 0.000, 0.000]

d_order = np.argsort(dominant_val)
diseases_s = [diseases[i] for i in d_order]
dominant_val_s = [dominant_val[i] for i in d_order]
entry_st_s = [entry_st[i] for i in d_order]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))

ax = axes[0]
y = np.arange(len(labels_s))
offset = 0.22
for row_offset, vals, marker, alpha in [
    (offset, as_doc_s, "o", 1.0), (0, eq_w_s, "s", 0.65), (-offset, t1_s, "^", 0.4),
]:
    colors = [C_DELIV if d else C_ENZYME for d in is_delivery_s]
    ax.scatter(vals, y + row_offset, marker=marker, s=42, c=colors, alpha=alpha,
               edgecolor="black", linewidth=0.5, zorder=3)
for yi in y:
    ax.axhline(yi, color="gray", lw=0.4, alpha=0.3, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(labels_s, fontsize=8)
ax.set_xlabel("Total-order Sobol index $S_T$", fontsize=8.5)
ax.set_title("A", loc="left", fontsize=12, fontweight="bold")
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=7, label="As-documented"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", alpha=0.65, markeredgecolor="black", markersize=7, label="Equal-width"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", alpha=0.4, markeredgecolor="black", markersize=7, label="Tier-1-promoted"),
]
ax.legend(handles=legend_elems, fontsize=7, loc="lower right")
ax.tick_params(labelsize=8)

ax = axes[1]
y = np.arange(len(diseases_s))
ax.scatter(dominant_val_s, y + 0.15, marker="o", color=C_ENZYME, s=50, label="Dominant parameter", zorder=3)
ax.scatter(entry_st_s, y - 0.15, marker="s", color=C_DELIV, s=50, label="BBB-entry rate", zorder=3)
for yi in y:
    ax.axhline(yi, color="gray", lw=0.4, alpha=0.3, zorder=0)
ax.axvline(0, color="black", lw=0.6)
ax.axvline(1.0, color="black", lw=0.6, ls=":")
ax.set_yticks(y)
ax.set_yticklabels(diseases_s, fontsize=8)
ax.set_xlabel("Total-order Sobol index $S_T$ (as-documented)", fontsize=8.5)
ax.set_title("B", loc="left", fontsize=12, fontweight="bold")
ax.legend(fontsize=7.5, loc="lower right")
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_robustness_combined.png", dpi=300)
plt.savefig(f"{OUT}/fig2_robustness_combined.tiff", dpi=300)
plt.close()
print("Fig 2 rebuilt with real 9-parameter GM2 data and cross-disease data.")
