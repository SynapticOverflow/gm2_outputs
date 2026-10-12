import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

C_ENZYME = "#D55E00"
C_DELIV = "#0072B2"
C_NEUTRAL = "#555555"
C_TRI = "#009E73"

DATA_DIR = "/home/claude/task_results/gm2_lsd_tasks1-6_results"
DATA_DIR2 = "/home/claude/task_results_round2/gm2_task_outputs"
OUT = "/home/claude/plos_paper/figures"

# ---------------------------------------------------------------------------
# Fig 0: "claim -> audit -> reversal" schematic. Purely conceptual/textual,
# built from boxes and arrows -- no data, no fabricated numbers.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, w, h, text, color, fontsize=8):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
             wrap=True)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"))

# Panel A: original claim
box(0.3, 4.3, 2.3, 1.3, "Original claim:\nBBB-entry kinetics\ndominate\n($S_T=0.909$)", "#f4a582", fontsize=7.5)
ax.text(1.45, 5.8, "A", fontsize=13, fontweight="bold")

# Panel B: three audit checks
box(3.1, 5.0, 1.9, 0.7, "Convergence\ncheck", "#d1e5f0", fontsize=7)
box(3.1, 4.1, 1.9, 0.7, "Prior-width\nrobustness", "#d1e5f0", fontsize=7)
box(3.1, 3.2, 1.9, 0.7, "Saturation /\nfactorial necessity", "#d1e5f0", fontsize=7)
ax.text(4.05, 5.9, "B", fontsize=13, fontweight="bold")
arrow(2.6, 4.95, 3.1, 5.35)
arrow(2.6, 4.95, 3.1, 4.45)
arrow(2.6, 4.95, 3.1, 3.55)

# Panel C: corrected interpretation
box(5.6, 4.3, 2.3, 1.3, "Corrected finding:\nenzyme/synthesis\nparameters dominate;\nFUS practically negligible", "#92c5de", fontsize=7.5)
ax.text(6.75, 5.8, "C", fontsize=13, fontweight="bold")
arrow(5.0, 5.35, 5.6, 4.95)
arrow(5.0, 4.45, 5.6, 4.65)
arrow(5.0, 3.55, 5.6, 4.35)

# Panel D: general lesson
box(8.2, 4.3, 1.6, 1.3, "General lesson:\nsensitivity rankings\nreflect model + input\ndistribution, not\nbiology alone", "#fddbc7", fontsize=7)
ax.text(9.0, 5.8, "D", fontsize=13, fontweight="bold")
arrow(7.9, 4.95, 8.2, 4.95)

# External validation note below
box(2.5, 0.8, 5.0, 1.3,
    "External validation: model output disagrees with\npublished natural-history and treatment-response data\nfor all three monotherapy arms tested --- a separate,\nabsolute-accuracy limitation orthogonal to (A)--(D)",
    "#e0e0e0", fontsize=7.5)
arrow(5.0, 3.5, 5.0, 2.1)

plt.tight_layout()
plt.savefig(f"{OUT}/fig0_claim_audit_reversal.png", dpi=300)
plt.savefig(f"{OUT}/fig0_claim_audit_reversal.tiff", dpi=300)
plt.close()
print("Fig 0 (schematic) built.")

# ---------------------------------------------------------------------------
# Elasticity E(k) = d(log Y)/d(log k) computed from EXISTING saturation
# sweep data (no new simulation) -- numerical finite-difference derivative.
# ---------------------------------------------------------------------------
with open(f"{DATA_DIR}/saturation_sweep_gm2_model_v2.json") as f:
    sat = json.load(f)

k_vals = np.array(sat["k_values"])
mean_burden = np.array(sat["mean_burden"])

log_k = np.log(k_vals)
log_Y = np.log(mean_burden)
# central finite difference for elasticity
elasticity = np.gradient(log_Y, log_k)

fig, ax = plt.subplots(figsize=(4.5, 3.4))
ax.plot(k_vals, elasticity, "o-", color=C_DELIV, markersize=4, linewidth=1.0)
ax.axhline(0, color="black", lw=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"$k_{T4,\mathrm{entry}}$ (normalized units, log scale)", fontsize=9)
ax.set_ylabel(r"Local elasticity $E(k) = \partial \log Y / \partial \log k$", fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_elasticity.png", dpi=300)
plt.savefig(f"{OUT}/fig_elasticity.tiff", dpi=300)
plt.close()
print(f"Elasticity range: [{elasticity.min():.5f}, {elasticity.max():.5f}]")
print(f"Max abs elasticity: {np.abs(elasticity).max():.5f}")

# ---------------------------------------------------------------------------
# Rebuild Fig 6 (factorial) with paired-contrast forest-plot panel added
# ---------------------------------------------------------------------------
factorial_file = "/mnt/user-data/uploads/factorial_arms_shapley.json"
with open(factorial_file) as f:
    d365 = json.load(f)

arm_order = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
arm_labels = ["Untreated", "FUS", "AAV", "AAV+\nFUS", "SRT", "SRT+\nFUS", "SRT+\nAAV", "Tri-\nmodal"]
arm_colors = [C_NEUTRAL, C_NEUTRAL, C_DELIV, C_DELIV, "#E69F00", "#E69F00", C_TRI, C_TRI]
arm_lookup = {(a["SRT"], a["AAV"], a["FUS"]): a["raw_burden"] for a in d365["arms"]}

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

ax = axes[0]
rng = np.random.default_rng(0)
for i, (key, label, color) in enumerate(zip(arm_order, arm_labels, arm_colors)):
    y = np.array(arm_lookup[key])
    x_jitter = i + rng.normal(0, 0.06, size=len(y))
    ax.scatter(x_jitter, y, s=6, color=color, alpha=0.45, linewidths=0)
    ax.errorbar(i, y.mean(), yerr=y.std(), fmt="_", color="black", markersize=16,
                markeredgewidth=2, capsize=4, capthick=1.1, elinewidth=1.1, zorder=5)
ax.set_xticks(range(8))
ax.set_xticklabels(arm_labels, fontsize=7.5)
ax.set_ylabel("Day-365 burden (nmol/g)", fontsize=8.5)
ax.set_title("A", loc="left", fontsize=12, fontweight="bold")

# Panel B: paired-contrast forest plot (real data)
contrasts = ["Untreated vs.\nFUS alone", "AAV vs.\nAAV+FUS", "SRT vs.\nSRT+FUS", "SRT+AAV vs.\ntri-modal"]
means = [0.0000, -0.0318, 0.0128, -0.0284]
los = [0.0000, -0.0454, 0.0104, -0.0407]
his = [0.0000, -0.0192, 0.0157, -0.0173]
delta_practical = 2.5076  # 1% of untreated day-365 mean

ax = axes[1]
y = np.arange(len(contrasts))
err_lo = [m - l for m, l in zip(means, los)]
err_hi = [h - m for h, m in zip(his, means)]
ax.axvspan(-delta_practical, delta_practical, color="gray", alpha=0.15,
           label=r"practical equivalence ($|\Delta|<\delta_{\mathrm{practical}}$)")
ax.errorbar(means, y, xerr=[err_lo, err_hi], fmt="o", color=C_TRI, markersize=7,
            capsize=4, capthick=1.2, elinewidth=1.2)
ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(y)
ax.set_yticklabels(contrasts, fontsize=8)
ax.set_xlabel("Paired mean $\\Delta$ (nmol/g), 95\\% bootstrap CI", fontsize=8.5)
ax.set_xlim(-5, 5)
ax.set_title("B", loc="left", fontsize=12, fontweight="bold")
ax.legend(fontsize=6.5, loc="lower right")

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_factorial_necessity.png", dpi=300)
plt.savefig(f"{OUT}/fig3_factorial_necessity.tiff", dpi=300)
plt.close()
print("Fig 6 (factorial + paired contrasts) rebuilt.")
