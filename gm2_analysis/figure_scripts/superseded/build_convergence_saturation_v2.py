import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

DATA_DIR = "/home/claude/task_results/gm2_lsd_tasks1-6_results"
OUT = "/home/claude/plos_paper/figures"

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

# Panel A: real convergence grid (Task 2)
with open(f"{DATA_DIR}/convergence_grid.json") as f:
    conv = json.load(f)

N_vals = np.array(conv["N"])
st_vals = np.array(conv["k_entry_ST"])
se_vals = np.array(conv["bootstrap_se"])

ax = axes[0]
ax.errorbar(N_vals, st_vals, yerr=se_vals, fmt="o-", color=C_DELIV,
            markersize=7, linewidth=1.2, capsize=4, capthick=1.2, elinewidth=1.2)
ax.axhline(0.01, color="gray", ls=":", lw=0.8)
ax.set_xscale("log", base=2)
ax.set_xticks(N_vals)
ax.set_xticklabels(N_vals)
ax.set_xlabel("Base sample size $N$", fontsize=9.5)
ax.set_ylabel(r"Total-order index $S_T(k_{T4,\mathrm{entry}})$", fontsize=9.5)
ax.set_title("A", loc="left", fontsize=13, fontweight="bold")
ax.tick_params(labelsize=8.5)

# Panel B: real saturation sweep (Task 3, correct model gm2_model_v2)
with open(f"{DATA_DIR}/saturation_sweep_gm2_model_v2.json") as f:
    sat = json.load(f)

k_vals = np.array(sat["k_values"])
mean_burden = np.array(sat["mean_burden"])
std_burden = np.array(sat["std_burden"])

ax = axes[1]
ax.plot(k_vals, mean_burden, "o-", color=C_DELIV, markersize=5, linewidth=1.2)
ax.fill_between(k_vals, mean_burden - std_burden, mean_burden + std_burden,
                 color=C_DELIV, alpha=0.15)
ax.axvline(0.06, color=C_NEUTRAL, ls="--", lw=1.1, label="Calibrated value (0.06)")
ax.set_xscale("log")
ax.set_xlabel(r"$k_{T4,\mathrm{entry}}$ (normalized units, log scale)", fontsize=9.5)
ax.set_ylabel("Terminal substrate burden (nmol/g)", fontsize=9.5)
ax.set_title("B", loc="left", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="lower left")
ax.tick_params(labelsize=8.5)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_convergence_saturation.png", dpi=300)
plt.savefig(f"{OUT}/fig_convergence_saturation.tiff", dpi=300)
plt.close()

print("Rebuilt with real data:")
print(f"  Panel A: N={list(N_vals)}, ST={[round(x,4) for x in st_vals]}, SE={[round(x,4) for x in se_vals]}")
print(f"  Panel B: {len(k_vals)} points, k range [{k_vals.min():.1e}, {k_vals.max():.1e}], "
      f"burden range [{mean_burden.min():.2f}, {mean_burden.max():.2f}]")
print(f"  Calibrated k_T4_entry = 0.06 is {np.log10(0.06/k_vals.max()):.1f} orders of magnitude above the tested max")
