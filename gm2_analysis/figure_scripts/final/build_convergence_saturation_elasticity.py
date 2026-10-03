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

C_DELIV = "#0072B2"

# Source data: convergence_grid.json and saturation_sweep_gm2_model_v2.json,
# both archived in gm2_analysis/sobol_outputs/ in the paper's data repository.
DATA_DIR = "gm2_analysis/sobol_outputs"
OUT = "."

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

# Panel A: convergence
with open(f"{DATA_DIR}/convergence_grid.json") as f:
    conv = json.load(f)
N_vals = np.array(conv["N"])
st_vals = np.array(conv["k_entry_ST"])
se_vals = np.array(conv["bootstrap_se"])

ax = axes[0]
ax.errorbar(N_vals, st_vals, yerr=se_vals, fmt="o-", color=C_DELIV,
            markersize=6, linewidth=1.1, capsize=4, capthick=1.1, elinewidth=1.1)
ax.axhline(0.01, color="gray", ls=":", lw=0.8)
ax.set_xscale("log", base=2)
ax.set_xticks(N_vals)
ax.set_xticklabels(N_vals)
ax.set_xlabel("Base sample size $N$", fontsize=8.5)
ax.set_ylabel(r"$S_T(k_{T4,\mathrm{entry}})$", fontsize=8.5)
ax.set_title("A", loc="left", fontsize=12, fontweight="bold")
ax.tick_params(labelsize=7.5)

# Panel B: saturation sweep
with open(f"{DATA_DIR}/saturation_sweep_gm2_model_v2.json") as f:
    sat = json.load(f)
k_vals = np.array(sat["k_values"])
mean_burden = np.array(sat["mean_burden"])
std_burden = np.array(sat["std_burden"])

ax = axes[1]
ax.plot(k_vals, mean_burden, "o-", color=C_DELIV, markersize=4, linewidth=1.1)
ax.fill_between(k_vals, mean_burden - std_burden, mean_burden + std_burden,
                 color=C_DELIV, alpha=0.15)
ax.axvline(0.06, color="#555555", ls="--", lw=1.0, label="Calibrated (0.06)")
ax.set_xscale("log")
ax.set_xlabel(r"$k_{T4,\mathrm{entry}}$ (log scale)", fontsize=8.5)
ax.set_ylabel("Terminal burden (nmol/g)", fontsize=8.5)
ax.set_title("B", loc="left", fontsize=12, fontweight="bold")
ax.legend(fontsize=7, loc="lower left")
ax.tick_params(labelsize=7.5)

# Panel C: elasticity (computed from the same sweep, not a separate source file)
log_k = np.log(k_vals)
log_Y = np.log(mean_burden)
elasticity = np.gradient(log_Y, log_k)

ax = axes[2]
ax.plot(k_vals, elasticity, "o-", color=C_DELIV, markersize=3.5, linewidth=1.0)
ax.axhline(0, color="black", lw=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"$k_{T4,\mathrm{entry}}$ (log scale)", fontsize=8.5)
ax.set_ylabel(r"$E(k)=\partial \log Y/\partial \log k$", fontsize=8.5)
ax.set_title("C", loc="left", fontsize=12, fontweight="bold")
ax.tick_params(labelsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_convergence_saturation_elasticity.png", dpi=300)
plt.savefig(f"{OUT}/fig_convergence_saturation_elasticity.tiff", dpi=300)
plt.close()
print("Merged 3-panel figure (convergence, saturation, elasticity) built -- this is the figure actually used in the paper, superseding make_figures.py's older separate fig_convergence_saturation.png + fig_elasticity.png.")
