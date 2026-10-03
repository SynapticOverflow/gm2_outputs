import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

mpl.rcParams.update({
    "font.size": 8,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
})

UPLOAD_DIR = "/mnt/user-data/uploads"
OUT = "/home/claude/plos_paper/figures"

files = {
    "Tay-Sachs": "tay-sachs_Y_raw.csv",
    "Sandhoff": "sandhoff_Y_raw.csv",
    "Pompe": "pompe_Y_raw.csv",
    "Krabbe": "krabbe_infantile_Y_raw.csv",
    "GM1": "gm1_infantile_Y_raw.csv",
    "MPS I": "mps1_hurler_Y_raw.csv",
    "MLD": "mld_late_infantile_Y_raw.csv",
    "CLN2": "cln2_Y_raw.csv",
    "Niemann-Pick C": "niemann_pick_c_Y_raw.csv",
    "Fabry": "fabry_Y_raw.csv",
}

fig, axes = plt.subplots(2, 5, figsize=(13, 5.2))
axes = axes.flatten()

for ax, (name, fname) in zip(axes, files.items()):
    df = pd.read_csv(f"{UPLOAD_DIR}/{fname}")
    y = df["Y_raw"].values
    N = len(y) // 16

    ax.hist(y, bins=40, density=True, color="#0072B2", alpha=0.55, edgecolor="none")
    kde = stats.gaussian_kde(y)
    x_grid = np.linspace(y.min(), y.max(), 300)
    ax.plot(x_grid, kde(x_grid), color="#D55E00", lw=1.3)

    ax.set_title(f"{name}\n$N$={N}, $k$={len(y)}", fontsize=7.5)
    ax.set_xlabel("Model output $Y$", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.tick_params(labelsize=6.5)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_raw_output_distributions.png", dpi=300)
plt.savefig(f"{OUT}/fig_raw_output_distributions.tiff", dpi=300)
plt.close()

print("Built real-data distribution figure from actual raw Saltelli sample outputs.")

# Also emit a clean summary table for cross-checking against the paper's Table 2
print("\nSummary statistics (for cross-checking against manuscript Table 2):")
print(f"{'Disease':<16}{'N':>6}{'k':>8}{'mean(Y)':>12}{'std(Y)':>12}")
for name, fname in files.items():
    df = pd.read_csv(f"{UPLOAD_DIR}/{fname}")
    y = df["Y_raw"].values
    N = len(y) // 16
    print(f"{name:<16}{N:>6}{len(y):>8}{y.mean():>12.3f}{y.std():>12.3f}")
