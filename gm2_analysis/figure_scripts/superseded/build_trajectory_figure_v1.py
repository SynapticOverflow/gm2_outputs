import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    "Tay-Sachs": "tay-sachs_trajectory.csv",
    "Sandhoff": "sandhoff_trajectory.csv",
    "Pompe": "pompe_trajectory.csv",
    "Krabbe": "krabbe_infantile_trajectory.csv",
    "GM1": "gm1_infantile_trajectory.csv",
    "MPS I": "mps1_hurler_trajectory.csv",
    "MLD": "mld_late_infantile_trajectory.csv",
    "CLN2": "cln2_trajectory.csv",
    "Niemann-Pick C": "niemann_pick_c_trajectory.csv",
    "Fabry": "fabry_trajectory.csv",
}

fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), sharey=False)
axes = axes.flatten()

for ax, (name, fname) in zip(axes, files.items()):
    df = pd.read_csv(f"{UPLOAD_DIR}/{fname}")
    t = df["time_days"].values
    y = df["gm2_brain"].values

    ax.plot(t, y, color="#0072B2", lw=1.0)
    ax.set_title(f"{name}", fontsize=8)
    ax.set_xlabel("Day", fontsize=7)
    ax.set_ylabel("Brain GM2 (nmol/g)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_xlim(0, t.max())

plt.tight_layout()
plt.savefig(f"{OUT}/fig_real_trajectories.png", dpi=300)
plt.savefig(f"{OUT}/fig_real_trajectories.tiff", dpi=300)
plt.close()

print("Built real trajectory figure from actual uploaded time-course data.")
print()
print(f"{'Disease':<16}{'t_end':>8}{'Y0':>10}{'Y_min':>10}{'t(min)':>8}{'Y_end':>10}")
for name, fname in files.items():
    df = pd.read_csv(f"{UPLOAD_DIR}/{fname}")
    t = df["time_days"].values
    y = df["gm2_brain"].values
    tmin_idx = np.argmin(y)
    print(f"{name:<16}{t[-1]:>8.0f}{y[0]:>10.2f}{y[tmin_idx]:>10.2f}{t[tmin_idx]:>8.1f}{y[-1]:>10.2f}")
