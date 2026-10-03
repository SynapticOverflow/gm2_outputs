import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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

fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
axes = axes.flatten()

zoom_log = []

for ax, (name, fname) in zip(axes, files.items()):
    df = pd.read_csv(f"{UPLOAD_DIR}/{fname}")
    t = df["time_days"].values
    y = df["gm2_brain"].values

    ax.plot(t, y, color="#0072B2", lw=0.8)
    ax.set_title(name, fontsize=8)
    ax.set_xlabel("Day", fontsize=7)
    ax.set_ylabel("Substrate burden (nmol/g)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_xlim(0, t.max())

    # Zoom window: a fixed-width slice starting shortly after the initial
    # crash, scaled to each disease's own timescale (10% of total duration,
    # min 20 days), where post-crash stochastic fluctuation is visible.
    t_end = t.max()
    zoom_start = 0.25 * t_end
    zoom_width = max(0.08 * t_end, 15)
    zoom_end = zoom_start + zoom_width
    mask = (t >= zoom_start) & (t <= zoom_end)

    axins = inset_axes(ax, width="55%", height="55%", loc="upper right")
    axins.plot(t[mask], y[mask], color="#D55E00", lw=0.7, marker=".", markersize=1.2)
    axins.set_xticks([])
    axins.tick_params(labelsize=5)
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)

    zoom_log.append((name, zoom_start, zoom_end, mask.sum(),
                      y[mask].mean(), y[mask].std()))

plt.tight_layout()
plt.savefig(f"{OUT}/fig_real_trajectories.png", dpi=300)
plt.savefig(f"{OUT}/fig_real_trajectories.tiff", dpi=300)
plt.close()

print("Rebuilt trajectory figure with zoomed insets revealing raw per-timestep noise.")
print()
print(f"{'Disease':<16}{'zoom window (d)':<20}{'n pts':>8}{'mean':>10}{'std':>10}")
for name, zs, ze, n, m, s in zoom_log:
    print(f"{name:<16}{f'{zs:.0f}-{ze:.0f}':<20}{n:>8}{m:>10.3f}{s:>10.3f}")
