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
C_TRI = "#009E73"

UPLOAD_DIR = "/mnt/user-data/uploads"
OUT = "/home/claude/plos_paper/figures"

# ---------------------------------------------------------------------------
# Fig 3 (rebuilt). Full 8-arm factorial, real strip plot using actual
# per-seed raw_burden arrays (n=200 per arm), not summary bars.
# ---------------------------------------------------------------------------
with open(f"{UPLOAD_DIR}/factorial_arms_shapley.json") as f:
    d365 = json.load(f)

arm_order = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
arm_labels = ["Untreated", "FUS", "AAV", "AAV+\nFUS", "SRT", "SRT+\nFUS", "SRT+\nAAV", "Tri-\nmodal"]
arm_colors = [C_NEUTRAL, C_NEUTRAL, C_DELIV, C_DELIV, "#E69F00", "#E69F00", C_TRI, C_TRI]

arm_lookup = {(a["SRT"], a["AAV"], a["FUS"]): a["raw_burden"] for a in d365["arms"]}

fig, ax = plt.subplots(figsize=(6.6, 4.2))
rng = np.random.default_rng(0)
for i, (key, label, color) in enumerate(zip(arm_order, arm_labels, arm_colors)):
    y = np.array(arm_lookup[key])
    x_jitter = i + rng.normal(0, 0.06, size=len(y))
    ax.scatter(x_jitter, y, s=6, color=color, alpha=0.45, linewidths=0)
    # overlay mean +/- 1 SD as a marker, computed from the same raw data
    ax.errorbar(i, y.mean(), yerr=y.std(), fmt="_", color="black", markersize=18,
                markeredgewidth=2, capsize=5, capthick=1.3, elinewidth=1.3, zorder=5)

ax.set_xticks(range(8))
ax.set_xticklabels(arm_labels, fontsize=8)
ax.set_ylabel("Day-365 GM2 burden (nmol/g)", fontsize=9)

# statistical bracket (real comparison, untreated vs FUS-alone)
y0, y1 = np.array(arm_lookup[(0,0,0)]), np.array(arm_lookup[(0,0,1)])
y_bracket = max(y0.max(), y1.max()) + 40
ax.plot([0, 0, 1, 1], [y_bracket-15, y_bracket, y_bracket, y_bracket-15], color="black", lw=0.8)
ax.text(0.5, y_bracket+8, "n.s.", ha="center", fontsize=8)
ax.set_ylim(-20, y_bracket + 60)

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_factorial_necessity.png", dpi=300)
plt.savefig(f"{OUT}/fig3_factorial_necessity.tiff", dpi=300)
plt.close()
print(f"Fig 3 rebuilt with real n={len(arm_lookup[(0,0,0)])} points per arm.")

# ---------------------------------------------------------------------------
# New Fig: FUS's Shapley contribution over time (real, from own script's
# Shapley computation at 4 time horizons).
# ---------------------------------------------------------------------------
files = {45: "factorial_arms_shapley_day45.json", 90: "factorial_arms_shapley_day90.json",
         180: "factorial_arms_shapley_day180.json", 365: "factorial_arms_shapley.json"}
days, aav_pct, srt_pct, fus_pct = [], [], [], []
for day, fname in files.items():
    with open(f"{UPLOAD_DIR}/{fname}") as f:
        d = json.load(f)
    days.append(day)
    total = d["total_benefit"]
    aav_pct.append(100 * d["shapley_benefit"]["AAV"] / total)
    srt_pct.append(100 * d["shapley_benefit"]["SRT"] / total)
    fus_pct.append(100 * d["shapley_benefit"]["FUS"] / total)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

ax = axes[0]
ax.plot(days, aav_pct, "o-", color=C_DELIV, markersize=6, label="AAV")
ax.plot(days, srt_pct, "s-", color=C_ENZYME, markersize=6, label="SRT")
ax.plot(days, fus_pct, "^-", color=C_TRI, markersize=6, label="FUS")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Day", fontsize=9)
ax.set_ylabel("Shapley contribution (% of total benefit)", fontsize=9)
ax.set_title("A", loc="left", fontsize=12, fontweight="bold")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(days, fus_pct, "^-", color=C_TRI, markersize=7)
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Day", fontsize=9)
ax.set_ylabel("FUS Shapley contribution (%)", fontsize=9)
ax.set_title("B", loc="left", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_fus_shapley_time.png", dpi=300)
plt.savefig(f"{OUT}/fig_fus_shapley_time.tiff", dpi=300)
plt.close()
print("Built temporal FUS Shapley figure.")
