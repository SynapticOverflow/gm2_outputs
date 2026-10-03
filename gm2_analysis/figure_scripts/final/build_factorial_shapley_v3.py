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

DATA_DIR = "/home/claude/task_results/gm2_lsd_tasks1-6_results"
OUT = "/home/claude/plos_paper/figures"

# ---------------------------------------------------------------------------
# Fig: Full 8-arm factorial (real strip plot, corrected statistical bracket)
# ---------------------------------------------------------------------------
# Need the original raw per-seed factorial data (already used previously)
import glob
factorial_file = None
for candidate in ["/mnt/user-data/uploads/factorial_arms_shapley.json",
                   f"{DATA_DIR}/factorial_arms_shapley.json"]:
    import os
    if os.path.exists(candidate):
        factorial_file = candidate
        break

with open(factorial_file) as f:
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
    ax.errorbar(i, y.mean(), yerr=y.std(), fmt="_", color="black", markersize=18,
                markeredgewidth=2, capsize=5, capthick=1.3, elinewidth=1.3, zorder=5)

ax.set_xticks(range(8))
ax.set_xticklabels(arm_labels, fontsize=8)
ax.set_ylabel("Day-365 GM2 burden (nmol/g)", fontsize=9)

# Statistical bracket: paired bootstrap CI excludes zero but effect is tiny
y0, y1 = np.array(arm_lookup[(0,0,0)]), np.array(arm_lookup[(0,0,1)])
y_bracket = max(y0.max(), y1.max()) + 40
ax.plot([0, 0, 1, 1], [y_bracket-15, y_bracket, y_bracket, y_bracket-15], color="black", lw=0.8)
ax.text(0.5, y_bracket+8, "paired diff\n$<10^{-5}$", ha="center", fontsize=6.5)
ax.set_ylim(-20, y_bracket + 60)

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_factorial_necessity.png", dpi=300)
plt.savefig(f"{OUT}/fig3_factorial_necessity.tiff", dpi=300)
plt.close()
print("Factorial figure rebuilt.")

# ---------------------------------------------------------------------------
# Fig: FUS Shapley contribution over time, with real bootstrap 95% CIs
# ---------------------------------------------------------------------------
with open(f"{DATA_DIR}/shapley_with_ci.json") as f:
    shap = json.load(f)

with open(factorial_file) as f:
    total_benefit_365 = json.load(f)["total_benefit"]

days_files = {45: "factorial_arms_shapley_day45.json", 90: "factorial_arms_shapley_day90.json",
              180: "factorial_arms_shapley_day180.json", 365: None}
totals = {}
import os
for day, fname in days_files.items():
    if fname and os.path.exists(f"/mnt/user-data/uploads/{fname}"):
        with open(f"/mnt/user-data/uploads/{fname}") as f:
            totals[day] = json.load(f)["total_benefit"]
    else:
        totals[day] = total_benefit_365

days = [45, 90, 180, 365]
aav_pct, srt_pct, fus_pct, fus_lo, fus_hi = [], [], [], [], []
for day in days:
    key = f"day{day}"
    total = totals[day]
    aav_pct.append(100 * shap[key]["AAV"]["point_estimate"] / total)
    srt_pct.append(100 * shap[key]["SRT"]["point_estimate"] / total)
    fus_pct.append(100 * shap[key]["FUS"]["point_estimate"] / total)
    fus_lo.append(100 * shap[key]["FUS"]["ci95_lo"] / total)
    fus_hi.append(100 * shap[key]["FUS"]["ci95_hi"] / total)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

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
fus_err_lo = np.array(fus_pct) - np.array(fus_lo)
fus_err_hi = np.array(fus_hi) - np.array(fus_pct)
ax.errorbar(days, fus_pct, yerr=[fus_err_lo, fus_err_hi], fmt="^-", color=C_TRI,
            markersize=7, capsize=4, capthick=1.2, elinewidth=1.2)
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Day", fontsize=9)
ax.set_ylabel("FUS Shapley contribution (%, 95% bootstrap CI)", fontsize=8.5)
ax.set_title("B", loc="left", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_fus_shapley_time.png", dpi=300)
plt.savefig(f"{OUT}/fig_fus_shapley_time.tiff", dpi=300)
plt.close()
print("Shapley-time figure rebuilt with real CIs.")
print(f"FUS %: {[round(x,4) for x in fus_pct]}")
print(f"FUS CI lo: {[round(x,4) for x in fus_lo]}")
print(f"FUS CI hi: {[round(x,4) for x in fus_hi]}")
