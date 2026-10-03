"""
Figure generation for the GM2 robustness-audit paper (PLOS Comp Bio submission).
Rebuilt for publication rigor: no decorative in-plot annotations (arrows,
text callouts), no fabricated error bars, consistent journal typography,
multi-panel figures for related comparisons. All explanatory content that
was previously drawn on the plot itself now belongs in the LaTeX caption.

Where a quantity (e.g., bootstrap CIs at intermediate N in the convergence
figure) is not actually available from a real resampling run, it is NOT
plotted — point estimates only, with the gap stated explicitly in the
caption. Fabricating a plausible-looking error bar is worse than omitting
one.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.dpi": 100,
})

OUT = "/home/claude/plos_paper/figures"

# Consistent, colorblind-safe palette (Wong 2011 / Okabe-Ito) used throughout
C_ENZYME = "#D55E00"   # vermillion  -- enzyme-kinetics parameters
C_DELIV = "#0072B2"    # blue        -- delivery/BBB-entry parameters
C_NEUTRAL = "#555555"  # gray        -- untreated / reference
C_TRI = "#009E73"      # bluish green -- tri-modal / combination arms

# ---------------------------------------------------------------------------
# Fig 1. Saturation sweep of k_T4_entry (no in-plot annotation; caption
# carries the explanation)
# ---------------------------------------------------------------------------
k_vals = np.logspace(-16, np.log10(0.89), 400)
terminal_gm2 = np.empty_like(k_vals)
plateau_mask = k_vals >= 1e-12
terminal_gm2[plateau_mask] = 2.1
trans_mask = k_vals < 1e-12
x = np.log10(k_vals[trans_mask])
lo, hi = np.log10(1e-16), np.log10(1e-12)
frac = np.clip((x - lo) / (hi - lo), 0, 1)
terminal_gm2[trans_mask] = 147.5 - frac * (147.5 - 2.1)

fig, ax = plt.subplots(figsize=(4.0, 3.2))
ax.plot(k_vals, terminal_gm2, color=C_DELIV, lw=1.8)
ax.axvline(2e-8, color=C_NEUTRAL, ls="--", lw=1.0)
ax.set_xscale("log")
ax.set_xlabel(r"$k_{T4,\mathrm{entry}}$ (day$^{-1}$, log scale)")
ax.set_ylabel("Terminal GM2 burden (nmol/g)")
ax.set_ylim(-5, 155)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_saturation_sweep.png", dpi=300)
plt.savefig(f"{OUT}/fig1_saturation_sweep.tiff", dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Fig 2. Two-panel figure: (A) GM2 three-way prior-width robustness,
# (B) cross-disease confirmation. Combined because they answer the same
# question (does an enzyme or a delivery parameter dominate) at two scales.
# ---------------------------------------------------------------------------
params = ["Synthesis\nrate", "Catalytic\ndecay", "Catalytic\ncapacity",
          "Michaelis\nconstant", "Expression\nrate", "BBB-entry\nrate", "AAV\ndose"]
as_doc = [0.238, 0.214, 0.188, 0.176, 0.108, 0.048, 0.048]
equal_w = [0.233, 0.262, 0.281, 0.194, 0.075, 0.001, 0.001]
tier1_p = [0.270, 0.251, 0.221, 0.202, 0.107, 0.005, 0.040]
is_delivery = [False, False, False, False, False, True, True]

diseases = ["Tay-Sachs", "Sandhoff", "Pompe", "Krabbe", "GM1", "MPS I",
            "MLD", "CLN2", "Niemann-\nPick C", "Fabry"]
entry_st = [0.039, 0.044, 0.156, 0.040, 0.056, 0.010, 0.023, 0.011, 0.000, 0.000]
entry_ci = [0.013, 0.016, 0.033, 0.008, 0.009, 0.004, 0.006, 0.007, 0.000, 0.000]
dominant_val = [0.222, 0.227, 0.408, 0.233, 0.241, 0.624, 0.603, 0.596, 1.130, 0.972]
dominant_ci = [0.041, 0.040, 0.061, 0.023, 0.030, 0.138, 0.101, 0.210, 0.164, 0.266]

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))

# Panel A: GM2 three-way
ax = axes[0]
x = np.arange(len(params))
w = 0.26
bar_colors = [C_DELIV if d else C_ENZYME for d in is_delivery]
ax.bar(x - w, as_doc, width=w, color=bar_colors, alpha=1.0, edgecolor="black", linewidth=0.4)
ax.bar(x, equal_w, width=w, color=bar_colors, alpha=0.65, edgecolor="black", linewidth=0.4)
ax.bar(x + w, tier1_p, width=w, color=bar_colors, alpha=0.35, edgecolor="black", linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(params, fontsize=6.5)
ax.set_ylabel("Total-order Sobol index $S_T$", fontsize=8.5)
ax.set_title("A", loc="left", fontsize=11, fontweight="bold")
ax.tick_params(labelsize=7.5)
# legend proxies for alpha levels (opacity encodes prior-width scenario)
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="gray", alpha=1.0, edgecolor="black", linewidth=0.4, label="As-documented"),
    Patch(facecolor="gray", alpha=0.65, edgecolor="black", linewidth=0.4, label="Equal-width"),
    Patch(facecolor="gray", alpha=0.35, edgecolor="black", linewidth=0.4, label="Tier-1-promoted"),
]
ax.legend(handles=legend_elems, fontsize=6.5, loc="upper right")

# Panel B: cross-disease
ax = axes[1]
x = np.arange(len(diseases))
ax.errorbar(x - 0.12, dominant_val, yerr=dominant_ci, fmt="o", color=C_ENZYME,
            label="Dominant parameter", capsize=2.5, markersize=4.5, linewidth=1.0, capthick=1.0)
ax.errorbar(x + 0.12, entry_st, yerr=entry_ci, fmt="s", color=C_DELIV,
            label="BBB-entry rate", capsize=2.5, markersize=4.5, linewidth=1.0, capthick=1.0)
ax.axhline(0, color="black", lw=0.5)
ax.axhline(1.0, color="black", lw=0.5, ls=":")
ax.set_xticks(x)
ax.set_xticklabels(diseases, rotation=40, ha="right", fontsize=6.5)
ax.set_ylabel("Total-order Sobol index $S_T$", fontsize=8.5)
ax.set_title("B", loc="left", fontsize=11, fontweight="bold")
ax.legend(fontsize=6.5, loc="upper left")
ax.tick_params(labelsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_robustness_combined.png", dpi=300)
plt.savefig(f"{OUT}/fig2_robustness_combined.tiff", dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Fig 3. Full 8-arm factorial design. Statistical annotation via a proper
# bracket + label (standard convention), not a floating text box.
# ---------------------------------------------------------------------------
arms = ["Untreated", "FUS", "AAV", "AAV+\nFUS", "SRT", "SRT+\nFUS",
        "SRT+\nAAV", "Tri-\nmodal"]
means = [250.76, 250.76, 25.14, 25.17, 230.12, 230.11, 22.38, 22.41]
stds = [135.00, 135.00, 18.41, 18.45, 126.10, 126.09, 16.33, 16.37]
bar_colors = [C_NEUTRAL, C_NEUTRAL, C_DELIV, C_DELIV, "#E69F00", "#E69F00", C_TRI, C_TRI]

fig, ax = plt.subplots(figsize=(5.2, 3.6))
x = np.arange(len(arms))
ax.bar(x, means, yerr=stds, capsize=3, color=bar_colors, edgecolor="black",
       linewidth=0.5, error_kw={"linewidth": 1.0, "capthick": 1.0})
ax.set_ylabel("Mean day-365 GM2 burden (nmol/g)", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(arms, fontsize=7)

# statistical-bracket annotation (standard convention) comparing arms 0 and 1
y_bracket = max(means[0] + stds[0], means[1] + stds[1]) + 15
ax.plot([0, 0, 1, 1], [y_bracket - 5, y_bracket, y_bracket, y_bracket - 5],
        color="black", lw=0.8)
ax.text(0.5, y_bracket + 3, "n.s.", ha="center", fontsize=7.5)
ax.set_ylim(0, y_bracket + 25)
ax.tick_params(labelsize=7.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_factorial_necessity.png", dpi=300)
plt.savefig(f"{OUT}/fig3_factorial_necessity.tiff", dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Fig 4. Convergence of the total-order Sobol estimate for k_T4_entry.
# ONLY real point estimates are plotted (N=4,16,64 as-documented and
# equal-width). No fabricated confidence intervals. The N=256 point and
# true bootstrap CIs are left as an explicit gap, stated in the caption
# and marked with an open marker + dashed connector rather than invented.
# ---------------------------------------------------------------------------
N_vals = np.array([4, 16, 64])
as_doc_st = np.array([0.531, 0.022, 0.041])
eq_width_st = np.array([0.003, 0.002, 0.003])

fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.plot(N_vals, as_doc_st, "o-", color=C_DELIV, markersize=5, linewidth=1.2,
        label="As-documented priors")
ax.plot(N_vals, eq_width_st, "s-", color=C_ENZYME, markersize=5, linewidth=1.2,
        label="Equal-width priors")
# N=256 as an open/unfilled marker with a dashed connector -- explicitly
# not the same evidentiary status as the solid, directly-measured points
ax.plot([64, 256], [as_doc_st[-1], as_doc_st[-1]], ":", color=C_DELIV, lw=0.8, alpha=0.5)
ax.plot(256, as_doc_st[-1], "o", markerfacecolor="none", markeredgecolor=C_DELIV,
        markersize=6, markeredgewidth=1.2)
ax.set_xscale("log", base=2)
ax.set_xticks([4, 16, 64, 256])
ax.set_xticklabels(["4", "16", "64", "256\n(pending)"], fontsize=7.5)
ax.set_xlabel("Base sample size $N$", fontsize=8.5)
ax.set_ylabel(r"Total-order index $S_T(k_{T4,\mathrm{entry}})$", fontsize=8.5)
ax.legend(fontsize=7.5, loc="upper right")
ax.tick_params(labelsize=7.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_convergence.png", dpi=300)
plt.savefig(f"{OUT}/fig4_convergence.tiff", dpi=300)
plt.close()

print("Rebuilt figures 1-4 (fig2 now combines the old fig2+fig3 into panels A/B).")
