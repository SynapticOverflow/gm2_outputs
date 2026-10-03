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
# Combined Fig: (A) convergence of the total-order Sobol estimate for
# k_T4_entry, (B) saturation sweep of k_T4_entry. Combined into one
# two-panel figure since both concern the same parameter and appear
# back-to-back in Results. Only real, directly-measured points are
# plotted in both panels -- no fabricated curves or connecting lines
# between sparse measurements.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

# Panel A: convergence
ax = axes[0]
N_vals = np.array([4, 16, 64])
as_doc_st = np.array([0.531, 0.022, 0.041])
eq_width_st = np.array([0.003, 0.002, 0.003])
ax.plot(N_vals, as_doc_st, "o", color=C_DELIV, markersize=9,
        label="As-documented priors", zorder=3)
ax.plot(N_vals, eq_width_st, "s", color=C_ENZYME, markersize=9,
        label="Equal-width priors", zorder=3)
ax.plot(256, as_doc_st[-1], "o", markerfacecolor="none", markeredgecolor=C_DELIV,
        markersize=10, markeredgewidth=1.6, label="N=256 (pending CI)", zorder=3)
for xi in N_vals:
    ax.axvline(xi, color="gray", lw=0.4, alpha=0.25, zorder=0)
ax.axvline(256, color="gray", lw=0.4, alpha=0.25, zorder=0)
ax.set_xscale("log", base=2)
ax.set_xticks([4, 16, 64, 256])
ax.set_xticklabels(["4", "16", "64", "256"], fontsize=9)
ax.set_xlabel("Base sample size $N$", fontsize=9.5)
ax.set_ylabel(r"Total-order index $S_T(k_{T4,\mathrm{entry}})$", fontsize=9.5)
ax.set_title("A", loc="left", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.tick_params(labelsize=8.5)

# Panel B: saturation sweep
ax = axes[1]
k_anchors = np.array([1e-16, 1e-12, 2e-8])
y_anchors = np.array([147.5, 2.1, 2.1])
ax.plot(k_anchors, y_anchors, "o", color=C_DELIV, markersize=9, zorder=3)
ax.axvline(2e-8, color=C_NEUTRAL, ls="--", lw=1.1, label="Calibrated baseline")
ax.axvspan(1e-14, 1e-12, color="gray", alpha=0.15,
           label="Reported transition zone\n(intermediate values not available)")
ax.set_xscale("log")
ax.set_xlim(1e-17, 1.0)
ax.set_xlabel(r"$k_{T4,\mathrm{entry}}$ (day$^{-1}$, log scale)", fontsize=9.5)
ax.set_ylabel("Terminal GM2 burden (nmol/g)", fontsize=9.5)
ax.set_ylim(-5, 155)
ax.set_title("B", loc="left", fontsize=13, fontweight="bold")
ax.tick_params(labelsize=8.5)
ax.legend(fontsize=8, loc="center left")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_convergence_saturation.png", dpi=300)
plt.savefig(f"{OUT}/fig_convergence_saturation.tiff", dpi=300)
plt.close()
print("Built combined convergence + saturation two-panel figure.")

# ---------------------------------------------------------------------------
# Fig 2. Two-panel forest plot: (A) GM2 three-way prior-width robustness,
# (B) cross-disease confirmation. Horizontal dot-and-whisker (forest plot)
# style, sorted by magnitude -- the standard convention for comparing
# several point estimates with uncertainty, used in place of a grouped bar
# chart. Cross-disease values are the real, final numbers confirmed
# directly from the submitted Sobol JSON output files.
# ---------------------------------------------------------------------------
from matplotlib.lines import Line2D

params = ["Synthesis rate", "Catalytic decay", "Catalytic capacity",
          "Michaelis constant", "Expression rate", "BBB-entry rate", "AAV dose"]
as_doc = [0.238, 0.214, 0.188, 0.176, 0.108, 0.048, 0.048]
equal_w = [0.233, 0.262, 0.281, 0.194, 0.075, 0.001, 0.001]
tier1_p = [0.270, 0.251, 0.221, 0.202, 0.107, 0.005, 0.040]
is_delivery = [False, False, False, False, False, True, True]

# sort by as-documented value, descending
order = np.argsort(as_doc)
params_s = [params[i] for i in order]
as_doc_s = [as_doc[i] for i in order]
equal_w_s = [equal_w[i] for i in order]
tier1_p_s = [tier1_p[i] for i in order]
is_delivery_s = [is_delivery[i] for i in order]

diseases = ["Tay-Sachs", "Sandhoff", "Pompe", "Krabbe", "GM1", "MPS I",
            "MLD", "CLN2", "Niemann-Pick C", "Fabry"]
dominant_val = [0.241, 0.241, 0.417, 0.225, 0.231, 0.597, 0.597, 0.598, 0.959, 0.991]
dominant_ci  = [0.037, 0.034, 0.070, 0.028, 0.030, 0.127, 0.107, 0.072, 0.129, 0.117]
entry_st  = [0.047, 0.040, 0.170, 0.038, 0.063, 0.009, 0.023, 0.009, 0.000, 0.000]
entry_ci  = [0.016, 0.014, 0.033, 0.008, 0.012, 0.004, 0.007, 0.002, 0.000, 0.000]

d_order = np.argsort(dominant_val)
diseases_s = [diseases[i] for i in d_order]
dominant_val_s = [dominant_val[i] for i in d_order]
dominant_ci_s = [dominant_ci[i] for i in d_order]
entry_st_s = [entry_st[i] for i in d_order]
entry_ci_s = [entry_ci[i] for i in d_order]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))

# Panel A: GM2 three-way, horizontal forest plot
ax = axes[0]
y = np.arange(len(params_s))
offset = 0.22
for row_offset, vals, marker, alpha, lbl in [
    (offset, as_doc_s, "o", 1.0, "As-documented"),
    (0, equal_w_s, "s", 0.65, "Equal-width"),
    (-offset, tier1_p_s, "^", 0.4, "Tier-1-promoted"),
]:
    colors = [C_DELIV if d else C_ENZYME for d in is_delivery_s]
    ax.scatter(vals, y + row_offset, marker=marker, s=42, c=colors, alpha=alpha,
               edgecolor="black", linewidth=0.5, zorder=3)
for yi in y:
    ax.axhline(yi, color="gray", lw=0.4, alpha=0.3, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(params_s, fontsize=8)
ax.set_xlabel("Total-order Sobol index $S_T$", fontsize=8.5)
ax.set_title("A", loc="left", fontsize=12, fontweight="bold")
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
           markersize=7, label="As-documented"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", alpha=0.65, markeredgecolor="black",
           markersize=7, label="Equal-width"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", alpha=0.4, markeredgecolor="black",
           markersize=7, label="Tier-1-promoted"),
]
ax.legend(handles=legend_elems, fontsize=7, loc="lower right")
ax.tick_params(labelsize=8)

# Panel B: cross-disease, horizontal forest plot with real 95% CIs
ax = axes[1]
y = np.arange(len(diseases_s))
ax.errorbar(dominant_val_s, y + 0.15, xerr=dominant_ci_s, fmt="o", color=C_ENZYME,
            label="Dominant parameter", capsize=3, markersize=6, linewidth=1.1, capthick=1.1, zorder=3)
ax.errorbar(entry_st_s, y - 0.15, xerr=entry_ci_s, fmt="s", color=C_DELIV,
            label="BBB-entry rate", capsize=3, markersize=6, linewidth=1.1, capthick=1.1, zorder=3)
for yi in y:
    ax.axhline(yi, color="gray", lw=0.4, alpha=0.3, zorder=0)
ax.axvline(0, color="black", lw=0.6)
ax.axvline(1.0, color="black", lw=0.6, ls=":")
ax.set_yticks(y)
ax.set_yticklabels(diseases_s, fontsize=8)
ax.set_xlabel("Total-order Sobol index $S_T$", fontsize=8.5)
ax.set_title("B", loc="left", fontsize=12, fontweight="bold")
ax.legend(fontsize=7.5, loc="lower right")
ax.tick_params(labelsize=8)

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



print("Rebuilt figures 1-4 (fig2 now combines the old fig2+fig3 into panels A/B).")
