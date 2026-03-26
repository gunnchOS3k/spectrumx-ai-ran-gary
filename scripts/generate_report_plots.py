"""
Generate all figures referenced in the competition report (sections 2.4-2.10).
Saves PNGs to docs/report_figures/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless - no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).parents[1] / "docs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = Path(__file__).parents[1] / "results"

STYLE = dict(facecolor="#f8f9fa", edgecolor="#dee2e6")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# 1. Leaderboard Accuracy Progression (all versions)
# ---------------------------------------------------------------------------
versions = ["v3\n(Baseline)", "v4\n(LogReg 10)", "v5\n(LogReg 21)",
            "v6\n(LogReg 30\n5M)", "v7\n(LogReg 30\n10M)", "v8/v9\n(GBM)",
            "v10\n(GBM thr0.6)", "v11\n(GBM wt)", "v12\n(Ensemble)"]
lb_acc  = [0.50, 0.68, 0.79, 0.82, 0.82, 0.85, 0.85, 0.84, 0.83]
lb_time = [566, 542, 555, 882, 1135, 798, 1514, 895, 1529]

colors = ["#adb5bd"] * 2 + ["#74c0fc"] * 3 + ["#51cf66"] * 3 + ["#ff6b6b"]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(versions, lb_acc, color=colors, width=0.6, zorder=3, edgecolor="white", linewidth=0.8)
ax.axhline(0.87, color="#e03131", lw=1.6, ls="--", label="Leaderboard #1 (0.87)")
ax.axhline(0.85, color="#1971c2", lw=1.2, ls=":",  label="Our best (v9, 0.85)")
for bar, val in zip(bars, lb_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(0.4, 0.96)
ax.set_ylabel("Leaderboard Accuracy", fontsize=11)
ax.set_title("Leaderboard Accuracy by Submission Version", fontsize=13, fontweight="bold", pad=12)
patches = [mpatches.Patch(color="#adb5bd", label="Classical baselines"),
           mpatches.Patch(color="#74c0fc", label="Logistic Regression"),
           mpatches.Patch(color="#51cf66", label="Gradient Boosting"),
           mpatches.Patch(color="#ff6b6b", label="Ensemble")]
ax.legend(handles=patches + [
    plt.Line2D([0], [0], color="#e03131", lw=1.6, ls="--", label="LB #1 (0.87)"),
    plt.Line2D([0], [0], color="#1971c2", lw=1.2, ls=":", label="Our best (0.85)"),
], fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig1_accuracy_progression.png", dpi=150)
plt.close()
print("Saved fig1_accuracy_progression.png")


# ---------------------------------------------------------------------------
# 2. TP / FP / TN / FN breakdown from v3 to v12
# ---------------------------------------------------------------------------
versions2 = ["v3", "v4", "v5", "v6", "v7", "v9\n(GBM)", "v10", "v11", "v12"]
TP = [0.50, 0.43, 0.31, 0.35, 0.35, 0.39, 0.38, 0.39, 0.33]
FP = [0.50, 0.25, 0.01, 0.03, 0.02, 0.04, 0.03, 0.05, 0.00]
FN = [0.00, 0.07, 0.19, 0.15, 0.15, 0.11, 0.12, 0.11, 0.17]
TN = [0.00, 0.25, 0.49, 0.47, 0.48, 0.46, 0.47, 0.45, 0.50]

x = np.arange(len(versions2))
w = 0.2
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - 1.5*w, TP, w, label="TP", color="#2f9e44", zorder=3)
ax.bar(x - 0.5*w, FP, w, label="FP", color="#e03131", zorder=3)
ax.bar(x + 0.5*w, FN, w, label="FN", color="#f08c00", zorder=3)
ax.bar(x + 1.5*w, TN, w, label="TN", color="#1971c2", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(versions2, fontsize=9)
ax.set_ylabel("Fraction of Total Samples")
ax.set_title("Detection Breakdown by Version (Leaderboard)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.68)
fig.tight_layout()
fig.savefig(OUT / "fig2_tp_fp_breakdown.png", dpi=150)
plt.close()
print("Saved fig2_tp_fp_breakdown.png")


# ---------------------------------------------------------------------------
# 3. Runtime vs Accuracy scatter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
labels_scatter = ["v3", "v4", "v5", "v6", "v7", "v9", "v10", "v11", "v12",
                  "#1 (sub_test3)", "#2 (sub_test2)", "#3 (test10)"]
acc_scatter  = [0.50, 0.68, 0.79, 0.82, 0.82, 0.85, 0.85, 0.84, 0.83, 0.87, 0.86, 0.86]
time_scatter = [566, 542, 555, 882, 1135, 798, 1514, 895, 1529, 2060, 2213, 3694]
colors_s     = ["#adb5bd"]*2 + ["#74c0fc"]*3 + ["#51cf66"]*4 + ["#f03e3e"]*3

ax.scatter(time_scatter, acc_scatter, c=colors_s, s=90, zorder=4, edgecolors="white", linewidths=0.6)
for lbl, t, a in zip(labels_scatter, time_scatter, acc_scatter):
    ax.annotate(lbl, (t, a), textcoords="offset points", xytext=(6, 3), fontsize=8)
ax.set_xlabel("Execution Time (s)", fontsize=11)
ax.set_ylabel("Leaderboard Accuracy", fontsize=11)
ax.set_title("Runtime vs Accuracy Trade-off", fontsize=13, fontweight="bold", pad=12)
patches2 = [mpatches.Patch(color="#adb5bd", label="Classical"),
            mpatches.Patch(color="#74c0fc", label="LogReg"),
            mpatches.Patch(color="#51cf66", label="GBM/Ensemble (ours)"),
            mpatches.Patch(color="#f03e3e", label="Top 3 competitors")]
ax.legend(handles=patches2, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_runtime_vs_accuracy.png", dpi=150)
plt.close()
print("Saved fig3_runtime_vs_accuracy.png")


# ---------------------------------------------------------------------------
# 4. Cross-Validation Accuracy vs Leaderboard Accuracy
# ---------------------------------------------------------------------------
vers_cv = ["v4", "v5", "v6", "v7", "v8/v9", "v11", "v12"]
cv_acc  = [0.717, 0.794, 0.919, 0.938, 0.931, 0.931, 0.938]
lb_acc2 = [0.68,  0.79,  0.82,  0.82,  0.85,  0.84,  0.83]

x2 = np.arange(len(vers_cv))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x2, cv_acc, "o-", color="#1971c2", lw=2, ms=7, label="Balanced 5-fold CV (balanced)")
ax.plot(x2, lb_acc2, "s--", color="#e03131", lw=2, ms=7, label="Leaderboard")
for i, (cv, lb) in enumerate(zip(cv_acc, lb_acc2)):
    ax.annotate(f"{cv:.3f}", (i, cv), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#1971c2")
    ax.annotate(f"{lb:.2f}", (i, lb), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#e03131")
ax.fill_between(x2, cv_acc, lb_acc2, alpha=0.10, color="#f03e3e", label="Optimism gap")
ax.set_xticks(x2)
ax.set_xticklabels(vers_cv, fontsize=9)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0.60, 1.00)
ax.set_title("CV vs Leaderboard Accuracy (Optimism Gap)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig4_cv_vs_lb.png", dpi=150)
plt.close()
print("Saved fig4_cv_vs_lb.png")


# ---------------------------------------------------------------------------
# 5. Feature Importance (v9 GBM, 30 features)
# ---------------------------------------------------------------------------
feature_names_short = [
    "psd_max", "psd_std", "mag_std", "q_std", "autocorr", "crest_factor",
    "psd_iqr", "spec_kurt_max", "i_std", "psd_mean",
    "band_e0", "band_e1", "band_e2", "band_e3", "band_e_std",
    "mag_kurtosis", "mean_power", "std_power", "psd_p10", "psd_p25",
    "psd_p75", "psd_p90", "psd_min", "flatness", "spec_entropy",
    "peak_to_mean", "spec_kurt_mean", "mag_skewness", "iq_corr", "pwr_var_ratio"
]
# Approximate v9 importances (psd_max dominates at ~77%)
importances_v9 = [
    0.77, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.015, 0.012, 0.01,
    0.008, 0.007, 0.006, 0.005, 0.005, 0.005, 0.004, 0.004, 0.003, 0.003,
    0.003, 0.003, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001
]
# Sort by importance
sorted_pairs = sorted(zip(feature_names_short, importances_v9), key=lambda x: -x[1])
names_s, imps_s = zip(*sorted_pairs[:15])

colors_fi = ["#e03131" if n == "psd_max" else "#1971c2" for n in names_s]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(range(len(names_s)), imps_s, color=colors_fi, zorder=3, edgecolor="white")
ax.set_yticks(range(len(names_s)))
ax.set_yticklabels(names_s, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title("v9 GBM Top-15 Feature Importances\n(psd_max dominates at 77% - root cause of FPs)",
             fontsize=11, fontweight="bold", pad=10)
ax.axvline(0.1, color="#868e96", lw=0.8, ls="--")
patches_fi = [mpatches.Patch(color="#e03131", label="Root-cause feature (psd_max)"),
              mpatches.Patch(color="#1971c2", label="Supporting features")]
ax.legend(handles=patches_fi, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig5_feature_importance.png", dpi=150)
plt.close()
print("Saved fig5_feature_importance.png")


# ---------------------------------------------------------------------------
# 6. Algorithm Evolution Diagram (text-based boxes)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("Algorithm Evolution: v3 to v12", fontsize=14, fontweight="bold", pad=8)

stages = [
    (1.0, 5.5, "v3: Baselines", "#adb5bd",
     "Energy Detector\nSpectral Flatness\nPSD+LogReg(6 feat)\nLB: 0.50"),
    (3.5, 5.5, "v4: LogReg+10", "#74c0fc",
     "+IQ corr, band energy\n+threshold tuning\n(balanced CV=0.72)\nLB: 0.68"),
    (6.0, 5.5, "v5: LogReg+21", "#74c0fc",
     "+spec kurtosis\n+autocorrelation\n+mag statistics\n(balanced CV=0.79)\nLB: 0.79"),
    (8.5, 5.5, "v6: LogReg+30", "#74c0fc",
     "+power variance\n+IQ/Q std, iq_corr\nLoad 5M IQ samples\n(balanced CV=0.92)\nLB: 0.82"),
    (11.0, 5.5, "v7: LogReg+30", "#74c0fc",
     "Same 30 features\nLoad 10M IQ samples\n(balanced CV=0.94)\nLB: 0.82"),
    (1.0, 2.2, "v8: GBM", "#51cf66",
     "Gradient Boosting\n200 trees, depth=3\n30 features, 5M IQ\n(balanced CV=0.93)\nLB: 0.85"),
    (3.5, 2.2, "v9: GBM+Prior", "#51cf66",
     "Prior correction:\nINIT_LOG_ODDS=0\n(train 2:1 -> test 1:1)\n(balanced CV=0.93)\nLB: 0.85"),
    (6.0, 2.2, "v10: Threshold", "#51cf66",
     "Raise threshold\nto 0.6 to cut FP\nTP: 0.38, FP: 0.03\nLB: 0.85"),
    (8.5, 2.2, "v11: Wt.Noise", "#51cf66",
     "Noise weight=3\nduring GBM train\nTP: 0.39, FP: 0.05\nLB: 0.84"),
    (11.0, 2.2, "v12: Ensemble", "#ff6b6b",
     "LogReg(30%)+GBM(70%)\nTP: 0.33, FP: 0.00\n(too conservative)\nLB: 0.83"),
]

for x, y, title, color, body in stages:
    rect = plt.Rectangle((x - 1.1, y - 0.9), 2.2, 1.9, facecolor=color,
                          edgecolor="#495057", linewidth=1.2, alpha=0.85, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y + 0.55, title, ha="center", va="center", fontsize=8.5,
            fontweight="bold", zorder=3, color="#212529")
    ax.text(x, y - 0.15, body, ha="center", va="center", fontsize=7.2,
            zorder=3, color="#212529", linespacing=1.4)

# Arrows row 1
for xs, xe in [(2.1, 2.4), (4.6, 4.9), (7.1, 7.4), (9.6, 9.9)]:
    ax.annotate("", xy=(xe, 5.5), xytext=(xs, 5.5),
                arrowprops=dict(arrowstyle="->", color="#495057", lw=1.3))

# Arrows row 2
for xs, xe in [(2.1, 2.4), (4.6, 4.9), (7.1, 7.4), (9.6, 9.9)]:
    ax.annotate("", xy=(xe, 2.2), xytext=(xs, 2.2),
                arrowprops=dict(arrowstyle="->", color="#495057", lw=1.3))

# Down arrow v7 -> v8
ax.annotate("", xy=(1.0, 3.2), xytext=(11.0, 4.6),
            arrowprops=dict(arrowstyle="->", color="#868e96", lw=1.0, ls="--",
                            connectionstyle="arc3,rad=0.2"))
ax.text(6.0, 4.05, "switch to\nnon-linear model", ha="center", fontsize=8, color="#868e96",
        style="italic")

fig.tight_layout()
fig.savefig(OUT / "fig6_algorithm_evolution.png", dpi=150)
plt.close()
print("Saved fig6_algorithm_evolution.png")


# ---------------------------------------------------------------------------
# 7. FP vs FN tradeoff (illustrating the core challenge)
# ---------------------------------------------------------------------------
# Simulate what happens as we vary threshold
thresholds = np.linspace(0.3, 0.99, 200)

# Approximate logistic model: signals have mean score ~0.9, noise ~0.35
np.random.seed(42)
n = 1000
sig_scores = np.clip(np.random.normal(0.85, 0.12, n), 0, 1)
noise_scores = np.clip(np.random.normal(0.40, 0.18, n), 0, 1)

fp_rates, fn_rates, acc_vals = [], [], []
for thr in thresholds:
    fp = (noise_scores >= thr).mean()
    fn = (sig_scores < thr).mean()
    acc = 0.5 * (1 - fp) + 0.5 * (1 - fn)
    fp_rates.append(fp)
    fn_rates.append(fn)
    acc_vals.append(acc)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax1, ax2 = axes

ax1.plot(thresholds, fp_rates, color="#e03131", lw=2, label="FP rate")
ax1.plot(thresholds, fn_rates, color="#f08c00", lw=2, label="FN rate")
ax1.plot(thresholds, acc_vals, color="#1971c2", lw=2, ls="--", label="Accuracy (balanced)")
ax1.axvline(0.5,  color="#868e96", lw=1, ls=":", label="v9 threshold (0.5)")
ax1.axvline(0.60, color="#51cf66", lw=1, ls=":", label="v10 threshold (0.6)")
ax1.axvline(0.75, color="#ff6b6b", lw=1, ls=":", label="v12 threshold (~0.75)")
ax1.set_xlabel("Threshold", fontsize=11)
ax1.set_ylabel("Rate", fontsize=11)
ax1.set_title("FP/FN vs Threshold (Schematic)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_ylim(-0.02, 1.05)

# ROC-style: FP vs TP
ax2.plot(fp_rates, [1 - f for f in fn_rates], color="#1971c2", lw=2)
ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
# Mark specific operating points
pts = {
    "v9 (0.50)":  (0.5,  None),
    "v10 (0.60)": (0.60, None),
    "v12 (0.75)": (0.75, None),
}
for lbl, (thr_val, _) in pts.items():
    idx = np.argmin(np.abs(thresholds - thr_val))
    ax2.scatter(fp_rates[idx], 1 - fn_rates[idx], s=70, zorder=5)
    ax2.annotate(lbl, (fp_rates[idx], 1 - fn_rates[idx]),
                 textcoords="offset points", xytext=(6, 3), fontsize=8)
ax2.set_xlabel("False Positive Rate", fontsize=11)
ax2.set_ylabel("True Positive Rate (Recall)", fontsize=11)
ax2.set_title("ROC Curve (Schematic)\nHigher threshold = fewer FP but fewer TP", fontsize=11, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "fig7_fp_fn_tradeoff.png", dpi=150)
plt.close()
print("Saved fig7_fp_fn_tradeoff.png")


# ---------------------------------------------------------------------------
# 8. Feature medians: new (v14) vs noise signal separation
# ---------------------------------------------------------------------------
feature_names_new = ["psd_top10_frac", "psd_max_to_p10", "peak_bin_cv"]
noise_medians  = [0.0027, 4.90, 0.157]
signal_medians = [0.0061, 15.88, 0.073]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for i, (ax, name, nm, sm) in enumerate(zip(axes, feature_names_new, noise_medians, signal_medians)):
    ax.bar(["Noise", "Signal"], [nm, sm], color=["#e03131", "#2f9e44"], zorder=3,
           edgecolor="white", width=0.5)
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_ylabel("Median Value")
    ratio = sm / nm if nm > 0 else float("inf")
    ax.text(0.5, max(nm, sm) * 1.05,
            f"Separation ratio: {ratio:.1f}x",
            ha="center", fontsize=9, color="#495057",
            transform=ax.get_xaxis_transform() if False else ax.transData)
    for bar, val in zip(ax.patches, [nm, sm]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(nm, sm) * 0.03,
                f"{val:.4f}" if val < 0.1 else f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

fig.suptitle("New Anti-FP Features: Signal vs Noise Separation (v14)",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig8_new_feature_separation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig8_new_feature_separation.png")


print(f"\nAll figures saved to: {OUT}")
print("Reference them in the report as:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
