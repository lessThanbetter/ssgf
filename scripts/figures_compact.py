"""Summary figures for the compact IPT build, drawn from the verified aggregates.

Outputs (vector PDF, column-width or full-width of the elsarticle preprint):
  fig_claim_v1.pdf    the negative result and its modality reversal
  fig_regime_v1.pdf   where the deficit is, detection and segmentation
  fig_robust_v1.pdf   seeds, the RT-DETR-l repetition, and the index reversal
  fig_decision_v1.pdf legacy schematic; the submitted Figure 11 is
                     make_fig11_decision_v2.py:fig_decision_v2.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, FancyBboxPatch

AGG = r"D:\CSF-main\ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\aggregate"
OUT = r"D:\CSF-main\ssgf\_submission_IPT"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.4,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ORDER = ["spgfusion", "ifcnn", "piafusion", "activity_only", "mean", "probe",
         "tophat", "ensemble", "densefuse", "csf", "probe_wshuffle", "tophat_wshuffle",
         "probe_permuted", "tophat_permuted"]

LABEL = {
    "mean": "mean fusion", "activity_only": "activity-only", "tophat": "top-hat prior",
    "probe": "learned probe", "tophat_permuted": "top-hat, permuted",
    "probe_permuted": "probe, permuted", "tophat_wshuffle": "top-hat, w-shuffled",
    "probe_wshuffle": "probe, w-shuffled", "ensemble": "ensemble", "ifcnn": "IFCNN",
    "densefuse": "DenseFuse", "csf": "classif. saliency", "piafusion": "PIAFusion",
    "spgfusion": "SPGFusion", "ir": "IR source", "vis_gray": "VIS source, grey",
}

COL = {"yolo11n_vis": "#1f4e79", "yolo26n_vis": "#c0504d", "rtdetr_l_vis": "#2e7d32",
       "agnostic": "#7030a0", "agn_split": "#b8860b"}


def load(name):
    return json.load(open(os.path.join(AGG, name), encoding="utf-8"))


def interval_plot(ax, rows, series, xlim=None, legend=True, zero_line=True):
    """rows: list of input keys (bottom to top). series: list of (label, data, color, marker)."""
    y = np.arange(len(rows))
    for i, (label, data, color, marker) in enumerate(series):
        off = (i - (len(series) - 1) / 2.0) * 0.22
        xs = [data[r]["delta50"] for r in rows]
        lo = [data[r]["lo50"] for r in rows]
        hi = [data[r]["hi50"] for r in rows]
        for yy, x, a, b in zip(y + off, xs, lo, hi):
            ax.plot([a, b], [yy, yy], color=color, lw=1.0, solid_capstyle="butt",
                    zorder=2)
        ax.plot(xs, y + off, marker, color=color, ms=3.1, lw=0, label=label, zorder=3)
    if zero_line:
        ax.axvline(0.0, color="0.25", lw=0.7, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(r, r) for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    if xlim:
        ax.set_xlim(*xlim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        ax.legend(frameon=False, loc="lower left", handletextpad=0.3, borderpad=0.1)


def fig_claim():
    y11 = load("contrasts_yolo11n_colour.json")
    y26 = load("contrasts_yolo26n_colour.json")
    agn = load("contrasts_agnostic_colour.json")
    split = load("contrasts_agn_split_colour.json")
    rt3 = load("rtdetr_seeds.json")["per_seed"]
    rt = {k: {"delta50": v["mean_over_seeds"], "lo50": v["lo"], "hi50": v["hi"]}
          for k, v in rt3.items()}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    interval_plot(axes[0], ORDER,
                  [("YOLO11n", y11, COL["yolo11n_vis"], "o"),
                   ("YOLO26n", y26, COL["yolo26n_vis"], "s"),
                   ("RT-DETR-l, mean of 3 runs", rt, COL["rtdetr_l_vis"], "^")],
                  xlim=(-0.24, 0.06))
    axes[0].set_xlabel(r"$\Delta$ mAP50 against the colour-scored visible source")
    axes[0].set_title("(a) visible-trained detectors: 42 of 42 below", fontsize=7.4)

    interval_plot(axes[1], ORDER,
                  [("modality-agnostic", agn, COL["agnostic"], "o"),
                   ("the same, clean split", split, COL["agn_split"], "D")],
                  xlim=(-0.12, 0.12), legend=False)
    axes[1].set_xlabel(r"$\Delta$ mAP50 against the colour-scored visible source")
    axes[1].set_title("(b) modality-agnostic: the reversal", fontsize=7.4)
    axes[1].legend(frameon=False, loc="upper left", handletextpad=0.3, borderpad=0.1)

    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_claim_v1.pdf"))
    plt.close(fig)


def fig_regime():
    reg = load("regime_conditional.json")
    seg = load("seg_regime_summary.json")
    curve = load("regime_curve.json")

    keys = [k for k in reg["yolo11n_vis"] if k != "all 300 images"]
    want = ["low visible contrast (bottom tercile)", "high visible contrast (top tercile)",
            "small targets (below median area)"]
    want = [k for k in want if k in keys] or keys[:3]

    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 0.95, 0.95], wspace=0.5)
    ax0 = fig.add_subplot(gs[0])

    dets = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis"]
    cols = ["all 300 images", "low visible contrast (bottom tercile)",
            "high visible contrast (top tercile)", "three or more pedestrians",
            "small targets (below median area)"]
    cols = [c for c in cols if c in reg["yolo11n_vis"]]
    col_labels = ["all 300\nimages", "bottom\ntercile", "top\ntercile",
                  "three or more\npedestrians", "small\ntargets"]
    ax0.set_xlim(-0.5, len(cols) - 0.5)
    ax0.set_ylim(len(dets) - 0.5, -0.5)
    ax0.set_xticks(range(len(cols)))
    ax0.set_xticklabels(col_labels[:len(cols)], fontsize=6.0)
    ax0.set_yticks(range(len(dets)))
    ax0.set_yticklabels(["YOLO11n", "YOLO26n", "RT-DETR-l"], fontsize=6.6)
    ax0.tick_params(length=0)
    for ri, det in enumerate(dets):
        for ci, key in enumerate(cols):
            cell = reg[det][key]
            b, u, a_ = cell["below_resolved"], cell["unresolved"], cell["above_resolved"]
            if a_:
                fc = "#f2c14e"
            elif u >= 10:
                fc = "0.88"
            elif u:
                fc = "#cfe0c3"
            else:
                fc = "#9dc48a"
            ax0.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fc=fc, ec="white", lw=1.2))
            ax0.text(ci, ri, "%d / %d / %d" % (b, u, a_), ha="center", va="center",
                     fontsize=6.2)
    ax0.set_title("(a) detection: below / unresolved / above, of 14", fontsize=7.0)
    for s in ("top", "right", "bottom", "left"):
        ax0.spines[s].set_visible(False)
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)

    ax1 = fig.add_subplot(gs[1])
    bins = curve["yolo11n_vis"]
    vals = np.array([b["inputs"]["spgfusion"]["delta"] for b in bins])
    los = np.array([b["inputs"]["spgfusion"]["lo"] for b in bins])
    his = np.array([b["inputs"]["spgfusion"]["hi"] for b in bins])
    xs = np.arange(1, len(vals) + 1)
    ax1.bar(xs, vals, color="#4f6228", width=0.62)
    ax1.errorbar(xs, vals, yerr=[vals - los, his - vals], fmt="none", ecolor="0.3",
                 elinewidth=0.7, capsize=1.6)
    ax1.axhline(0, color="0.25", lw=0.6)
    ax1.set_xlabel("visible-contrast bin")
    ax1.set_ylabel("SPGFusion\n" + r"$\Delta$ mAP50")
    ax1.set_title("(b) along the contrast axis", fontsize=7.4)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[2])
    groups = seg["groups"]
    names = ["low visible contrast (bottom tercile)", "middle tercile",
             "high visible contrast (top tercile)"]
    short = ["bottom\ntercile", "middle\ntercile", "top\ntercile"]
    deltas = [groups[n]["inputs"]["mean"]["delta"] for n in names]
    los = [groups[n]["inputs"]["mean"]["lo"] for n in names]
    his = [groups[n]["inputs"]["mean"]["hi"] for n in names]
    y = np.arange(len(names))
    for yy, d, lo, hi in zip(y, deltas, los, his):
        ax2.plot([lo, hi], [yy, yy], color="#7030a0", lw=1.0)
    ax2.plot(deltas, y, "o", color="#7030a0", ms=3.2)
    ax2.axvline(0, color="0.25", lw=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(short, fontsize=6.4)
    ax2.set_xlabel(r"$\Delta$ mIoU")
    ax2.set_title("(c) segmentation, mean fusion", fontsize=7.4)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.savefig(os.path.join(OUT, "fig_regime_v1.pdf"))
    plt.close(fig)


def fig_robust():
    seeds = load("clean_split_seeds.json")["per_seed"]
    rt = load("rtdetr_seeds.json")["per_seed"]
    vif = load("vif_decomp.json")

    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.5)

    ax0 = fig.add_subplot(gs[0])
    rows = [r for r in ORDER if r in seeds]
    y = np.arange(len(rows))
    for yy, r in zip(y, rows):
        cell = seeds[r]
        ax0.plot([cell["lo"], cell["hi"]], [yy, yy], color="#b8860b", lw=1.0)
    ax0.plot([seeds[r]["mean_over_seeds"] for r in rows], y, "o", color="#b8860b", ms=3.2)
    ax0.axvline(0, color="0.25", lw=0.7)
    ax0.set_yticks(y)
    ax0.set_yticklabels([LABEL[r] for r in rows], fontsize=6.4)
    ax0.set_xlabel(r"$\Delta$ mAP50, four seeds")
    ax0.set_title("(a) clean split, mean of 4 seeds", fontsize=7.2)
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)

    ax1 = fig.add_subplot(gs[1])
    rows1 = [r for r in ORDER if r in rt][:6]
    y = np.arange(len(rows1))
    for yy, r in zip(y, rows1):
        ax1.plot([rt[r]["lo"], rt[r]["hi"]], [yy, yy], color="0.55", lw=0.9)
        ax1.plot([rt[r]["lo"], rt[r]["hi"]], [yy, yy], "|", color="0.45", ms=4)
    ax1.plot([rt[r]["mean_over_seeds"] for r in rows1], y, "o", color="#2e7d32", ms=3.4,
             label="mean of 3 runs")
    ax1.axvline(0, color="0.25", lw=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels([LABEL[r] for r in rows1], fontsize=6.4)
    ax1.set_xlabel(r"$\Delta$ mAP50, RT-DETR-l")
    ax1.set_title("(b) RT-DETR-l, 3 runs", fontsize=7.2)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[2])
    rows2 = [r["variant"] for r in vif["rows"]]
    cons = ["reported", "width_only", "decimate_only", "noise_only", "reference"]
    nice = ["reported", "width only", "decimation only", "noise only", "reference VIF"]
    for ci, (c, nm) in enumerate(zip(cons, nice)):
        vals = [r[c] for r in vif["rows"]]
        ax2.plot(vals, np.arange(len(rows2)), "o-", ms=2.6, lw=0.8,
                 color=["#1f4e79", "#8faadc", "#2e75b6", "#c0504d", "#7030a0"][ci],
                 label=nm)
    ax2.set_yticks(np.arange(len(rows2)))
    ax2.set_yticklabels([LABEL.get(v, v) for v in rows2], fontsize=6.4)
    ax2.set_xlabel("fidelity index value (VIF)")
    ax2.set_title("(c) the index changes the ranking", fontsize=7.2)
    ax2.legend(frameon=False, fontsize=6.0, loc="lower right", handletextpad=0.3)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.savefig(os.path.join(OUT, "fig_robust_v1.pdf"))
    plt.close(fig)


def fig_decision():
    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="0.35"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.0,
                linespacing=1.30)

    box(3.1, 8.9, 3.8, 0.95, "same fourteen\nfused images", "#eaeaea")
    box(0.0, 5.9, 4.7, 2.5,
        "evaluator\nvisible-trained:\nall 14 rules lose\n0.023 to 0.152\n"
        "modality-agnostic:\nten of 14 gain\n0.023 to 0.040", "#dbe5f1")
    box(5.3, 5.9, 4.7, 2.5,
        "fidelity index\nreported VIF:\nSPGFusion first\nof eight\n"
        "reference VIF:\nSPGFusion last\nof eight", "#e8dfec")
    box(2.3, 2.9, 5.4, 2.0,
        "seed and split\ndeficits survive\nretraining; after a\nclean split the gain\n"
        "shrinks to 0.022-0.028", "#e2efda")
    box(1.2, 0.3, 7.6, 1.6,
        "pooled over 300 images: no fused rule\nbeats the unfused visible image for a\n"
        "visible-trained detector; the best-aligned\nquartile differs", "#fff2cc")

    for x0, y0, x1, y1 in [(5.0, 8.9, 2.4, 8.4), (5.0, 8.9, 7.6, 8.4),
                           (2.4, 5.9, 3.8, 4.9), (7.6, 5.9, 6.2, 4.9),
                           (5.0, 2.9, 5.0, 1.9)]:
        ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.05,
                                head_width=0.28, head_length=0.3, length_includes_head=True,
                                fc="0.35", ec="0.35"))
    fig.savefig(os.path.join(OUT, "fig_decision_v1.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_claim()
    fig_regime()
    fig_robust()
    fig_decision()
    print("figures written to", OUT)
