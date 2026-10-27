"""Redraw every figure after Fig. 2 in the style of the supplied reference manuscript.

Same data and the same file names as before, so the manuscript source does not change.
Style: Times-family serif, full box frame with a light dotted grid, outward ticks,
bordered legends, bold panel labels inside the axes, muted palette.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter

import _figstyle as fs

AGG = r"D:\CSF-main\ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\aggregate"
OUT = r"D:\CSF-main\ssgf\_submission_IPT"
FID = r"D:\CSF-main\ssgf\_remote_jobs\results\j1_fidelity.json"
IOU = np.round(np.linspace(0.5, 0.95, 10), 2)

C = fs.C
EV = [("yolo11n_vis", "yolo11n"), ("yolo26n_vis", "yolo26n"),
      ("rtdetr_l_vis", "rtdetr_l"), ("agnostic", "agnostic")]
EV_LBL = {"yolo11n_vis": "YOLO11n", "yolo26n_vis": "YOLO26n",
          "rtdetr_l_vis": "RT-DETR-l", "agnostic": "agnostic"}

ORDER = ["spgfusion", "ifcnn", "piafusion", "activity_only", "mean", "probe",
         "tophat", "ensemble", "densefuse", "csf", "probe_wshuffle", "tophat_wshuffle",
         "probe_permuted", "tophat_permuted"]


def load(p):
    return json.load(open(p, encoding="utf-8"))


def save(fig, name):
    """Write both the vector PDF used by the manuscript and an SVG for editing."""
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".svg"))


def ytick_labels(rows):
    return [fs.LABEL.get(r, r) for r in rows]


# ---------------------------------------------------------------- Fig. 3
def fig_claim():
    """Fig. 2: clean-split evidence first, then the optimistic protocol audit."""
    ranges = load(r"D:\CSF-main\ssgf\_clean_split_seed_ranges.json")
    y11 = load(os.path.join(AGG, "contrasts_yolo11n_colour.json"))
    y26 = load(os.path.join(AGG, "contrasts_yolo26n_colour.json"))
    rt3 = load(os.path.join(AGG, "rtdetr_seeds.json"))["per_seed"]
    rt = {k: {"delta50": v["mean_over_seeds"], "lo50": v["lo"], "hi50": v["hi"]}
          for k, v in rt3.items()}

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 4.15), sharey=True,
                             gridspec_kw={"wspace": 0.10,
                                          "width_ratios": [1.0, 1.0, 1.18]})
    y = np.arange(len(ORDER))

    ax = axes[0]
    for k in range(1, len(ORDER), 2):
        ax.axhspan(k - 0.5, k + 0.5, color="#f4f4f4", zorder=0)
    vals = ranges["vis_split"]["inputs"]
    x = [vals[r]["mean_contrast"] for r in ORDER]
    lo = [vals[r]["min_contrast"] for r in ORDER]
    hi = [vals[r]["max_contrast"] for r in ORDER]
    ax.errorbar(x, y, xerr=[np.array(x) - np.array(lo), np.array(hi) - np.array(x)],
                fmt="o", color=C["teal"], ecolor=C["teal"], elinewidth=1.0,
                capsize=2.2, capthick=1.0, ms=4.2, lw=0, zorder=3,
                label="clean VIS")
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlim(-0.20, 0.02)
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels(ORDER), fontsize=8.4)
    ours = {"mean", "activity_only", "tophat", "probe", "ensemble"}
    published = {"ifcnn", "densefuse", "csf", "piafusion", "spgfusion"}
    for tick, key in zip(ax.get_yticklabels(), ORDER):
        if key in ours:
            tick.set_color(C["blue"])
        elif key in published:
            tick.set_color(C["red"])
        else:
            tick.set_color(C["grey"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.65, len(ORDER) - 0.35)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS")
    ax.set_title("clean visible-only", fontsize=8.8, pad=5)
    fs.tidy(ax, grid="x")
    fs.panel(ax, "a")

    ax = axes[1]
    for k in range(1, len(ORDER), 2):
        ax.axhspan(k - 0.5, k + 0.5, color="#f4f4f4", zorder=0)
    vals = ranges["agn_split"]["inputs"]
    x = [vals[r]["mean_contrast"] for r in ORDER]
    lo = [vals[r]["min_contrast"] for r in ORDER]
    hi = [vals[r]["max_contrast"] for r in ORDER]
    ax.errorbar(x, y, xerr=[np.array(x) - np.array(lo), np.array(hi) - np.array(x)],
                fmt="D", color=C["orange"], ecolor=C["orange"], elinewidth=1.0,
                capsize=2.2, capthick=1.0, ms=3.9, lw=0, zorder=3,
                label="clean agnostic")
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlim(-0.08, 0.08)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS")
    ax.set_title("clean modality-agnostic", fontsize=8.8, pad=5)
    fs.tidy(ax, grid="x")
    fs.panel(ax, "b")

    ax = axes[2]
    for k in range(1, len(ORDER), 2):
        ax.axhspan(k - 0.5, k + 0.5, color="#f4f4f4", zorder=0)
    series = [("YOLO11n", y11, fs.DET_COLOR["yolo11n_vis"], "o"),
              ("YOLO26n", y26, fs.DET_COLOR["yolo26n_vis"], "s"),
              ("RT-DETR-l (3 runs)", rt, fs.DET_COLOR["rtdetr_l_vis"], "^")]
    for i, (nice, data, col, mk) in enumerate(series):
        off = (i - 1) * 0.22
        lo = [data[r]["lo50"] for r in ORDER]
        hi = [data[r]["hi50"] for r in ORDER]
        x = [data[r]["delta50"] for r in ORDER]
        ax.errorbar(x, y + off, xerr=[np.array(x) - np.array(lo),
                                      np.array(hi) - np.array(x)],
                    fmt=mk, color=col, ecolor=col, elinewidth=1.0, capsize=2.2,
                    capthick=1.0, ms=4.2, lw=0, label=nice, zorder=3)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlim(-0.25, 0.15)
    ax.set_xlabel(r"$\Delta$ mAP50 vs VIS")
    ax.set_title("pooled audit (optimistic)", fontsize=8.8, pad=5)
    fs.tidy(ax, grid="x")
    fs.panel(ax, "c")
    handles, labels = axes[0].get_legend_handles_labels()
    for axis in axes[1:]:
        h, lab = axis.get_legend_handles_labels()
        handles += h
        labels += lab
    fig.subplots_adjust(bottom=0.24)
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, frameon=True, edgecolor="black", fancybox=False,
               framealpha=1.0, fontsize=7.8, handlelength=1.4, handletextpad=0.45,
               columnspacing=1.1, borderpad=0.30)
    leg.set_title("point: mean; bar: range over four runs (a,b) or 95% CI (c)",
                  prop={"size": 7.2, "style": "italic"})
    save(fig, "fig_claim_v1")
    plt.close(fig)


def fig_regime():
    reg = load(os.path.join(AGG, "regime_conditional.json"))
    seg = load(os.path.join(AGG, "seg_regime_summary.json"))
    curve = load(os.path.join(AGG, "regime_curve.json"))

    dets = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis"]
    cols = ["all 300 images", "low visible contrast (bottom tercile)",
            "high visible contrast (top tercile)", "three or more pedestrians",
            "small targets (below median area)"]
    col_labels = ["all 300\nimages", "bottom\ntercile", "top\ntercile",
                  "$\\geq$3\npedestrians", "small\ntargets"]

    fig = plt.figure(figsize=(6.9, 4.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.72, wspace=0.34)

    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(dets) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=8.4)
    ax.set_yticks(range(len(dets)))
    ax.set_yticklabels([EV_LBL[d] for d in dets], fontsize=8.6)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", pad=1.5)
    for ri, det in enumerate(dets):
        for ci, key in enumerate(cols):
            cell = reg[det][key]
            b, u, a_ = cell["below_resolved"], cell["unresolved"], cell["above_resolved"]
            if a_:
                fc = "#f0b357"          # one input is resolved above the source
            elif b == 0:
                fc = C["lightgrey"]      # none resolved
            elif u:
                fc = "#cfe3c2"           # some resolved, some not
            else:
                fc = "#6f9e5a"           # all fourteen resolved below
            ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fc=fc,
                                       ec="white", lw=1.6))
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#6f9e5a", ec="none"),
               plt.Rectangle((0, 0), 1, 1, fc="#cfe3c2", ec="none"),
               plt.Rectangle((0, 0), 1, 1, fc="#f0b357", ec="none")]
    fs.legend(ax, handles=handles,
              labels=["all below", "some below", "one above"],
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              fontsize=7.8, columnspacing=1.2, handlelength=1.2)
    fs.panel(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    bins = curve["yolo11n_vis"]
    vals = np.array([b["inputs"]["spgfusion"]["delta"] for b in bins])
    los = np.array([b["inputs"]["spgfusion"]["lo"] for b in bins])
    his = np.array([b["inputs"]["spgfusion"]["hi"] for b in bins])
    xs = np.arange(1, len(vals) + 1)
    ax.bar(xs, vals, width=0.62, color=C["green"], edgecolor="black", lw=0.5,
           label="SPGFusion")
    ax.errorbar(xs, vals, yerr=[vals - los, his - vals], fmt="none", ecolor="black",
                elinewidth=0.8, capsize=2.2)
    ax.axhline(0, color="black", lw=0.8)
    fs.tidy(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%.3f\u2013\n%.3f" % (b["contrast_lo"], b["contrast_hi"])
                        for b in bins], fontsize=7.8)
    ax.set_xlabel("visible-contrast bin (RMS contrast range)")
    ax.set_ylabel(r"$\Delta$ mAP50")
    ax.set_ylim(-0.42, 0.14)
    fs.panel(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    names = ["low visible contrast (bottom tercile)", "middle tercile",
             "high visible contrast (top tercile)"]
    short = ["bottom\ntercile", "middle\ntercile", "top\ntercile"]
    d = [seg["groups"][n]["inputs"]["mean"]["delta"] for n in names]
    lo = [seg["groups"][n]["inputs"]["mean"]["lo"] for n in names]
    hi = [seg["groups"][n]["inputs"]["mean"]["hi"] for n in names]
    y = np.arange(len(names))
    fs.interval(ax, y, d, lo, hi, C["purple"], ms=4.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=8.4)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel(r"$\Delta$ mIoU")
    fs.tidy(ax, grid="x")
    fs.panel(ax, "c")
    save(fig, "fig_regime_v1")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4
def fig_robust():
    seeds = load(os.path.join(AGG, "clean_split_seeds.json"))["per_seed"]
    rt = load(os.path.join(AGG, "rtdetr_seeds.json"))["per_seed"]
    vif = load(os.path.join(AGG, "vif_decomp.json"))

    fig = plt.figure(figsize=(6.9, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.55, wspace=0.45)

    ax = fig.add_subplot(gs[0, :])
    rows = [r for r in ORDER if r in seeds]
    y = np.arange(len(rows))
    fs.interval(ax, y, [seeds[r]["mean_over_seeds"] for r in rows],
                [seeds[r]["lo"] for r in rows], [seeds[r]["hi"] for r in rows],
                C["orange"], ms=4.2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels(rows), fontsize=8.4)
    ax.set_xlabel(r"$\Delta$ mAP50")
    fs.tidy(ax, grid="x")
    fs.panel(ax, "a")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)

    ax = fig.add_subplot(gs[1, 0])
    rows1 = [r for r in ORDER if r in rt][:6]
    y = np.arange(len(rows1))
    fs.interval(ax, y, [rt[r]["mean_over_seeds"] for r in rows1],
                [rt[r]["lo"] for r in rows1], [rt[r]["hi"] for r in rows1],
                C["green"], ms=4.2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels(rows1), fontsize=8.4)
    ax.set_xlabel(r"$\Delta$ mAP50")
    fs.tidy(ax, grid="x")
    fs.panel(ax, "b")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows1) - 0.4)

    ax = fig.add_subplot(gs[1, 1])
    rows2 = [r["variant"] for r in vif["rows"]]
    cons = [("reported", "reported index", C["blue"], "o"),
            ("noise_only", "noise calibration only", C["red"], "v"),
            ("reference", "reference VIF", C["purple"], "D")]
    y = np.arange(len(rows2))
    for key, nice, col, mk in cons:
        ax.plot([r[key] for r in vif["rows"]], y, mk + "-", color=col, ms=4.0,
                lw=1.0, label=nice)
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels(rows2), fontsize=8.4)
    ax.set_xlabel("fidelity index value (VIF)")
    fs.tidy(ax, grid="x")
    fs.legend(ax, loc="lower right", fontsize=7.8)
    fs.panel(ax, "c")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows2) - 0.4)
    ax.set_xlim(0.15, 0.95)
    save(fig, "fig_robust_v1")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 5
def fig_decision():
    """The argument as one schematic: four short statements, numbers in the caption."""
    fig, ax = plt.subplots(figsize=(4.7, 5.2))
    # The rounded boxes carry a pad of 0.10 canvas units beyond their rectangle, so the
    # limits are set with a margin and the patches are drawn unclipped: with the exact
    # 0..10 limits the outer boxes were cut by the axes.
    ax.set_xlim(-0.9, 10.9)
    ax.set_ylim(-0.7, 13.9)
    ax.axis("off")
    boxes = []

    def box(x, y, w, h, text, fc, fs_=8.6, lw=0.9, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                    fc=fc, ec="black", lw=lw, clip_on=False))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs_, linespacing=1.45, fontweight=weight)
        boxes.append((x - 0.10, y - 0.10, x + w + 0.10, y + h + 0.10))

    box(3.1, 11.70, 3.8, 1.05, "same fused images", C["fillgrey"], fs_=8.6)
    box(0.20, 7.70, 4.70, 2.90,
        "evaluator\nvisible-trained: all lose $\\downarrow$\n"
        "clean split: four gain $\\uparrow$\npooled: ten gain $\\uparrow$",
        C["fillblue"], fs_=7.2)
    box(5.20, 7.70, 4.70, 2.90,
        "fidelity index\nreported: SPGFusion first\nreference: SPGFusion last $\\downarrow$",
        C["fillpurple"], fs_=7.2)
    box(2.15, 3.75, 5.70, 2.30,
        "seed and split\nthe deficit survives retraining;\nthe clean-split gain shrinks",
        C["fillgreen"], fs_=7.8)
    box(0.35, 0.35, 9.30, 2.05,
        "the ranking is a property of the scorer\n"
        "pooled: no fused rule beats the visible image\n"
        "for a visible-trained detector\n"
        "(the best-aligned quartile differs)",
        C["fillsand"], fs_=7.2, lw=1.6, weight="bold")

    arrows = [(5.0, 11.70, 2.55, 10.75), (5.0, 11.70, 7.45, 10.75),
              (2.55, 7.70, 3.65, 6.20), (7.45, 7.70, 6.35, 6.20),
              (5.0, 3.75, 5.0, 2.55)]
    for x0, y0, x1, y1 in arrows:
        ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.045,
                                head_width=0.26, head_length=0.28,
                                length_includes_head=True, fc="#404040", ec="#404040"))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bad = [b for b in boxes if b[0] < x0 or b[1] < y0 or b[2] > x1 or b[3] > y1]
    print("fig_decision: boxes outside the canvas:", len(bad))
    save(fig, "fig_decision_v1")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 6
def fig_corr(extra):
    corr = extra["rank_correlation"]
    fid = load(FID)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.32})
    ax = axes[0]
    for det, fname in EV:
        c = load(os.path.join(AGG, "contrasts_%s_colour.json" % fname))
        rows = [r for r in c if r in fid and r != "vis"]
        xs = np.array([fid[r]["VIF_vis"] for r in rows])
        ys = np.array([c[r]["delta50"] for r in rows])
        ax.plot(xs, ys, fs.DET_MARKER[det], color=fs.DET_COLOR[det], ms=4.6, lw=0,
                label="%s ($\\rho$ %+.2f)" % (EV_LBL[det], corr[det]["VIF"]))
        if len(xs) > 2:
            fit = np.polyfit(xs, ys, 1)
            grid = np.linspace(xs.min(), xs.max(), 20)
            ax.plot(grid, np.polyval(fit, grid), "--", color=fs.DET_COLOR[det],
                    lw=0.9, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlim(0.24, 0.90)
    ax.set_xlabel("reported-index VIF of the fused image")
    ax.set_ylabel(r"$\Delta$ mAP50 vs the colour-scored source")
    fs.tidy(ax, grid="both")
    fs.panel(ax, "a")

    ax = axes[1]
    names = ["VIF", "SSIM_vis", "EN"]
    nice = ["VIF", "SSIM$_{vis}$", "EN"]
    w = 0.2
    for i, (det, _f) in enumerate(EV):
        ax.bar(np.arange(len(names)) + (i - 1.5) * w, [corr[det][k] for k in names],
               w, color=fs.DET_COLOR[det], edgecolor="black", lw=0.5, label=EV_LBL[det])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(nice)
    ax.set_ylabel(r"Spearman $\rho$ with the contrast")
    ax.set_ylim(-0.8, 0.8)
    fs.tidy(ax)
    fs.panel(ax, "b")
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    seen = list(labels)
    for h, l in zip(h2, l2):
        if l not in seen:
            handles.append(h)
            labels.append(l)
            seen.append(l)
    fig.subplots_adjust(bottom=0.34)
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                     ncol=4, frameon=True, edgecolor="black", fancybox=False,
                     framealpha=1.0, fontsize=7.8, handlelength=1.4,
                     handletextpad=0.45, columnspacing=1.1, borderpad=0.30)
    leg.set_title("evaluator of the downstream task", prop={"size": 7.2, "style": "italic"})
    save(fig, "fig_corr_v1")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 7
def fig_criterion(extra):
    crit = extra["criterion"]
    show = [("mean", C["green"], "o"), ("probe", C["blue"], "s"),
            ("ifcnn", C["red"], "^"), ("spgfusion", C["purple"], "D")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    for ax, det in zip(axes, ("yolo11n_vis", "agnostic")):
        for name, col, mk in show:
            if name not in crit[det]["contrasts"]:
                continue
            ax.plot(IOU, crit[det]["contrasts"][name], mk + "-", color=col, ms=4.2,
                    lw=1.1, label=fs.LABEL[name])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("IoU criterion")
        fs.tidy(ax, grid="both")
    axes[0].set_ylabel(r"$\Delta$ mAP vs the visible source")
    fs.panel(axes[0], "a", text=EV_LBL["yolo11n_vis"])
    fs.panel(axes[1], "b", text=EV_LBL["agnostic"])
    fig.subplots_adjust(bottom=0.30)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=True, edgecolor="black",
               fancybox=False, framealpha=1.0, fontsize=7.8, handlelength=1.4,
               handletextpad=0.45, columnspacing=1.1, borderpad=0.30)
    save(fig, "fig_criterion_v1")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 8
def fig_operating(extra):
    op = extra["operating_point"]
    rows = ["vis", "mean", "probe", "ifcnn", "spgfusion"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1),
                             gridspec_kw={"wspace": 0.28})
    w = 0.2
    for ax, key, letter, ylab in ((axes[0], "recall", "a", "recall"),
                                  (axes[1], "precision", "b", "precision")):
        for i, (det, _f) in enumerate(EV):
            ax.bar(np.arange(len(rows)) + (i - 1.5) * w,
                   [op[det][r][key] for r in rows], w, color=fs.DET_COLOR[det],
                   edgecolor="black", lw=0.5, label=EV_LBL[det])
            ax.text(len(rows) - 1 + (i - 1.5) * w, op[det]["spgfusion"][key] + 0.015,
                    "%.2f" % op[det]["spgfusion"][key], ha="center", va="bottom",
                    fontsize=7.8, rotation=90)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([fs.LABEL[r] for r in rows], rotation=18, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0.0, 1.0)
        fs.tidy(ax)
        fs.panel(ax, letter)
    fs.legend(axes[0], loc="lower center", bbox_to_anchor=(1.15, -0.42), ncol=4,
              fontsize=7.8, columnspacing=1.1)
    save(fig, "fig_operating_v1")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 9
def fig_tau(tau):
    rows = tau["rows"]
    styles = {"activity_only": ("activity-only", C["red"], "o"),
              "tophat": ("top-hat prior", C["blue"], "s"),
              "probe": ("learned probe", C["green"], "^")}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0),
                             gridspec_kw={"wspace": 0.28})
    for ax, ev, letter in ((axes[0], "vis", "a"), (axes[1], "agn", "b")):
        for rule, (lab, col, mk) in styles.items():
            xs = [r["tau"] for r in rows if r["evaluator"] == ev and r["rule"] == rule]
            ys = [r["mAP50"] for r in rows if r["evaluator"] == ev and r["rule"] == rule]
            ax.plot(xs, ys, mk + "-", color=col, ms=4.2, lw=1.1, label=lab)
        ax.axhline(tau["mean_fusion"][ev], color="black", lw=0.9, ls="--")
        ax.text(3.05, tau["mean_fusion"][ev], "mean fusion", fontsize=7.8,
                va="center", ha="left")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1.0, 3.0])
        ax.set_xticklabels(["0.1", "0.3", "1", "3"])
        ax.set_xlim(0.08, 4.6)
        ax.set_xlabel(r"temperature $\tau$ (log scale)")
        fs.tidy(ax, grid="both")
        fs.panel(ax, letter)
    axes[0].set_ylabel("mAP50")
    fig.subplots_adjust(bottom=0.30)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=True, edgecolor="black",
               fancybox=False, framealpha=1.0, fontsize=7.8, handlelength=1.4,
               handletextpad=0.45, columnspacing=1.2, borderpad=0.30)
    save(fig, "fig_tau_v1")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 10
def fig_scale(y11s, res):
    base = load(os.path.join(AGG, "contrasts_yolo11n_colour.json"))
    order = ["spgfusion", "ifcnn", "piafusion", "csf", "activity_only", "mean",
             "probe_wshuffle", "tophat_wshuffle", "probe_permuted", "tophat_permuted",
             "tophat", "densefuse", "ensemble", "probe", "ir"]
    rows = []
    for r in order:
        if r in y11s["rows"] and r not in rows:
            rows.append(r)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.25), sharey=False,
                             gridspec_kw={"wspace": 0.12})
    y = np.arange(len(rows))

    ax = axes[0]
    fs.interval(ax, y, [y11s["rows"][r]["delta50"] for r in rows],
                [y11s["rows"][r]["lo50"] for r in rows],
                [y11s["rows"][r]["hi50"] for r in rows], C["blue"], ms=4.2,
                label="YOLO11s, mean of runs (95% CI)")
    ax.plot([base[r]["delta50"] if r in base else np.nan for r in rows], y, "s",
            color=C["red"], ms=4.2, lw=0, label="YOLO11n, reported run")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels(rows), fontsize=8.4)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.27, 0.19)
    ax.set_xlabel(r"$\Delta$ mAP50 vs the colour-scored source")
    fs.tidy(ax, grid="x")
    fig.subplots_adjust(bottom=0.30)
    fs.legend(ax, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              fontsize=7.8, columnspacing=1.2)
    fs.panel(ax, "a")

    ax = axes[1]
    series = [("i512", "imgsz 512", C["green"], "o"),
              ("i640", "imgsz 640", C["blue"], "s"),
              ("i960", "imgsz 960", C["red"], "^")]
    rows2 = [r for r in rows if r in res["i640"]["rows"]]
    y2 = np.arange(len(rows2))
    for key, nice, col, mk in series:
        ax.plot([res[key]["rows"][r]["delta50"] for r in rows2], y2, mk, color=col,
                ms=4.2, lw=0, label=nice)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y2)
    ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_ylim(-0.6, len(rows2) - 0.4)
    ax.set_xlim(-0.27, 0.19)
    ax.set_xlabel(r"$\Delta$ mAP50 vs the colour-scored source")
    fs.tidy(ax, grid="x")
    fs.legend(ax, loc="upper right", fontsize=7.8)
    fs.panel(ax, "b")
    save(fig, "fig_scale_v1")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 11
def fig_corrupt(corr):
    """Corruption robustness.

    The additive-noise condition moves the contrast by +0.24 to +0.30 while the other
    three conditions stay between -0.15 and +0.01, so the three are drawn on their own
    scale and the noise condition in a separate panel of its own scale; the caption
    says so. Rules are on the x axis and conditions are the series, which keeps the
    legend to three entries and off the bars.
    """
    rules = [("mean", "mean fusion"), ("probe", "learned probe"),
             ("ifcnn", "IFCNN"), ("spgfusion", "SPGFusion")]
    conds = [("yolo11n_vis_clean", "clean", C["green"]),
             ("yolo11n_vis_jpeg30", "JPEG-30", C["blue"]),
             ("yolo11n_vis_blur", "blur 1.5 px", C["orange"])]
    order = ["yolo11n_vis_clean", "yolo11n_vis_jpeg30", "yolo11n_vis_blur",
             "yolo11n_vis_noise"]

    fig = plt.figure(figsize=(6.9, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.72, 1.02], wspace=0.55)

    ax = fig.add_subplot(gs[0])
    w = 0.26
    y = np.arange(len(rules))
    for i, (key, nice, col) in enumerate(conds):
        vals = [corr[key]["rows"][r]["delta"] for r, _ in rules]
        ax.barh(y + (i - 1) * w, vals, w, color=col, edgecolor="black", lw=0.5,
                label=nice)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rules], fontsize=8.4)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.62, len(rules) - 0.38)
    ax.set_xlim(-0.19, 0.045)
    ax.set_xlabel(r"$\Delta$ mAP50 vs the degraded source")
    fs.tidy(ax, grid="x")
    fs.legend(ax, loc="lower left", fontsize=7.8)
    fs.panel(ax, "a")

    ax = fig.add_subplot(gs[1])
    vals = [corr["yolo11n_vis_noise"]["rows"][r]["delta"] for r, _ in rules]
    los = [corr["yolo11n_vis_noise"]["rows"][r]["lo"] for r, _ in rules]
    his = [corr["yolo11n_vis_noise"]["rows"][r]["hi"] for r, _ in rules]
    ax.barh(y, vals, 0.55, color=C["red"], edgecolor="black", lw=0.5)
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(los),
                               np.array(his) - np.array(vals)],
                fmt="none", ecolor="black", elinewidth=0.8, capsize=2.2)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.62, len(rules) - 0.38)
    ax.set_xlim(0.0, 0.42)
    ax.set_xlabel(r"$\Delta$ mAP50")
    fs.tidy(ax, grid="x")
    fs.panel(ax, "b", text="(own scale)")

    ax = fig.add_subplot(gs[2])
    counts = []
    names = ["clean", "JPEG-30", "blur 1.5 px", "noise"]
    for k in order:
        rows = corr[k]["rows"]
        below = sum(1 for r in rows.values() if r["hi"] < 0)
        above = sum(1 for r in rows.values() if r["lo"] > 0)
        counts.append((below, len(rows) - below - above, above))
    counts = np.array(counts)
    y = np.arange(len(order))
    ax.barh(y, counts[:, 0], color="#8fbf7f", edgecolor="black", lw=0.5,
            label="below")
    ax.barh(y, counts[:, 1], left=counts[:, 0], color=C["lightgrey"],
            edgecolor="black", lw=0.5, label="unresolved")
    ax.barh(y, counts[:, 2], left=counts[:, 0] + counts[:, 1], color=C["sand"],
            edgecolor="black", lw=0.5, label="above")
    ax.set_yticks(y)
    ax.set_yticklabels(["%s\n%.3f" % (n, corr[k]["vis"])
                        for n, k in zip(names, order)], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("fused inputs (of 15)")
    ax.set_xlim(0, 15)
    fs.tidy(ax, grid="x")
    fs.legend(ax, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3,
              fontsize=7.8, columnspacing=0.9, handlelength=1.2)
    fs.panel(ax, "c")
    save(fig, "fig_corrupt_v1")
    plt.close(fig)


def main():
    fs.use()
    fig_claim()
    fig_regime()
    fig_robust()
    fig_decision()
    extra = load(r"D:\CSF-main\ssgf\_restore_extra.json")
    fig_corr(extra)
    fig_criterion(extra)
    fig_operating(extra)
    fig_tau(load(r"D:\CSF-main\ssgf\_restore_tau.json"))
    fig_scale(load(r"D:\CSF-main\ssgf\_restore_yolo11s.json"),
              load(r"D:\CSF-main\ssgf\_restore_resolution.json"))
    fig_corrupt(load(r"D:\CSF-main\ssgf\_restore_corruption.json"))
    print("styled figures written to", OUT)


if __name__ == "__main__":
    main()
