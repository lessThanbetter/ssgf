"""Figures for the additional-experiments section of the extended manuscript."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AGG = r"D:\CSF-main\ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\aggregate"
ROP = r"D:\CSF-main\ssgf\_restore_extra.json"
RTAU = r"D:\CSF-main\ssgf\_restore_tau.json"
OUT = r"D:\CSF-main\ssgf\_submission_IPT"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.4, "axes.linewidth": 0.6, "axes.labelsize": 7.4,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.6,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

LABEL = {"mean": "mean fusion", "activity_only": "activity-only", "tophat": "top-hat prior",
         "probe": "learned probe", "tophat_permuted": "top-hat, permuted",
         "probe_permuted": "probe, permuted", "tophat_wshuffle": "top-hat, w-shuffled",
         "probe_wshuffle": "probe, w-shuffled", "ensemble": "ensemble", "ifcnn": "IFCNN",
         "densefuse": "DenseFuse", "csf": "classif. saliency", "piafusion": "PIAFusion",
         "spgfusion": "SPGFusion", "vis": "VIS source", "ir": "IR source"}
COL = {"yolo11n_vis": "#1f4e79", "yolo26n_vis": "#c0504d", "rtdetr_l_vis": "#2e7d32",
       "agnostic": "#7030a0"}
EV = [("yolo11n_vis", "yolo11n", "YOLO11n"), ("yolo26n_vis", "yolo26n", "YOLO26n"),
      ("rtdetr_l_vis", "rtdetr_l", "RT-DETR-l"), ("agnostic", "agnostic", "agnostic")]
IOU = np.round(np.linspace(0.5, 0.95, 10), 2)
FID = json.load(open(r"D:\CSF-main\ssgf\_remote_jobs\results\j1_fidelity.json",
                     encoding="utf-8"))


def load(p):
    return json.load(open(p, encoding="utf-8"))


def fig_corr(extra):
    corr = extra["rank_correlation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    for det, fname, nice in EV:
        c = load(os.path.join(AGG, "contrasts_%s_colour.json" % fname))
        rows = [r for r in c if r in FID and r != "vis"]
        x = [FID[r]["VIF_vis"] for r in rows]
        y = [c[r]["delta50"] for r in rows]
        ax.plot(x, y, "o", ms=3.2, color=COL[det], label="%s (rho %+.2f)"
                % (nice, corr[det]["VIF"]))
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xlabel(r"reported-index VIF of the fused image")
    ax.set_ylabel(r"$\Delta$ mAP50 against the colour-scored source")
    ax.set_title("(a) fidelity against the downstream contrast", fontsize=7.2)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    names = ["VIF", "SSIM_vis", "EN"]
    w = 0.26
    for i, (det, _fname, nice) in enumerate(EV):
        vals = [corr[det][k] for k in names]
        ax.bar(np.arange(len(names)) + (i - 1.5) * w, vals, w, color=COL[det], label=nice)
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(["VIF", "SSIM$_{vis}$", "EN"])
    ax.set_ylabel("Spearman rho with the contrast")
    ax.set_ylim(-0.8, 0.8)
    ax.set_title("(b) the same for other indices", fontsize=7.2)
    ax.legend(frameon=False, ncol=2, loc="upper right", handletextpad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_corr_v1.pdf"))
    plt.close(fig)


def fig_criterion(extra):
    crit = extra["criterion"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    show = [("mean", "o", "#4f6228"), ("probe", "s", "#2e75b6"),
            ("ifcnn", "^", "#c0504d"), ("spgfusion", "D", "#1f4e79")]
    for ax, (det, _fname, nice) in zip(axes, [(EV[0]), (EV[3])]):
        for name, mk, c in show:
            if name not in crit[det]["contrasts"]:
                continue
            y = crit[det]["contrasts"][name]
            ax.plot(IOU, y, mk + "-", ms=3.2, lw=0.9, color=c, label=LABEL[name])
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_xlabel("IoU criterion")
        ax.set_title("(%s) %s" % ("a" if det == EV[0][0] else "b", nice), fontsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$\Delta$ mAP against the visible source")
    axes[0].legend(frameon=False, loc="lower right", handletextpad=0.3)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_criterion_v1.pdf"))
    plt.close(fig)


def fig_operating(extra):
    op = extra["operating_point"]
    rows = ["vis", "mean", "probe", "ifcnn", "spgfusion"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    w = 0.2
    for i, (det, _fname, nice) in enumerate(EV):
        rec = [op[det][r]["recall"] for r in rows]
        pre = [op[det][r]["precision"] for r in rows]
        axes[0].bar(np.arange(len(rows)) + (i - 1.5) * w, rec, w, color=COL[det], label=nice)
        axes[1].bar(np.arange(len(rows)) + (i - 1.5) * w, pre, w, color=COL[det])
    for ax, title, ylab in ((axes[0], "(a) recall at maximum F1", "recall"),
                            (axes[1], "(b) precision at maximum F1", "precision")):
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([LABEL[r] for r in rows], rotation=20, ha="right", fontsize=6.4)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7.2)
        ax.set_ylim(0.0, 1.0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.45),
                   handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_operating_v1.pdf"))
    plt.close(fig)


def fig_tau(tau):
    rows = tau["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    styles = {"activity_only": ("activity-only", "o", "#c0504d"),
              "tophat": ("top-hat prior", "s", "#1f4e79"),
              "probe": ("learned probe", "^", "#2e7d32")}
    for ax, ev, nice in ((axes[0], "vis", "visible-trained"), (axes[1], "agn", "modality-agnostic")):
        for rule, (lab, mk, c) in styles.items():
            xs = [r["tau"] for r in rows if r["evaluator"] == ev and r["rule"] == rule]
            ys = [r["mAP50"] for r in rows if r["evaluator"] == ev and r["rule"] == rule]
            ax.plot(xs, ys, mk + "-", ms=3.4, lw=0.9, color=c, label=lab)
        ax.axhline(tau["mean_fusion"][ev], color="0.35", lw=0.9, ls="--")
        ax.text(2.6, tau["mean_fusion"][ev], " mean fusion", fontsize=6.2, va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1.0, 3.0])
        ax.set_xticklabels(["0.1", "0.3", "1", "3"])
        ax.set_xlabel(r"temperature $\tau$")
        ax.set_title("(%s) %s evaluator" % ("a" if ev == "vis" else "b", nice), fontsize=7.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("mAP50")
    axes[0].legend(frameon=False, loc="lower left", handletextpad=0.3)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_tau_v1.pdf"))
    plt.close(fig)


def fig_scale(y11s, res):
    """(a) the larger backbone against the reported one; (b) the inference resolution."""
    order = ["spgfusion", "ifcnn", "piafusion", "csf", "activity_only", "mean",
             "probe_wshuffle", "tophat_wshuffle", "probe_permuted", "tophat_permuted",
             "tophat", "mean", "densefuse", "ensemble", "probe", "ir"]
    seen = []
    for r in order:
        if r not in seen:
            seen.append(r)
    rows = [r for r in seen if r in y11s["rows"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    ax = axes[0]
    y = np.arange(len(rows))
    for yy, r in zip(y, rows):
        c = y11s["rows"][r]
        ax.plot([c["lo50"], c["hi50"]], [yy, yy], color="0.55", lw=0.9)
        ax.plot(c["delta50"], yy, "o", ms=3.2, color="#1f4e79")
    base = load(os.path.join(AGG, "contrasts_yolo11n_colour.json"))
    y11n = [base[r]["delta50"] if r in base else np.nan for r in rows]
    ax.plot(y11n, y, "s", ms=3.0, color="#c0504d", label="YOLO11n, reported run")
    ax.plot([y11s["rows"][r]["delta50"] for r in rows], y, "o", ms=3.2,
            color="#1f4e79", label="YOLO11s, mean over runs")
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(r, r) for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"$\Delta$ mAP50 against the colour-scored source")
    ax.set_title("(a) a larger visible-trained backbone", fontsize=7.2)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    series = [("i512", "imgsz 512", "o", "#2e7d32"),
              ("i640", "imgsz 640 (reported)", "s", "#1f4e79"),
              ("i960", "imgsz 960", "^", "#c0504d")]
    rows2 = [r for r in rows if r in res["i640"]["rows"]]
    y = np.arange(len(rows2))
    for key, nice, mk, c in series:
        ax.plot([res[key]["rows"][r]["delta50"] for r in rows2], y, mk, ms=3.0,
                color=c, label=nice)
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(r, r) for r in rows2])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows2) - 0.4)
    ax.set_xlabel(r"$\Delta$ mAP50 against the colour-scored source")
    ax.set_title("(b) the inference resolution", fontsize=7.2)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_scale_v1.pdf"))
    plt.close(fig)


def fig_corrupt(corr):
    order = ["yolo11n_vis_clean", "yolo11n_vis_jpeg30", "yolo11n_vis_blur",
             "yolo11n_vis_noise"]
    names = ["clean", "JPEG-30", "blur 1.5 px", "noise sigma 0.05"]
    rules = [("mean", "#4f6228"), ("probe", "#2e75b6"), ("ifcnn", "#c0504d"),
             ("spgfusion", "#1f4e79")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    w = 0.2
    for i, (rule, col) in enumerate(rules):
        vals = [corr[k]["rows"][rule]["delta"] for k in order]
        ax.bar(np.arange(len(order)) + (i - 1.5) * w, vals, w, color=col,
               label=LABEL[rule])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylabel(r"$\Delta$ mAP50 against the source of that condition")
    ax.set_title("(a) the contrast under each degradation", fontsize=7.2)
    ax.legend(frameon=False, ncol=2, loc="lower left", handletextpad=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    counts = []
    for k in order:
        rows = corr[k]["rows"]
        below = sum(1 for r in rows.values() if r["hi"] < 0)
        above = sum(1 for r in rows.values() if r["lo"] > 0)
        counts.append((below, len(rows) - below - above, above))
    counts = np.array(counts)
    y = np.arange(len(order))
    ax.barh(y, counts[:, 0], color="#9dc48a", edgecolor="white", lw=0.4, label="below")
    ax.barh(y, counts[:, 1], left=counts[:, 0], color="0.8", edgecolor="white", lw=0.4,
            label="unresolved")
    ax.barh(y, counts[:, 2], left=counts[:, 0] + counts[:, 1], color="#f2c14e",
            edgecolor="white", lw=0.4, label="above")
    tv = [corr[k]["vis"] for k in order]
    ax.set_yticks(y)
    ax.set_yticklabels(["%s\nvis %.3f" % (n, t) for n, t in zip(names, tv)], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("fused inputs (of 15)")
    ax.set_title("(b) resolved, per condition", fontsize=7.2)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42),
              handletextpad=0.3, columnspacing=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_corrupt_v1.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    extra = load(ROP)
    fig_corr(extra)
    fig_criterion(extra)
    fig_operating(extra)
    fig_tau(load(RTAU))
    fig_scale(load(r"D:\CSF-main\ssgf\_restore_yolo11s.json"),
              load(r"D:\CSF-main\ssgf\_restore_resolution.json"))
    fig_corrupt(load(r"D:\CSF-main\ssgf\_restore_corruption.json"))
    print("extra figures written")
