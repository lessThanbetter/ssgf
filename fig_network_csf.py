"""Encoder + semantic-probe architecture in CSF style (mirrors CSF Fig. 3)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import csf_style as S
import matplotlib.pyplot as plt


def main():
    fig, ax = S.new_fig(12.5, 6.2)
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.6)
    ink = S.INK

    # ---- encoder stages: two-cell columns (Conv cell + LeakyReLU cell) ----
    specs = ["Conv5 1→16", "Conv5 16→24", "Conv5 24→24", "Conv5 24→24"]
    wcell, gap = 1.95, 0.22
    x0 = 1.9
    yc_top, yc_bot, hcell = 2.15, 1.15, 0.78

    # input
    S.box(ax, 0.3, yc_bot + 0.35, 0.8, 1.0, "Input\n$x$", fc=S.WHITE, fs=11)
    S.arrow(ax, (1.1, yc_bot + 0.85), (x0, yc_bot + 0.85))

    col_fill = [S.BLUE, S.BLUE, S.BLUE, S.BLUE]
    act_fill = S.GREEN
    for i, lab in enumerate(specs):
        cx = x0 + i * (wcell + gap)
        # conv cell (top)
        S.box(ax, cx, yc_top, wcell, hcell, lab, fc=col_fill[i], ec=S.BLUE_EDGE,
              fs=9.5, rounding=0.08)
        # activation cell (bottom)
        S.box(ax, cx, yc_bot, wcell, hcell, "LeakyReLU", fc=act_fill,
              ec=S.GREEN_EDGE, fs=9.5, rounding=0.08)
        S.box(ax, cx + 0.02, yc_bot - 0.72, wcell - 0.04, 0.5,
              f"Layer$_{i+1}$", fc="none", ec="none", fs=10, italic=True)
        if i < len(specs) - 1:
            S.arrow(ax, (cx + wcell, yc_top + hcell / 2),
                    (cx + wcell + gap, yc_top + hcell / 2))

    fx = x0 + 4 * (wcell + gap) + 0.2
    S.feature_fan(ax, fx, 1.5, 1.6, 1.3, n=4, fc=S.ORANGE, ec=S.ORANGE_EDGE,
                  dx=0.08, dy=0.05, label="$F$\n(24ch)", fs=9)
    S.arrow(ax, (x0 + 4 * (wcell + gap) - gap, yc_top + hcell / 2),
            (fx, yc_top + hcell / 2))

    # ---- probe (top row, orange) ----
    py = 5.35
    px = fx - 0.5
    S.box(ax, px, py, 2.3, 1.35, "Probe $\\Phi$:\nConv3 24→16\n+ LeakyReLU",
          fc=S.ORANGE, ec=S.ORANGE_EDGE, fs=9.5)
    S.box(ax, px + 2.45, py + 0.2, 1.6, 1.0, "Conv3\n16→1",
          fc=S.ORANGE, ec=S.ORANGE_EDGE, fs=9.5)
    S.box(ax, px + 4.2, py + 0.2, 1.5, 1.0, "$S{=}\\sigma(\\cdot)$",
          fc=S.ORANGE, ec=S.ORANGE_EDGE, fs=9.5)
    S.arrow(ax, (px + 2.3, py + 0.7), (px + 2.45, py + 0.7))
    S.arrow(ax, (px + 4.05, py + 0.7), (px + 4.2, py + 0.7))
    # F -> probe (elbow)
    S.arrow(ax, (fx + 0.8, 2.8), (fx + 0.8, py + 0.05), lw=1.1)
    S.arrow(ax, (fx + 0.8, py + 0.05), (px + 1.15, py + 0.05), lw=1.1)

    ax.text(6.0, 0.35,
            "encoder: 4×(Conv5, stride-1, reflection pad) + LeakyReLU (no BN)",
            ha="center", fontsize=9.5, color=S.GRAY, style="italic")
    ax.text(px + 2.4, 7.15, "semantic probe (targetness map)",
            ha="center", fontsize=9.5, color=S.GRAY, style="italic")

    out = "D:/CSF-main/ssgf/figures_csf/network"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    S.save(fig, out)
    print("saved", out, ".*")


if __name__ == "__main__":
    main()
