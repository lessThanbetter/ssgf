"""Overall framework (target-aware semantic saliency fusion) in CSF style."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import csf_style as S
import matplotlib.pyplot as plt


def main():
    vis_img = S.read_gray("D:/CSF-main/ssgf/data/det_subset/test/visible/190002.jpg")
    ir_img = S.read_gray("D:/CSF-main/ssgf/data/det_subset/test/infrared/190002.jpg")
    if vis_img is None:
        vis_img = S.gradient_thumb(phase=0.1)
    if ir_img is None:
        ir_img = S.gradient_thumb(phase=0.8)

    fig, ax = S.new_fig(14.5, 6.2)
    ax.set_xlim(0, 23); ax.set_ylim(0, 8)

    ink = S.INK
    # ---- sources (VIS top, IR bottom) ----
    S.thumb(ax, vis_img, 0.4, 5.35, 1.5, 1.3)
    ax.text(1.15, 6.85, "Visible $I_v$", ha="center", fontsize=12, color=ink)
    S.thumb(ax, ir_img, 0.4, 1.35, 1.5, 1.3)
    ax.text(1.15, 2.85, "Infrared $I_i$", ha="center", fontsize=12, color=ink)

    # ---- shared encoder ----
    S.box(ax, 2.35, 3.5, 1.25, 1.35, "Encoder\n$E$", fc=S.WHITE, fs=13)
    S.arrow(ax, (1.9, 6.0), (2.35, 4.5), rad=-0.12)
    S.arrow(ax, (1.9, 2.0), (2.35, 3.9), rad=0.12)

    # ---- visible feature stack (orange) ----
    S.feature_fan(ax, 4.1, 5.15, 2.0, 1.5, fc=S.ORANGE, ec=S.ORANGE_EDGE,
                  n=5, dx=0.09, dy=0.05, label="$F_v$")
    ax.text(5.9, 7.2, r"feature $\varphi_v^1,\dots,\varphi_v^N$", fontsize=11,
            color=ink, style="italic")
    # ---- infrared feature stack (blue) ----
    S.feature_fan(ax, 4.1, 1.15, 2.0, 1.5, fc=S.BLUE, ec=S.BLUE_EDGE,
                  n=5, dx=0.09, dy=0.05, label="$F_i$")
    ax.text(5.4, 0.55, r"feature $\varphi_i^1,\dots,\varphi_i^N$", fontsize=11,
            color=ink, style="italic")
    S.arrow(ax, (3.6, 4.4), (4.1, 5.75), rad=-0.1)
    S.arrow(ax, (3.6, 4.0), (4.1, 2.0), rad=0.1)

    # ---- purple rule container ----
    S.box(ax, 7.0, 2.6, 5.6, 3.7, fc=S.PURPLE, ec=S.PURPLE_EDGE, lw=1.4)
    S.box(ax, 7.3, 4.75, 1.35, 0.80, "Targetness\n$S_v$", fc=S.ORANGE,
          ec=S.ORANGE_EDGE, fs=10)
    S.box(ax, 8.8, 4.75, 1.30, 0.80, "Activity\n$A_v$", fc=S.WHITE, fs=10)
    S.box(ax, 10.25, 4.75, 1.95, 0.80, "$d_v{=}S_v{\\cdot}A_v$", fc=S.WHITE, fs=11)
    S.box(ax, 7.3, 2.95, 1.35, 0.80, "Targetness\n$S_i$", fc=S.BLUE,
          ec=S.BLUE_EDGE, fs=10)
    S.box(ax, 8.8, 2.95, 1.30, 0.80, "Activity\n$A_i$", fc=S.WHITE, fs=10)
    S.box(ax, 10.25, 2.95, 1.95, 0.80, "$d_i{=}S_i{\\cdot}A_i$", fc=S.WHITE, fs=11)
    ax.text(9.7, 4.12, "targetness $=$ top-hat prior $\\Phi$\nor learned probe",
            ha="center", va="center", fontsize=9.5, color=ink, style="italic",
            linespacing=1.15, zorder=6)

    S.arrow(ax, (8.0, 5.55), (8.0, 5.05))          # S_v -> d_v
    S.arrow(ax, (10.1, 5.15), (10.25, 5.15))       # A_v -> d_v
    S.arrow(ax, (8.0, 3.75), (8.0, 3.35))          # S_i -> d_i
    S.arrow(ax, (10.1, 3.35), (10.25, 3.35))       # A_i -> d_i

    # feed activity from feature stacks
    S.arrow(ax, (6.5, 5.9), (7.3, 5.15), rad=-0.12)   # F_v -> container
    S.arrow(ax, (6.5, 1.9), (7.3, 2.95), rad=0.12)    # F_i -> container

    # ---- softmax + weighted fusion ----
    S.box(ax, 12.8, 3.55, 1.45, 1.0, "Softmax\n$w_s$ ($\\tau{=}0.3$)", fs=10)
    S.box(ax, 14.5, 3.55, 2.0, 1.0, "Weighted fusion\n$F{=}\\sum_s w_s F_s$", fs=10)
    S.arrow(ax, (12.2, 5.15), (12.8, 4.15), rad=-0.1)  # d_v -> softmax
    S.arrow(ax, (12.2, 3.35), (12.8, 3.85), rad=0.1)   # d_i -> softmax
    S.arrow(ax, (14.25, 4.05), (14.5, 4.05))

    # ---- fused stack (green) + decoder + output ----
    S.feature_fan(ax, 16.85, 3.3, 2.0, 1.5, fc=S.GREEN, ec=S.GREEN_EDGE,
                  n=5, dx=0.09, dy=0.05, label="$F$")
    ax.text(18.4, 5.35, r"$\varphi_f^1,\dots,\varphi_f^N$", fontsize=11,
            color=ink, style="italic")
    S.arrow(ax, (16.5, 4.05), (16.85, 4.05))
    S.box(ax, 19.6, 3.5, 1.35, 1.1, "Decoder\n$D$", fs=12)
    S.arrow(ax, (18.85, 4.05), (19.6, 4.05))
    S.thumb(ax, ir_img, 21.35, 3.4, 1.5, 1.3)
    ax.text(22.1, 4.9, "Fused $I_f$", ha="center", fontsize=12, color=ink)
    S.arrow(ax, (20.95, 4.05), (21.35, 4.05))

    out = "D:/CSF-main/ssgf/figures_csf/framework"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    S.save(fig, out)
    print("saved", out, ".*")


if __name__ == "__main__":
    main()
