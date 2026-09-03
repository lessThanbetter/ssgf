"""Draw the overall framework of the target-aware semantic saliency fusion rule."""
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 11,
})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text, fc='#eef3fb', ec='#2f5aa0', fs=14, tc='#12233f', lw=1.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, zorder=3, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2, color='#2f5aa0', lw=1.6, style='-|>', rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color, zorder=1,
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11.6, 5.9), dpi=300)
    ax.set_xlim(0, 19.3); ax.set_ylim(0, 7.4)
    ax.axis('off')

    R1, R2 = 4.9, 1.9          # row y for visible / infrared branch
    # ---- visible branch (top row) ----
    box(ax, 0.4, R1, 1.5, 1.0, 'Visible\n$I_v$', fc='#dce7f7')
    box(ax, 2.7, 2.45, 1.9, 2.3, 'Feature\nEncoder $E$', fc='#eef3fb')
    box(ax, 5.0, R1, 1.5, 1.0, 'Feature\n$F_v$', fc='#e6eefb')
    box(ax, 7.0, R1, 1.5, 1.0, 'Activity\n$A_v$', fc='#f3ecf9')
    box(ax, 7.0, 6.05, 1.5, 0.78, 'Targetness\n$S_v$', fc='#fdf0e3', fs=12)
    box(ax, 9.0, R1, 1.8, 1.0, '$d_v{=}S_v{\\cdot}A_v$', fc='#fbe9e4')
    arrow(ax, 1.9, R1 + 0.5, 2.7, 3.0)
    arrow(ax, 4.6, 3.4, 5.0, R1 + 0.4)
    arrow(ax, 6.5, R1 + 0.5, 7.0, R1 + 0.5)
    arrow(ax, 8.5, R1 + 0.5, 9.0, R1 + 0.5)
    arrow(ax, 7.75, 6.05, 9.0, R1 + 0.85)     # S_v -> d_v

    # ---- infrared branch (bottom row) ----
    box(ax, 0.4, R2 - 0.4, 1.5, 1.0, 'Infrared\n$I_i$', fc='#c9d8ef')
    box(ax, 5.0, R2 - 0.4, 1.5, 1.0, 'Feature\n$F_i$', fc='#dde8f8')
    box(ax, 7.0, R2 - 0.4, 1.5, 1.0, 'Activity\n$A_i$', fc='#f3ecf9')
    box(ax, 7.0, 0.32, 1.5, 0.78, 'Targetness\n$S_i$', fc='#fdf0e3', fs=12)
    box(ax, 9.0, R2 - 0.4, 1.8, 1.0, '$d_i{=}S_i{\\cdot}A_i$', fc='#fbe9e4')
    arrow(ax, 1.9, R2, 2.7, 3.3)
    arrow(ax, 4.6, 3.4, 5.0, R2)
    arrow(ax, 6.5, R2, 7.0, R2)
    arrow(ax, 8.5, R2, 9.0, R2)
    arrow(ax, 7.75, 1.1, 9.0, R2)             # S_i -> d_i

    # ---- softmax + fusion + decoder + output (center) ----
    box(ax, 11.1, 2.9, 2.3, 1.4, 'Softmax\n$w_s$  ($\\tau{=}0.3$)', fc='#e6f2e9', fs=12)
    box(ax, 13.7, 2.9, 2.3, 1.5, 'Weighted fusion\n$F{=}\\sum_s w_s F_s$',
        fc='#e9eef9', fs=12)
    box(ax, 16.2, 3.0, 1.5, 1.3, 'Decoder\n$D$', fc='#dce7f7')
    box(ax, 17.9, 3.6, 1.1, 1.1, '$I_f$', fc='#f0f0f0', ec='#555', lw=1.2)
    arrow(ax, 10.8, R1 + 0.5, 11.1, 4.0)      # d_v -> softmax (down-right)
    arrow(ax, 10.8, R2, 11.1, 3.2)            # d_i -> softmax (up-right)
    arrow(ax, 13.4, 3.6, 13.7, 3.6)           # softmax -> fusion
    arrow(ax, 16.0, 3.6, 16.2, 3.6)           # fusion -> decoder
    arrow(ax, 17.7, 3.7, 17.9, 4.0)           # decoder -> output

    ax.text(3.2, 1.05, 'shared encoder, self-supervised reconstruction',
            ha='center', fontsize=11, color='#666', style='italic')
    ax.text(7.75, 6.95, 'targetness = top-hat prior  or  learned probe $\\Phi$',
            ha='center', fontsize=12, color='#7a4a0a', style='italic')

    plt.tight_layout()
    plt.savefig('figures/framework.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figures/framework.svg', bbox_inches='tight', facecolor='white')
    plt.savefig('figures/framework.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig('figures/framework.tiff', dpi=300, bbox_inches='tight', facecolor='white')
    print('saved figures/framework.{png,svg,pdf,tiff}')


if __name__ == '__main__':
    main()
