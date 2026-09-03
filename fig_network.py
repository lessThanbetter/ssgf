"""Draw the encoder (24-ch conv stack) + semantic probe architecture."""
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


def box(ax, x, y, w, h, text, fc='#eef3fb', ec='#2f5aa0', fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.05',
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=3, linespacing=1.2)


def arr(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
                                 linewidth=1.6, color='#2f5aa0', zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(9.4, 6.2), dpi=300)
    ax.set_xlim(0, 9.6); ax.set_ylim(0, 6.4); ax.axis('off')

    # encoder row (y=3.0): Input + 4x Conv5 stride-1 + LeakyReLU, ends in F
    box(ax, 0.1, 2.6, 0.9, 1.0, 'Input\n$x$', fc='#dce7f7')
    enc = ['Conv5 1→16\n+ Leaky', 'Conv5 16→24\n+ Leaky', 'Conv5 24→24\n+ Leaky', 'Conv5 24→24\n+ Leaky']
    x = 1.2
    for lab in enc:
        box(ax, x, 2.6, 1.45, 1.0, lab, fc='#e6eefb', fs=10)
        x += 1.55
    # F box (right end of encoder)
    fx = 1.2 + 4 * 1.55 + 0.1
    box(ax, fx, 2.6, 1.3, 1.0, 'Feature\n$F$ (24ch)', fc='#f3ecf9')
    arr(ax, 1.0, 3.1, 1.2, 3.1)
    for i in range(4):
        arr(ax, 1.2 + i * 1.55 + 1.45, 3.1, 1.2 + (i + 1) * 1.55, 3.1)
    arr(ax, 1.2 + 4 * 1.55, 3.1, fx, 3.1)

    # probe row (y=0.6): F -> Conv3 24->16(+Leaky) -> Conv3 16->1 -> sigmoid
    px = fx - 0.2
    box(ax, px, 0.6, 1.9, 1.0, 'Probe $\\Phi$: Conv3 24→16\n+ LeakyReLU', fc='#fdf0e3', ec='#b8722a', fs=10)
    box(ax, px + 2.0, 0.6, 1.4, 1.0, 'Conv3 16→1', fc='#fdf0e3', ec='#b8722a', fs=10)
    box(ax, px + 3.5, 0.6, 1.2, 1.0, '$S{=}\\sigma(\\cdot)$', fc='#fbe9e4', ec='#b8722a', fs=10)
    arr(ax, px + 1.9, 1.1, px + 2.0, 1.1)
    arr(ax, px + 3.4, 1.1, px + 3.5, 1.1)
    arr(ax, fx + 0.65, 2.6, fx + 0.65, 1.6)  # F down to probe
    arr(ax, fx + 0.65, 1.6, px + 0.95, 1.6)  # into probe first box
    ax.text(3.8, 1.85, 'encoder: 4x (Conv5, stride-1, reflection pad) + LeakyReLU (no BN)',
            fontsize=9, color='#666', style='italic')

    ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig('figures/network.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figures/network.svg', bbox_inches='tight', facecolor='white')
    plt.savefig('figures/network.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig('figures/network.tiff', dpi=300, bbox_inches='tight', facecolor='white')
    print('saved figures/network.{png,svg,pdf,tiff}')


if __name__ == '__main__':
    main()
