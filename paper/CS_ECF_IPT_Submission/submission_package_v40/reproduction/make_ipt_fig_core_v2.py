"""Build the compact three-panel core evidence figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.3,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.unicode_minus": False,
    }
)

DATASET_COLORS = {
    "LLVIP": "#557FA8",
    "M3FD": "#3F8E89",
    "DroneVehicle": "#C07A57",
}


def add_panel_label(ax, label: str, x: float = -0.22, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="#263238",
    )

RETROSPECTIVE_ROWS = (
    ("llvip/yolo11", "LLVIP V11", "llvip"),
    ("llvip/agnostic", "LLVIP Agn", "llvip"),
    ("llvip/yolo26_s0", "LLVIP 26-0", "llvip"),
    ("llvip/rtdetr", "LLVIP RT", "llvip"),
    ("llvip/yolo26_s1", "LLVIP 26-1", "llvip"),
    ("llvip/yolo26_s2", "LLVIP 26-2", "llvip"),
    ("llvip/yolo8_s0", "LLVIP 8-0", "llvip"),
    ("llvip/yolo8_s1", "LLVIP 8-1", "llvip"),
    ("m3fd/yolo26_m3fd6", "M3FD 26", "m3fd"),
    ("m3fd/yolo8n_m3fd6", "M3FD 8", "m3fd"),
    ("dronevehicle_test/best", "DV best", "dronevehicle_test"),
    ("dronevehicle_test/last", "DV last", "dronevehicle_test"),
)

RETROSPECTIVE_POLICIES = (
    ("always_cdf", "CDF"),
    ("always_pia", "PIA"),
    ("always_spg", "SPG"),
)


def load_retrospective(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["gain"] = float(row["equal_group_gain_ap50_points"])
        row["low"] = float(row["ci_low_ap50_points"])
        row["high"] = float(row["ci_high_ap50_points"])
        row["dataset"] = row["checkpoint"].split("/", 1)[0]
    return rows


def prospective_contrast(prospective: dict, name: str) -> tuple[float, float, float]:
    if "llvip" in prospective:
        block = prospective["llvip"]["primary_six_last_checkpoints"][
            "comparisons"
        ][name]
    else:
        block = prospective["confirmation"]["comparisons"][name]
    low, high = block["conditional_sequence_by_run_interval"]
    return (
        100.0 * float(block["summary"]["sequence_equal_point"]),
        100.0 * float(low),
        100.0 * float(high),
    )


def vedai_contrast(vedai: dict, name: str) -> tuple[float, float, float]:
    block = (
        vedai["analysis"]["aggregate"][name]
        if "analysis" in vedai
        else vedai["aggregate"][name]
    )
    low, high = block["conditional_run_by_block_interval"]
    return 100.0 * float(block["equal_run_mean"]), 100.0 * float(low), 100.0 * float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prospective", type=Path, required=True)
    parser.add_argument("--vedai", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    prospective = json.loads(args.prospective.read_text(encoding="utf-8"))
    vedai = json.loads(args.vedai.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.09, 3.25),
        gridspec_kw={"width_ratios": (1.48, 1.08, 0.76), "wspace": 0.48},
    )

    ax = axes[0]
    matrix = np.asarray([
        [
            100.0 * float(policy["per_checkpoint"][key][name])
            for name, _ in RETROSPECTIVE_POLICIES
        ]
        for key, _, _ in RETROSPECTIVE_ROWS
    ])
    policy_colors = {
        "always_cdf": "#416C94",
        "always_pia": "#3F8E89",
        "always_spg": "#8B5EAA",
    }
    policy_markers = {
        "always_cdf": "o",
        "always_pia": "^",
        "always_spg": "s",
    }
    y_positions = np.arange(matrix.shape[0])
    for policy_index, (policy_name, policy_label) in enumerate(
        RETROSPECTIVE_POLICIES
    ):
        ax.scatter(
            matrix[:, policy_index],
            y_positions,
            s=24,
            marker=policy_markers[policy_name],
            color=policy_colors[policy_name],
            edgecolor="white",
            linewidth=0.45,
            label=policy_label,
            zorder=3,
        )
    ax.axvline(0, color="#333333", lw=0.65, zorder=1)
    ax.axhspan(-0.5, 7.5, color=DATASET_COLORS["LLVIP"], alpha=0.045, zorder=0)
    ax.axhspan(7.5, 9.5, color=DATASET_COLORS["M3FD"], alpha=0.055, zorder=0)
    ax.axhspan(9.5, 11.5, color=DATASET_COLORS["DroneVehicle"], alpha=0.055, zorder=0)
    for boundary in (7.5, 9.5):
        ax.axhline(boundary, color="#D7DDE2", lw=0.55, zorder=1)
    ax.set_xlim(-32, 16)
    ax.set_ylim(matrix.shape[0] - 0.35, -0.65)
    ax.set_yticks(range(len(RETROSPECTIVE_ROWS)))
    ax.set_yticklabels(
        [label for _, label, _ in RETROSPECTIVE_ROWS],
        fontsize=5.5,
    )
    ax.set_xlabel("Fixed fusion minus visible (AP50 pp)", fontsize=5.7)
    ax.tick_params(axis="x", labelsize=5.2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", color="#E8EDF1", linewidth=0.45, zorder=0)
    ax.set_title(
        "Evaluator-dependent fixed fusion",
        loc="left",
        fontsize=6.7,
        pad=14,
    )
    ax.legend(
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=5.1,
        columnspacing=0.8,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    add_panel_label(ax, "a", x=-0.28, y=1.08)

    ax = axes[1]
    llvip = {
        "Visible": prospective_contrast(prospective, "csecf_minus_visible"),
        "Always CDF": prospective_contrast(prospective, "csecf_minus_cdf"),
        "No shield": prospective_contrast(prospective, "csecf_minus_no_shield"),
    }
    aerial = {
        "Visible": vedai_contrast(vedai, "csecf_minus_visible"),
        "Always CDF": vedai_contrast(vedai, "csecf_minus_always_cdf"),
        "No shield": vedai_contrast(vedai, "csecf_minus_no_shield"),
    }
    comparator_colors = {
        "Visible": "#737373",
        "Always CDF": "#416C94",
        "No shield": "#D17A32",
    }
    offsets = {"LLVIP": -0.17, "VEDAI": 0.17}
    for dataset_name, values in (("LLVIP", llvip), ("VEDAI", aerial)):
        for row_index, comparator in enumerate(("Visible", "Always CDF", "No shield")):
            point, low, high = values[comparator]
            y_value = row_index + offsets[dataset_name]
            ax.plot([low, high], [y_value, y_value], color=comparator_colors[comparator], lw=1.0, alpha=0.7)
            ax.scatter(point, y_value, s=18, color=comparator_colors[comparator], marker="o" if dataset_name == "LLVIP" else "D", zorder=3)
    ax.axvline(0, color="#333333", lw=0.65)
    ax.set_xlim(-5.5, 8.5)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Visible", "Always CDF", "No shield"], fontsize=5.8)
    for tick_label, comparator in zip(
        ax.get_yticklabels(),
        ("Visible", "Always CDF", "No shield"),
        strict=True,
    ):
        tick_label.set_color(comparator_colors[comparator])
    ax.set_xlabel("CS-ECF minus comparator (AP50 pp)", fontsize=5.8)
    ax.set_title(
        "Held-out contrasts",
        loc="left",
        fontsize=6.7,
        pad=14,
    )
    ax.tick_params(axis="x", labelsize=5.3)
    ax.grid(axis="x", color="#E5E5E5", lw=0.45)
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="#333333",
            markerfacecolor="white",
            markeredgewidth=0.7,
            markersize=5.0,
            label="LLVIP",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color="#333333",
            markerfacecolor="white",
            markeredgewidth=0.7,
            markersize=5.0,
            label="VEDAI",
        ),
    ]
    ax.legend(
        handles=dataset_handles,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=5.1,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    add_panel_label(ax, "b", x=-0.28, y=1.08)

    ax = axes[2]
    groups = ("LLVIP", "VEDAI")
    no_shield = (3.270, 20.73)
    shield = (0.451, 7.51)
    y = np.asarray([1.0, 0.0])
    for index, group in enumerate(groups):
        ax.plot(
            [shield[index], no_shield[index]],
            [y[index], y[index]],
            color="#AEB8C2",
            lw=1.2,
            zorder=1,
        )
        ax.text(
            0.5 * (shield[index] + no_shield[index]),
            y[index] + 0.13,
            f"Δ {shield[index] - no_shield[index]:+.2f} pp",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color="#596672",
        )
    ax.scatter(
        no_shield,
        y,
        s=42,
        marker="o",
        facecolor="white",
        edgecolor="#D17A32",
        linewidth=1.2,
        label="No shield",
        zorder=3,
    )
    ax.scatter(
        shield,
        y,
        s=38,
        marker="s",
        facecolor="#3F8E89",
        edgecolor="white",
        linewidth=0.5,
        label="CS-ECF",
        zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=5.8)
    ax.set_ylim(-0.5, 1.55)
    ax.set_xlim(0, 23)
    ax.set_xlabel("Observed worst loss (AP50 pp)", fontsize=5.8)
    ax.set_title("Worst-case exposure", loc="left", fontsize=6.7, pad=14)
    ax.tick_params(axis="x", labelsize=5.3)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", color="#E5E5E5", lw=0.45)
    ax.legend(
        frameon=False,
        fontsize=5.0,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    add_panel_label(ax, "c", x=-0.28, y=1.08)

    fig.subplots_adjust(
        left=0.12,
        right=0.99,
        top=0.80,
        bottom=0.16,
        wspace=0.48,
    )
    for artist in fig.findobj(Text):
        if artist.get_fontsize() < 5.7:
            artist.set_fontsize(5.7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for checkpoint, _, _ in RETROSPECTIVE_ROWS:
        for policy_name, policy_label in RETROSPECTIVE_POLICIES:
            rows.append(
                {
                    "panel": "a",
                    "scope": checkpoint,
                    "comparator": policy_label,
                    "point_ap50_points": round(
                        100.0
                        * float(policy["per_checkpoint"][checkpoint][policy_name]),
                        6,
                    ),
                    "low_ap50_points": "",
                    "high_ap50_points": "",
                    "no_shield_worst_ap50_points": "",
                    "csecf_worst_ap50_points": "",
                }
            )
    for scope, values in (("LLVIP", llvip), ("VEDAI", aerial)):
        for comparator, (point, low, high) in values.items():
            rows.append(
                {
                    "panel": "b",
                    "scope": scope,
                    "comparator": comparator,
                    "point_ap50_points": point,
                    "low_ap50_points": low,
                    "high_ap50_points": high,
                    "no_shield_worst_ap50_points": "",
                    "csecf_worst_ap50_points": "",
                }
            )
    for scope, no_shield_loss, shield_loss in zip(groups, no_shield, shield):
        rows.append(
            {
                "panel": "c",
                "scope": scope,
                "comparator": "No shield vs CS-ECF",
                "point_ap50_points": "",
                "low_ap50_points": "",
                "high_ap50_points": "",
                "no_shield_worst_ap50_points": no_shield_loss,
                "csecf_worst_ap50_points": shield_loss,
            }
        )
    args.source_out.parent.mkdir(parents=True, exist_ok=True)
    with args.source_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
