"""Render compact LaTeX rows from the verified clean-split R2-2 summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROWS = [
    ("mean", "mean fusion"),
    ("activity_only", "activity-only"),
    ("tophat", "top-hat prior"),
    ("probe", "learned probe"),
    ("tophat_permuted", "top-hat, permuted"),
    ("probe_permuted", "probe, permuted"),
    ("tophat_wshuffle", "top-hat, w-shuffled"),
    ("probe_wshuffle", "probe, w-shuffled"),
    ("ensemble", "ensemble"),
    ("ifcnn", "IFCNN"),
    ("densefuse", "DenseFuse"),
    ("csf", "classif. saliency"),
    ("piafusion", "PIAFusion"),
    ("spgfusion", "SPGFusion"),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def observed_range(row: dict) -> str:
    return (
        f"${num(row['mean_contrast'])}$ "
        f"$[{num(row['min_contrast'])},{num(row['max_contrast'])}]$"
    )


def bootstrap_interval(row: dict) -> str:
    return (
        f"${num(row['mean_contrast'])}$ "
        f"$[{num(row['lo50'])},{num(row['hi50'])}]$"
    )


def q_cell(row: dict) -> str:
    q_expression = (
        "<0.002"
        if row["q_bh50"] <= 0.001998
        else f"={row['q_bh50']:.3f}"
    )
    significance = "*" if row["q_bh50"] < 0.05 else ""
    return (
        f"\\shortstack[l]{{${num(row['mean_contrast'])}$ "
        f"$[{num(row['lo50'])},{num(row['hi50'])}]$\\\\"
        f"$q{q_expression}${significance}}}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--regimes", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    summary = read_json(Path(args.summary))
    evaluators = summary["evaluators"]
    visible11 = evaluators["yolo11n_visible_clean"]
    visible26 = evaluators["yolo26n_visible_clean"]
    rtdetr = evaluators["rtdetr_l_visible_clean"]
    agnostic = evaluators["yolo11n_modality_agnostic_clean"]

    main_rows = []
    supplement_rows = []
    for key, label in ROWS:
        main_rows.append(
            " & ".join(
                [
                    label,
                    observed_range(visible11["inputs"][key]),
                    bootstrap_interval(visible26["inputs"][key]),
                    bootstrap_interval(rtdetr["inputs"][key]),
                    observed_range(agnostic["inputs"][key]),
                ]
            )
            + r"\\"
        )
        supplement_rows.append(
            " & ".join(
                [
                    label,
                    observed_range(visible11["inputs"][key]),
                    q_cell(visible26["inputs"][key]),
                    q_cell(rtdetr["inputs"][key]),
                    observed_range(agnostic["inputs"][key]),
                ]
            )
            + r"\\"
        )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "main_clean_rows.tex").write_text(
        "\n".join(main_rows),
        encoding="utf-8",
    )
    (output / "supplement_clean_rows.tex").write_text(
        "\n".join(supplement_rows),
        encoding="utf-8",
    )

    if args.regimes:
        rendered = []
        for path_text in args.regimes:
            path = Path(path_text)
            payload = read_json(path)
            cells = []
            for regime, row in payload["summary"].items():
                cells.append(
                    f"{row['below']} / {row['above']} / {row['unresolved']}"
                )
            rendered.append(
                f"{payload['detector']} & " + " & ".join(cells) + r"\\"
            )
        (output / "supplement_clean_regimes.tex").write_text(
            "\n".join(rendered),
            encoding="utf-8",
        )

    metadata = {
        "visible_source_mAP50": {
            "yolo11n_visible_clean": visible11["visible_source_mAP50"],
            "yolo26n_visible_clean": visible26["visible_source_mAP50"],
            "rtdetr_l_visible_clean": rtdetr["visible_source_mAP50"],
            "yolo11n_modality_agnostic_clean": agnostic["visible_source_mAP50"],
        },
        "summary": summary["summary"],
        "rank_associations": summary["rank_associations"],
    }
    (output / "table_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
