"""Regenerate all final manuscript figures and source-data tables.

The historical generator filenames reflect an earlier display order.  This
driver records the mapping to the final manuscript order and writes outputs
under the current figure and source-data filenames.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent

MAIN_FIGURE_MODULES = (
    "make_ipt_fig2",   # final Figure 1
    "make_ipt_fig5",   # final Figure 2
    "make_ipt_fig8",   # final Figure 4
    "make_ipt_fig11",  # final Figure 6
)

SUPPLEMENT_FIGURE_MODULES = (
    (
        "make_ipt_fig7",
        "Figure6_controls_heuristic.pdf",
        "FigureS1_controls_heuristic.pdf",
        "Figure6_source.csv",
        "FigureS1_controls_source.csv",
    ),
    (
        "make_ipt_fig6",
        "Figure4_checkpoint_intervals.pdf",
        "FigureS2_checkpoint_intervals.pdf",
        "Figure4_source.csv",
        "FigureS2_checkpoint_source.csv",
    ),
    (
        "make_ipt_fig9",
        "Figure8_calibration_boundary.pdf",
        "FigureS3_calibration_boundary.pdf",
        "Figure8_source.csv",
        "FigureS3_calibration_source.csv",
    ),
    (
        "make_ipt_fig4",
        "Figure10_thermal_class_boundary.pdf",
        "FigureS4_thermal_class_boundary.pdf",
        "Figure10_source.csv",
        "FigureS4_thermal_class_source.csv",
    ),
    (
        "make_ipt_fig1",
        "Figure3_evaluator_dependence.pdf",
        "FigureS5_evaluator_dependence.pdf",
        "Figure3_source.csv",
        "FigureS5_evaluator_source.csv",
    ),
    (
        "make_ipt_fig3",
        "Figure9_observed_tail_tradeoff.pdf",
        "FigureS6_observed_tail_tradeoff.pdf",
        "Figure9_source.csv",
        "FigureS6_tail_source.csv",
    ),
    (
        "make_ipt_fig10",
        "Figure5_cdf_comparison.pdf",
        "FigureS7_cdf_comparison.pdf",
        "Figure5_source.csv",
        "FigureS7_cdf_source.csv",
    ),
)


def main() -> None:
    os.environ.setdefault("SSGF_ROOT", str(PACKAGE))
    os.environ.setdefault("SSGF_RESULTS_ROOT", str(HERE / "inputs"))
    os.environ.setdefault("SSGF_FIGURE_ROOT", str(PACKAGE / "figures"))
    os.environ.setdefault("SSGF_SOURCE_DATA_ROOT", str(PACKAGE / "source_data"))
    os.environ.setdefault("SSGF_FIGURE_SUFFIXES", ".pdf")

    figure_dir = PACKAGE / "figures"
    source_dir = PACKAGE / "source_data"
    case_source = source_dir / "Figure6_source.csv"
    case_source_bytes = case_source.read_bytes() if case_source.is_file() else None

    for module_name, old_figure, new_figure, old_source, new_source in SUPPLEMENT_FIGURE_MODULES:
        print(f"Building supplementary {module_name}")
        module = importlib.import_module(module_name)
        module.main()
        (figure_dir / old_figure).replace(figure_dir / new_figure)
        (source_dir / old_source).replace(source_dir / new_source)

    if case_source_bytes is not None:
        case_source.write_bytes(case_source_bytes)

    print("Building make_ipt_fig_core")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "make_ipt_fig_core_v2.py"),
            "--policy",
            str(HERE / "inputs" / "confirmatory_policy_comparison.json"),
            "--prospective",
            str(
                PACKAGE
                / "evidence"
                / "prospective_standard_matching"
                / "analysis.json"
            ),
            "--vedai",
            str(
                PACKAGE
                / "evidence"
                / "vedai"
                / "standard_matching"
                / "analysis.json"
            ),
            "--out",
            str(figure_dir / "Figure3_core_evidence.pdf"),
            "--source-out",
            str(source_dir / "Figure3_source.csv"),
        ],
        check=True,
    )

    for module_name in MAIN_FIGURE_MODULES:
        print(f"Building {module_name}")
        module = importlib.import_module(module_name)
        module.main()

    shutil.copy2(
        figure_dir / "Figure2_selection_sensitivity.pdf",
        figure_dir / "FigureS8_selection_sensitivity.pdf",
    )
    shutil.copy2(
        source_dir / "Figure2_source.csv",
        source_dir / "FigureS8_selection_source.csv",
    )

    print("Wrote 5 main-paper figures and 8 supplementary figures")
    print("The case-study figure is built by make_ipt_case_studies.py")


if __name__ == "__main__":
    main()
