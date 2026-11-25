"""Assemble the canonical prospective evidence file for manuscript version 21."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PROSPECTIVE = PACKAGE / "evidence" / "prospective"
CANONICAL = PROSPECTIVE / "canonical_prospective_evidence.json"
FIGURE_INPUT = HERE / "inputs" / "prospective_evidence.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_summary(directory: Path, model: str) -> dict:
    return load(directory / f"{model}_scores.json")


def main() -> None:
    primary = load(PROSPECTIVE / "llvip_primary_analysis.json")
    secondary = load(PROSPECTIVE / "llvip_original8_analysis.json")
    drone = load(PROSPECTIVE / "dronevehicle" / "native_analysis_all.json")

    protocol_paths = (
        PROSPECTIVE / "protocol" / "PREREGISTRATION.md",
        PROSPECTIVE / "protocol" / "PROTOCOL_ADDENDUM_20261029.md",
        PROSPECTIVE / "protocol" / "DRONEVEHICLE_CONFIRMATION.md",
    )
    protocol = {
        path.name: {
            "sha256": sha256(path),
            "externally_registered": False,
        }
        for path in protocol_paths
    }

    run1 = drone["yolo11n_obb_run1"]
    run2 = drone["yolo11n_obb_run2"]
    new_run_mean = (
        run1["summary"]["cs_ecf"]["mean_gain"]
        + run2["summary"]["cs_ecf"]["mean_gain"]
    ) / 2.0

    primary_policy = primary["confirmation"]["frozen_policy"]
    margin_grid = primary["validation_margin_audit"]["grid"]
    frozen_audit = margin_grid["1.50"]
    quality = primary["matched_quality"]

    payload = {
        "schema_version": "1.0",
        "recorded": "2026-10-29",
        "scope": (
            "Post-hoc assembly of locally recorded prospective analyses. "
            "The protocol files were hashed before the confirmation outputs; "
            "they were not externally registered."
        ),
        "protocol": protocol,
        "llvip": {
            "data_separation": {
                "training_sequences": ["010", "020", "030"],
                "policy_audit_sequence": "040",
                "confirmation_sequences": ["050", "060", "070"],
                "confirmation_images": 1395,
                "primary_checkpoints": primary["protocol"]["models"],
                "secondary_checkpoints": secondary["checkpoints"],
            },
            "primary_six_last_checkpoints": {
                "summary": primary_policy,
                "conditional_sequence_by_run_interval": primary[
                    "confirmation"
                ]["conditional_sequence_by_run_interval"],
                "comparisons": primary["confirmation"]["comparisons"],
            },
            "secondary_original_eight_checkpoints": {
                "summary": secondary["confirmation"]["frozen_policy"],
                "conditional_sequence_by_run_interval": secondary[
                    "confirmation"
                ]["conditional_sequence_by_run_interval"],
                "comparisons": secondary["confirmation"]["comparisons"],
                "role": (
                    "Prespecified sensitivity analysis; it cannot replace "
                    "the primary six-checkpoint result."
                ),
            },
            "policy_margin_audit": {
                "selected_margin": primary["validation_margin_audit"][
                    "selected_margin"
                ],
                "frozen_margin": primary["protocol"]["frozen_margin"],
                "frozen_margin_validation_gain": frozen_audit[
                    "balanced_gain"
                ],
                "frozen_margin_worst_checkpoint": frozen_audit[
                    "worst_checkpoint"
                ],
                "rule": primary["validation_margin_audit"]["constraint"],
                "interpretation": (
                    "The fixed 1.5 policy was retained for confirmation, but "
                    "the audit sequence did not reselect 1.5 under the stated "
                    "selection rule."
                ),
            },
            "matched_quality_audit": {
                "vif_all_margins_select_visible": all(
                    row["switch_fraction"] == 0.0
                    for row in quality["vif"]["validation_grid"].values()
                ),
                "ssim_only_margin_1_switches": (
                    quality["ssim"]["validation_grid"]["1.00"][
                        "switch_fraction"
                    ]
                    > 0
                    and all(
                        row["switch_fraction"] == 0.0
                        for margin, row in quality["ssim"][
                            "validation_grid"
                        ].items()
                        if margin != "1.00"
                    )
                ),
                "entropy_margin_1_5": quality["entropy"][
                    "validation_grid"
                ]["1.50"],
                "claim_status": (
                    "The matched quality controls did not provide a usable "
                    "matched contrast in the prospective audit."
                ),
            },
        },
        "dronevehicle": {
            "metric": (
                "Native oriented-box AP50 using probabilistic rotated-box "
                "IoU and one-to-one maximum-cardinality matching."
            ),
            "official_test_groups": [7, 32, 71],
            "training": {
                "run1": {
                    "training_groups": [4, 12],
                    "validation_group": 13,
                    "seed": 0,
                    "absolute_ap50": score_summary(
                        PROSPECTIVE
                        / "dronevehicle"
                        / "native_scores",
                        "yolo11n_obb_run1",
                    ),
                },
                "run2": {
                    "training_groups": [4, 13],
                    "validation_group": 12,
                    "seed": 1,
                    "absolute_ap50": score_summary(
                        PROSPECTIVE
                        / "dronevehicle"
                        / "native_scores",
                        "yolo11n_obb_run2",
                    ),
                },
            },
            "checkpoint_results": {
                name: {
                    "summary": row["summary"],
                    "switch_fraction": row["switch_fraction"],
                }
                for name, row in drone.items()
                if name != "aggregate"
            },
            "checkpoint_label_aggregate": drone["aggregate"],
            "new_independent_run_equal_mean": new_run_mean,
            "new_independent_run_positive": sum(
                value > 0
                for value in (
                    run1["summary"]["cs_ecf"]["mean_gain"],
                    run2["summary"]["cs_ecf"]["mean_gain"],
                )
            ),
            "independence_note": (
                "Public best and last are dependent checkpoints from one "
                "upstream training run. The four-label equal mean therefore "
                "overweights that lineage; the two newly trained runs are the "
                "independent-run estimate."
            ),
        },
        "claim_assessment": {
            "retrospective_plus_0_0092_reproduced": False,
            "primary_llvip_interval_excludes_zero": False,
            "csecf_separated_from_always_cdf": False,
            "csecf_separated_from_no_shield_primary": True,
            "matched_confidence_mechanism_established": False,
            "supported_claim": (
                "The shield reduces the observed mean and worst loss of an "
                "unshielded selector, but does not establish uniform benefit "
                "over visible input or Always CDF."
            ),
        },
    }

    CANONICAL.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    FIGURE_INPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANONICAL, FIGURE_INPUT)
    print(f"WROTE {CANONICAL}")
    print(f"WROTE {FIGURE_INPUT}")


if __name__ == "__main__":
    main()
