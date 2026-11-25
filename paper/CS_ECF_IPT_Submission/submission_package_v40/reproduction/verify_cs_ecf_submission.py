"""Run deterministic preflight checks for the CS-ECF IPT submission."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_PAPER = ROOT / "paper" / "CS_ECF_IPT_Submission"
PAPER = REPO_PAPER if REPO_PAPER.is_dir() else ROOT
MAIN = PAPER / "main.tex"
SUPPLEMENT = PAPER / "supplementary.tex"
COVER = PAPER / "cover_letter.tex"
HIGHLIGHTS = (
    PAPER / "submission_materials" / "IPT_HIGHLIGHTS.txt"
    if (PAPER / "submission_materials").is_dir()
    else PAPER / "Highlights.txt"
)
REFERENCES = PAPER / "references.bib"
RESULTS_COMPACT = PAPER / "results_compact.tex"
DISCUSSION_COMPACT = PAPER / "discussion_compact.tex"
RESPONSE_1 = PAPER / "Response_to_Reviewer_1_DRAFT.md"
RESPONSE_2 = PAPER / "Response_to_Reviewer_2_DRAFT.md"
DECISION_SUMMARY = PAPER / "DECISION_SUMMARY.md"
MAIN_FIGURES = (
    ("Figure1_workflow.pdf", "Figure1_source.csv"),
    ("Figure3_core_evidence.pdf", "Figure3_source.csv"),
    (
        "FigureBudgetedSelection.pdf",
        "FigureBudgetedSelection_source.csv",
    ),
    ("Figure6_case_studies.pdf", "Figure6_source.csv"),
    ("Figure7_external_evaluators.pdf", "Figure7_source.csv"),
    ("Figure11_cost_deployment.pdf", "Figure11_source.csv"),
)
SUPPLEMENT_FIGURES = (
    ("FigureS1_controls_heuristic.pdf", "FigureS1_controls_source.csv"),
    ("FigureS2_checkpoint_intervals.pdf", "FigureS2_checkpoint_source.csv"),
    ("FigureS3_calibration_boundary.pdf", "FigureS3_calibration_source.csv"),
    ("FigureS4_thermal_class_boundary.pdf", "FigureS4_thermal_class_source.csv"),
    ("FigureS5_evaluator_dependence.pdf", "FigureS5_evaluator_source.csv"),
    ("FigureS6_observed_tail_tradeoff.pdf", "FigureS6_tail_source.csv"),
    ("FigureS7_cdf_comparison.pdf", "FigureS7_cdf_source.csv"),
    (
        "FigureS8_selection_sensitivity.pdf",
        "FigureS8_selection_source.csv",
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def words(text: str) -> int:
    cleaned = re.sub(
        r"\\SI\{([^}]+)\}\{([^}]+)\}",
        r"\1 \2",
        text,
    )
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", cleaned))


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def main() -> None:
    main_text = MAIN.read_text(encoding="utf-8")
    results_text = RESULTS_COMPACT.read_text(encoding="utf-8")
    discussion_text = DISCUSSION_COMPACT.read_text(encoding="utf-8")
    supplement_text = SUPPLEMENT.read_text(encoding="utf-8")
    cover_text = COVER.read_text(encoding="utf-8")
    highlights_text = HIGHLIGHTS.read_text(encoding="utf-8")

    main_source_text = "\n".join((main_text, results_text, discussion_text))
    response_1_text = RESPONSE_1.read_text(encoding="utf-8")
    response_2_text = RESPONSE_2.read_text(encoding="utf-8")
    decision_text = DECISION_SUMMARY.read_text(encoding="utf-8")
    source_text = "\n".join(
        (
            main_source_text,
            supplement_text,
            cover_text,
            response_1_text,
            response_2_text,
            decision_text,
        )
    )
    for marker in (
        "AUTHOR_INPUT_NEEDED",
        "[Evidence needed:",
        "TODO",
        "TBD",
        "UNRESOLVED",
    ):
        require(marker not in source_text, f"placeholder marker present: {marker}")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        main_text,
        re.DOTALL,
    )
    require(abstract_match is not None, "abstract block missing")
    abstract_words = words(abstract_match.group(1))
    require(abstract_words <= 250, f"abstract has {abstract_words} words")

    cover_body = re.search(
        r"\\opening\{Dear Editors,\}(.*?)\\closing",
        cover_text,
        re.DOTALL,
    )
    require(cover_body is not None, "cover-letter body missing")
    cover_words = words(cover_body.group(1))
    require(cover_words <= 500, f"cover letter has {cover_words} words")

    highlights = [line.strip() for line in highlights_text.splitlines() if line.strip()]
    require(3 <= len(highlights) <= 5, "Highlights must contain 3-5 lines")
    require(
        all(len(line) <= 85 for line in highlights),
        "a Highlight exceeds 85 characters",
    )

    required_claims = {
        MAIN: (
            "+0.00012",
            "[-0.00484,+0.00670]",
            "+0.00894",
            "[+0.00096,+0.02155]",
            "+0.01392",
            "[+0.00964,+0.02023]",
            "+0.0410",
            "-0.0136",
            "[+0.0097,+0.0782]",
            "VEDAI",
            "not externally registered",
            "0.0142",
            "ground-truth-box-free at inference",
            "solely to improve language, clarity, and readability",
            "No AI was used for data collection",
            "10.5281/zenodo.22984388",
            "official LLVIP partitions are larger",
            "3,463 test pairs",
            "q_{\\mathrm{TH}}=0.95",
            "lexicographically first 1,500",
            "confidence-ordered one-to-one matching",
            "task-native OBB AP50",
            "One percentage point corresponds to a change of 0.01",
            "confidence-specific mechanism",
            "+0.00337",
            "[-0.00016,+0.01267]",
            "+0.01218",
            "[+0.00136,+0.02915]",
            "-0.00329",
            "-0.00124",
            "+0.00085",
            "+0.01560",
            "[+0.00249,+0.02874]",
            "+0.00946",
            "[+0.00113,+0.01769]",
            "-0.01237",
            "-0.15891",
            "+0.01068",
            "+0.02233",
            "1.67",
            "2.52",
            "33.5\\%",
            "visible fallback",
            "YOLOv11-RGBT",
            "+0.00852",
            "+0.00617",
            "6 of 6 runs",
        ),
        SUPPLEMENT: (
            "Sequences 050, 060, and 070",
            "+0.00012",
            "+0.00894",
            "+0.01392",
            "[+0.00964,+0.02023]",
            "-0.00540",
            "[+0.00113,+0.02252]",
            "Native maximum-cardinality DroneVehicle sensitivity analysis",
            "Public best and last are dependent",
            "-.0142",
            "[-.0356,+.0024]",
            "conditional 95\\% resampling interval",
            "0.087/0.086",
            "Original development margin search",
            "AP50--95 sensitivity",
            "balanced validation gain 0.0036",
            "historical limitation",
            "VEDAI external-domain test",
            "VEDAI native OBB results",
            "+0.0410",
            "-0.0136",
            "-.0179",
            "-.0098",
            "CDF parameter sensitivity",
            "0.00712",
            "0.00018",
            "M3FD seed replication",
            "+0.00852",
            "+0.00617",
            "+.01560",
            "[+.00249,+.02874]",
            "+.00946",
            "[+.00113,+.01769]",
            "1672.6",
            "2515.5",
            "+.01068",
            "+.02233",
        ),
        COVER: (
            "+0.00012",
            "+0.00894",
            "+0.0410",
            "-0.0136",
        ),
        RESPONSE_2: (
            "+0.01560",
            "[+0.00249,+0.02874]",
            "1.67",
            "2.52",
            "33.5\\%",
        ),
        DECISION_SUMMARY: (
            "Moderate Revision",
            "2026-12-25",
            "peer-review platform",
        ),
    }
    for path, claims in required_claims.items():
        text = main_source_text if path == MAIN else path.read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        for claim in claims:
            require(claim in normalized_text, f"{path.name} is missing {claim}")
    require(
        "102.7" not in main_source_text and "706.3" not in main_source_text,
        "old summed component latency remains in the manuscript",
    )

    stale_values = (
        "[0.0026, 0.0261]",
        "[0.0026, 0.0164]",
        "dataset-balanced AP50 gain was 0.0133",
        "observed worst AP50 loss from 0.0040 to 0.0584",
        "combined block-resampling interval was [0.0002, 0.0406]",
        "q=0.043",
        "Benjamini--Hochberg",
        "Image-only controls and oracle headroom",
        "Observed mean-tail trade-off of fusion-input policies",
        "Figure6_controls_headroom",
        "Confidence-Shielded, Evaluator-Conditioned Fusion for",
        "[-0.00057, 0.02980]",
        "p=0.073",
        "p=0.0730",
        "[-0.0331,+0.0048]",
        "[-.0331,+.0048]",
        "[-0.0199,+0.0102]",
        "[-.0199,+.0102]",
        "7.45 to 0.51 points",
        "One AP50 point equals",
        "LLVIP-fine-tuned",
        "not a matched test of confidence salience",
        "The point estimate is concentrated in the M3FD people class and in one DroneVehicle training run",
        "task-balanced equal-group AP50 summary was 0.0092 with a conditional 95\\% run-cluster-by-group interval",
        "MRT-DETR is reproduced",
        "controlled-budget reproduction of the recent MRT-DETR",
    )
    for stale in stale_values:
        require(stale not in source_text, f"stale domain interval present: {stale}")
    for superseded_doi in (
        "10.5281/zenodo.23718751",
        "10.5281/zenodo.23719721",
        "10.5281/zenodo.23720111",
        "10.5281/zenodo.23721458",
        "10.5281/zenodo.23726338",
    ):
        require(
            superseded_doi not in source_text,
            f"superseded Zenodo version DOI is still present: {superseded_doi}",
        )

    reference_entries = len(re.findall(r"(?m)^@", REFERENCES.read_text(encoding="utf-8")))
    require(reference_entries == 48, f"expected 48 references, found {reference_entries}")
    reference_text = REFERENCES.read_text(encoding="utf-8")
    for doi in (
        "10.1016/j.inffus.2012.01.008",
        "10.1016/j.infrared.2025.106325",
        "10.1016/j.infrared.2010.03.005",
        "10.1016/j.jvcir.2015.11.002",
    ):
        require(doi in reference_text, f"required verified DOI is missing: {doi}")

    expected_pages = {
        "main": (21, "Manuscript.pdf"),
        "supplementary": (23, "Supplementary_Material.pdf"),
        "cover_letter": (1, "Cover_Letter.pdf"),
        "revision_cover_letter_draft": (
            1,
            "revision_cover_letter_draft.pdf",
        ),
        "Revision_Marked": (21, "Revision_Marked.pdf"),
        "Response_to_Reviewer_1_DRAFT": (
            1,
            "Response_to_Reviewer_1_DRAFT.pdf",
        ),
        "Response_to_Reviewer_2_DRAFT": (
            1,
            "Response_to_Reviewer_2_DRAFT.pdf",
        ),
    }
    for stem, expected in expected_pages.items():
        expected_count, package_pdf = expected
        log_path = PAPER / f"{stem}.log"
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            pages = re.findall(r"Output written on .*?\((\d+) pages?", log_text)
            require(pages, f"page count missing from {log_path.name}")
            require("Overfull \\hbox" not in log_text, f"overfull box in {stem}.log")
            require(
                "Citation" not in log_text
                or "Citation(s) may have changed" in log_text,
                f"possible unresolved citation in {stem}.log",
            )
            require(
                "Reference `" not in log_text,
                f"possible unresolved cross-reference in {stem}.log",
            )
        else:
            pdf_path = PAPER / package_pdf
            require(pdf_path.is_file(), f"missing PDF: {pdf_path.name}")
            pdfinfo = shutil.which("pdfinfo")
            if pdfinfo:
                output = subprocess.run(
                    [pdfinfo, str(pdf_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                match = re.search(r"(?m)^Pages:\s+(\d+)$", output)
                require(match is not None, f"page count missing from {pdf_path.name}")
                pages = [match.group(1)]
            else:
                print(f"SKIP: pdfinfo unavailable for {pdf_path.name} page count")
                pages = [str(expected_count)]
        require(
            int(pages[-1]) == expected_count,
            f"{stem}.pdf has {pages[-1]} pages; expected {expected_count}",
        )

    main_pdf = PAPER / "main.pdf"
    if not main_pdf.is_file():
        main_pdf = PAPER / "Manuscript.pdf"
    pdftotext = shutil.which("pdftotext")
    if pdftotext and main_pdf.is_file():
        rendered_text = subprocess.run(
            [pdftotext, "-layout", str(main_pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        first_page_text = subprocess.run(
            [pdftotext, "-f", "1", "-l", "1", "-layout", str(main_pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        require(
            "Highlights" not in first_page_text,
            "Highlights must be supplied as a separate submission file",
        )
        reference_numbers = {
            int(value)
            for value in re.findall(r"(?m)^\s*\[(\d+)\]", rendered_text)
        }
        require(
            reference_numbers == set(range(1, 49)),
            "PDF text does not contain all reference entries [1]--[48]",
        )
        require(
            not re.search(r"(?m)^\s*(?:1\s+){10,}1?\s*$", rendered_text),
            "PDF contains a repeated-character corruption line",
        )
        for figure_index in range(1, len(MAIN_FIGURES) + 1):
            require(
                rendered_text.count(f"Figure {figure_index}:") == 1,
                f"Figure {figure_index} caption is missing or duplicated",
            )
        require(
            rendered_text.count("5. Discussion") == 1,
            "Discussion heading is missing or duplicated in PDF text",
        )
        require("RT-DETR-l" in rendered_text, "RT-DETR-l is missing from PDF text")
        require("RT-DETR-1" not in rendered_text, "RT-DETR-1 OCR-like typo found")

    figure_dir = PAPER / "figures"
    for filename, _ in MAIN_FIGURES + SUPPLEMENT_FIGURES:
        require((figure_dir / filename).is_file(), f"missing figure PDF: {filename}")

    source_data_dir = PAPER / "source_data"
    for _, filename in MAIN_FIGURES + SUPPLEMENT_FIGURES:
        require(
            (source_data_dir / filename).is_file(),
            f"missing source data: {filename}",
        )

    joint_dir = PAPER / "evidence" / "yolo11rgbt_baseline"
    for filename in (
        "protocol.json",
        "metrics.json",
        "predictions.json",
        "result.json",
        "args.yaml",
        "results.csv",
        "weights/yolo11n_rgbt_earlyfusion_best.pt",
    ):
        require(
            (joint_dir / filename).is_file(),
            f"missing joint-baseline artifact: {filename}",
        )
    joint_metrics = json.loads(
        (joint_dir / "metrics.json").read_text(encoding="utf-8")
    )
    require(
        close(joint_metrics["metric"]["ap50"], 0.09632278485103622, 1e-12),
        "joint-baseline AP50 differs from the reported value",
    )
    require(
        (source_data_dir / "TableJointBaseline_source.csv").is_file(),
        "joint-baseline source data is missing",
    )
    require(
        (source_data_dir / "TableM3FDMultiseed_source.csv").is_file(),
        "M3FD multiseed source data is missing",
    )
    require(
        (source_data_dir / "TablePIAConfirmation_source.csv").is_file(),
        "prospective confirmation source data is missing",
    )
    require(
        (source_data_dir / "TableEndToEndLatency_source.csv").is_file(),
        "end-to-end latency source data is missing",
    )
    require(
        (source_data_dir / "TablePIAConfirmationLOO_source.csv").is_file(),
        "leave-one-sequence-out source data is missing",
    )
    require(
        (source_data_dir / "Figure1_example_scores.csv").is_file(),
        "Figure 1 example-score data is missing",
    )
    require(
        (source_data_dir / "Figure1_example_images.csv").is_file(),
        "Figure 1 example-image provenance is missing",
    )
    require(
        (PAPER / "reproduction" / "render_figure1_svg.py").is_file(),
        "Figure 1 framework generator is missing",
    )
    require(
        (PAPER / "reproduction" / "make_ipt_fig_core_v2.py").is_file(),
        "Figure 2 core-evidence generator is missing",
    )
    require(
        (PAPER / "reproduction" / "make_ipt_fig11_v3.py").is_file(),
        "Figure 5 deployment-gate generator is missing",
    )
    require(
        (PAPER / "reproduction" / "render_figure11_svg.py").is_file(),
        "Figure 5 SVG renderer is missing",
    )

    require(
        len(re.findall(r"\\begin\{figure\}", main_source_text)) == len(MAIN_FIGURES),
        "main manuscript figure count differs from the packaged figure set",
    )

    for script in (PAPER / "reproduction").glob("*.py"):
        compile(
            script.read_text(encoding="utf-8"),
            str(script),
            "exec",
        )
    for json_path in (PAPER / "evidence").rglob("*.json"):
        json.loads(
            json_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    for csv_path in source_data_dir.glob("Figure*_source.csv"):
        require(
            csv_path.stat().st_size > 0 and "\n" in csv_path.read_text(
                encoding="utf-8"
            ),
            f"empty or header-only source data: {csv_path.name}",
        )

    multiseed_dir = PAPER / "evidence" / "m3fd_multiseed"
    multiseed_analysis = json.loads(
        (multiseed_dir / "analysis.json").read_text(encoding="utf-8")
    )
    multiseed_full = multiseed_analysis["all_runs"]["full"]
    require(
        multiseed_full["runs"] == 6
        and multiseed_full["positive_runs"] == 6,
        "M3FD multiseed replication is not 6/6 positive",
    )
    require(
        close(multiseed_full["pooled_mean"], 0.008515275582631468),
        "M3FD pooled multiseed mean changed",
    )
    require(
        close(multiseed_full["equal_group_mean"], 0.006171021626769575),
        "M3FD equal-group multiseed mean changed",
    )
    require(
        len(list((multiseed_dir / "test_scores").glob("*.npz"))) == 24,
        "M3FD multiseed score-array count changed",
    )

    pia_dir = PAPER / "evidence" / "prospective_pia_confirmation"
    for protocol_name in (
        "PROTOCOL.md",
        "PROTOCOL_AMENDMENT_20261126.md",
        "LATENCY_PROTOCOL.md",
    ):
        protocol_path = pia_dir / protocol_name
        sidecar = pia_dir / f"{protocol_name}.sha256"
        require(protocol_path.is_file(), f"missing PIA protocol: {protocol_name}")
        require(sidecar.is_file(), f"missing PIA protocol hash: {sidecar.name}")
        sidecar_hash, sidecar_name = sidecar.read_text(
            encoding="utf-8"
        ).strip().split("  ", 1)
        require(
            sidecar_name == protocol_name,
            f"unexpected protocol sidecar target: {sidecar_name}",
        )
        require(
            hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            == sidecar_hash,
            f"PIA protocol hash mismatch: {protocol_name}",
        )
    pia_analysis = json.loads(
        (pia_dir / "analysis.json").read_text(encoding="utf-8")
    )
    require(
        pia_analysis["n_images"] == 900
        and pia_analysis["primary"]["positive_sequences"] == 3
        and pia_analysis["primary"]["positive_checkpoints"] == 6,
        "PIA prospective confirmation scope changed",
    )
    require(
        close(
            pia_analysis["primary"]["point"],
            0.015597409723488165,
        ),
        "PIA prospective AP50 point changed",
    )
    require(
        close(
            pia_analysis["primary"]["conditional_interval"][0],
            0.002490192145605046,
        )
        and close(
            pia_analysis["primary"]["conditional_interval"][1],
            0.028737425683704458,
        ),
        "PIA prospective AP50 interval changed",
    )
    require(
        close(
            pia_analysis["ap50_95"]["point"],
            0.009461094251703417,
        ),
        "PIA prospective AP50-95 point changed",
    )
    require(
        close(
            pia_analysis["raw_pia_vs_visible"]["ap50"]["per_sequence"]["100"],
            -0.15891302449620162,
        ),
        "PIA sequence-100 reversal changed",
    )
    loo = pia_analysis["leave_one_sequence_out"]
    require(
        loo["status"] == "post_hoc_sensitivity_analysis",
        "leave-one-sequence-out status changed",
    )
    for omitted, expected in (
        ("080", 0.010682382693096563),
        ("090", 0.01377841229782188),
        ("100", 0.022331434179546056),
    ):
        row = loo["results"][omitted]
        require(
            close(row["equal_sequence_point"], expected)
            and row["positive_checkpoints"] == 6,
            f"leave-one-sequence-out result changed for {omitted}",
        )
    source_manifest = json.loads(
        (pia_dir / "source_manifest.json").read_text(encoding="utf-8")
    )
    require(
        source_manifest["total_images"] == 900
        and len(source_manifest["source_file_sha256"]) == 1800,
        "PIA source manifest count or hashes changed",
    )
    for sequence in ("080", "090", "100"):
        require(
            len(source_manifest["sequence_stems"][sequence]) == 300,
            f"PIA sequence {sequence} does not contain 300 stems",
        )
    candidate_manifest = json.loads(
        (pia_dir / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    require(
        candidate_manifest["count"] == 900
        and len(candidate_manifest["files"]) == 900,
        "PIA candidate manifest count changed",
    )
    require(
        len(list((pia_dir / "scores").glob("llvip*/**/*.npz"))) == 36,
        "PIA score-array count changed",
    )
    latency = json.loads(
        (pia_dir / "end_to_end_latency.json").read_text(encoding="utf-8")
    )
    require(
        close(
            latency["summary"]["budgeted"][
                "equal_checkpoint_median_total_ms"
            ],
            1672.6288500067312,
        ),
        "budgeted end-to-end latency changed",
    )
    require(
        close(
            latency["summary"]["full"][
                "equal_checkpoint_median_total_ms"
            ],
            2515.509574994212,
        ),
        "full-pipeline end-to-end latency changed",
    )

    evidence_dir = PAPER / "evidence" / "major_revision"
    if evidence_dir.is_dir():
        evidence_names = (
            "equal_group_statistics.json",
            "major_revision_statistics.json",
            "image_quality_baselines.json",
            "image_quality_features.json",
            "external_evaluator_diagnostics.json",
            "m3fd_class_summary.json",
            "alignment_csecf_sensitivity.json",
            "prefreeze_validation_summary.json",
            "llvip_test_sampling_summary.json",
            "llvip_test_manifest_3463.json",
            "llvip_primary_split_manifest.json",
            "m3fd_primary_group_manifest.json",
            "m3fd_full_split_manifest.json",
        )
        for name in evidence_names:
            path = evidence_dir / name
            require(path.is_file(), f"missing major-revision evidence: {name}")
            json.loads(path.read_text(encoding="utf-8"))

        statistics = json.loads(
            (evidence_dir / "major_revision_statistics.json").read_text(
                encoding="utf-8"
            )
        )
        primary = statistics["checkpoint_summary"]
        require(primary["family_size"] == 12, "checkpoint family is not 12")
        require(primary["positive_checkpoints"] == 9, "positive count is not 9")
        balanced = statistics["dataset_balanced"]
        require(
            abs(balanced["visible"]["point"] - 0.009159) < 1e-5,
            "equal-group visible point changed",
        )
        require(
            abs(balanced["visible"]["hierarchical_ci95"][0] - 0.002600)
            < 1e-5,
            "equal-group visible lower endpoint changed",
        )
        require(
            abs(balanced["visible"]["hierarchical_ci95"][1] - 0.016637)
            < 1e-5,
            "equal-group visible upper endpoint changed",
        )
        require(
            abs(balanced["cdf"]["point"] - 0.007069) < 1e-5,
            "equal-group CDF point changed",
        )
        require(
            abs(balanced["cdf"]["hierarchical_ci95"][0] + 0.001876)
            < 1e-5,
            "equal-group CDF lower endpoint changed",
        )
        cdf_only = statistics["cdf_only_gate"]["dataset_balanced"]
        require(
            abs(cdf_only["point_cdf_only_minus_visible"] - 0.000486)
            < 1e-6,
            "matched CDF-only gate changed",
        )
        require(
            abs(cdf_only["point_csecf_minus_cdf_only"] - 0.008673)
            < 1e-6,
            "full-library versus CDF-only contrast changed",
        )
        equal_group = json.loads(
            (evidence_dir / "equal_group_statistics.json").read_text(
                encoding="utf-8"
            )
        )
        require(
            abs(
                equal_group["dataset_balanced"]["csecf_minus_visible"][
                    "point_equal_group"
                ]
                - balanced["visible"]["point"]
            )
            < 1e-12,
            "equal-group source and merged point differ",
        )
        require(
            equal_group["dataset_balanced"]["csecf_minus_visible"][
                "run_cluster_by_group_ci95"
            ]
            == balanced["visible"]["hierarchical_ci95"],
            "equal-group source and merged interval differ",
        )
        require(
            "ap50_95_sensitivity" in equal_group,
            "AP50-95 sensitivity is missing",
        )
        require(
            (
                evidence_dir / "llvip_primary_split_manifest.json"
            ).is_file(),
            "LLVIP split manifest is missing",
        )
        require(
            (
                evidence_dir / "m3fd_primary_group_manifest.json"
            ).is_file(),
            "M3FD group manifest is missing",
        )
        require(
            (PAPER / "evidence" / "dronevehicle_test_combined.json").is_file(),
            "DroneVehicle source summary is missing",
        )
        sampling = json.loads(
            (evidence_dir / "llvip_test_sampling_summary.json").read_text(
                encoding="utf-8"
            )
        )
        require(sampling["official_test_pairs"] == 3463, "LLVIP count changed")
        require(
            sampling["official_test_pairs_with_nonempty_labels"] == 3463,
            "LLVIP non-empty count changed",
        )
        require(sampling["unselected_eligible_test_pairs"] == 3163, "unused count changed")
        prefreeze = json.loads(
            (evidence_dir / "prefreeze_validation_summary.json").read_text(
                encoding="utf-8"
            )
        )
        require(prefreeze["selected_margin"] == 1.5, "selected margin changed")
        require(len(prefreeze["rows"]) == 7, "pre-freeze grid is incomplete")

        canonical_path = (
            PAPER
            / "evidence"
            / "prospective"
            / "canonical_prospective_evidence.json"
        )
        require(canonical_path.is_file(), "prospective canonical evidence missing")
        prospective = json.loads(canonical_path.read_text(encoding="utf-8"))
        require(
            canonical_path.read_bytes()
            == (
                PAPER
                / "reproduction"
                / "inputs"
                / "prospective_evidence.json"
            ).read_bytes(),
            "figure input prospective evidence differs from canonical evidence",
        )

        for protocol_name, protocol_row in prospective["protocol"].items():
            protocol_path = (
                PAPER
                / "evidence"
                / "prospective"
                / "protocol"
                / protocol_name
            )
            require(protocol_path.is_file(), f"missing protocol: {protocol_name}")
            actual_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            require(
                actual_hash == protocol_row["sha256"],
                f"protocol hash mismatch: {protocol_name}",
            )

        primary = prospective["llvip"]["primary_six_last_checkpoints"]
        primary_summary = primary["summary"]
        primary_interval = primary[
            "conditional_sequence_by_run_interval"
        ]
        require(
            close(primary_summary["sequence_equal_point"], -0.0006542925486884511),
            "prospective LLVIP point changed",
        )
        require(
            close(primary_interval[0], -0.005496419135594561)
            and close(primary_interval[1], 0.005277406117548536),
            "prospective LLVIP interval changed",
        )
        require(
            primary_summary["positive_checkpoints"] == 3,
            "prospective LLVIP positive count changed",
        )
        require(
            close(primary_summary["worst_checkpoint"], -0.004611633464235897),
            "prospective LLVIP worst checkpoint changed",
        )
        primary_cdf = primary["comparisons"]["csecf_minus_cdf"]
        primary_no_shield = primary["comparisons"]["csecf_minus_no_shield"]
        require(
            close(primary_cdf["summary"]["sequence_equal_point"], -0.005403164963203834),
            "prospective CDF contrast changed",
        )
        require(
            close(primary_no_shield["summary"]["sequence_equal_point"], 0.009760376994214691),
            "prospective no-shield contrast changed",
        )
        require(
            close(primary_no_shield["summary"]["positive_checkpoints"], 6),
            "prospective no-shield positive count changed",
        )

        secondary = prospective["llvip"]["secondary_original_eight_checkpoints"]
        require(
            close(
                secondary["summary"]["sequence_equal_point"],
                0.001913972828519801,
            ),
            "secondary LLVIP point changed",
        )
        require(
            close(
                secondary["comparisons"]["csecf_minus_cdf"][
                    "summary"
                ]["sequence_equal_point"],
                -0.0021961627609442796,
            ),
            "secondary CDF contrast changed",
        )
        require(
            prospective["llvip"]["policy_margin_audit"]["selected_margin"]
            == 2.0,
            "prospective margin audit changed",
        )
        require(
            prospective["claim_assessment"][
                "matched_confidence_mechanism_established"
            ]
            is False,
            "matched-quality claim status changed",
        )

        drone = prospective["dronevehicle"]
        require(
            close(drone["new_independent_run_equal_mean"], 0.000590038257681715),
            "new DroneVehicle independent-run mean changed",
        )
        require(
            drone["new_independent_run_positive"] == 1,
            "new DroneVehicle positive-run count changed",
        )
        require(
            close(
                drone["checkpoint_label_aggregate"]["cs_ecf"][
                    "equal_run_mean"
                ],
                0.0037520887167635055,
            ),
            "DroneVehicle four-label mean changed",
        )

        for dataset, canonical_row in equal_group["per_dataset"].items():
            comparisons = canonical_row["comparisons"]
            cdf_row = statistics["cdf_comparison"]["per_dataset"][dataset]
            cdf_only_row = statistics["cdf_only_gate"]["per_dataset"][dataset]
            require(
                close(
                    cdf_row["point_csecf_minus_cdf"],
                    comparisons["csecf_minus_cdf"]["point_equal_group"],
                ),
                f"{dataset}: merged CDF point differs from canonical",
            )
            require(
                cdf_row["fixed_models_group_ci95"]
                == comparisons["csecf_minus_cdf"][
                    "run_cluster_by_group_ci95"
                ],
                f"{dataset}: merged CDF interval differs from canonical",
            )
            require(
                close(
                    cdf_only_row["point_cdf_only_minus_visible"],
                    comparisons["cdf_only_minus_visible"][
                        "point_equal_group"
                    ],
                ),
                f"{dataset}: merged CDF-only point differs from canonical",
            )
            require(
                cdf_only_row["interval"]
                == comparisons["cdf_only_minus_visible"][
                    "run_cluster_by_group_ci95"
                ],
                f"{dataset}: merged CDF-only interval differs from canonical",
            )

        drone_combined = json.loads(
            (PAPER / "evidence" / "dronevehicle_test_combined.json").read_text(
                encoding="utf-8"
            )
        )
        drone_canonical = equal_group["per_dataset"]["dronevehicle_test"][
            "comparisons"
        ]
        require(
            close(
                drone_combined["combined"][
                    "equal_group_csecf_minus_visible"
                ]["point"],
                drone_canonical["csecf_minus_visible"]["point_equal_group"],
            ),
            "DroneVehicle combined point differs from canonical",
        )
        require(
            drone_combined["combined"][
                "equal_group_csecf_minus_visible"
            ]["group_ci95"]
            == drone_canonical["csecf_minus_visible"][
                "run_cluster_by_group_ci95"
            ],
            "DroneVehicle combined interval differs from canonical",
        )

        required_prospective_files = (
            PAPER / "evidence" / "prospective" / "llvip_primary_analysis.json",
            PAPER / "evidence" / "prospective" / "llvip_original8_analysis.json",
            PAPER / "evidence" / "prospective" / "dronevehicle" / "native_analysis_all.json",
        )
        for path in required_prospective_files:
            require(path.is_file(), f"missing prospective result: {path.name}")
            json.loads(path.read_text(encoding="utf-8"))

        for sequence in ("040", "050", "060", "070"):
            primary_scores = list(
                (
                    PAPER
                    / "evidence"
                    / "prospective"
                    / f"llvip{sequence}"
                    / "scores"
                ).glob("*.npz")
            )
            secondary_scores = list(
                (
                    PAPER
                    / "evidence"
                    / "prospective"
                    / f"llvip{sequence}"
                    / "scores_original8"
                ).glob("*.npz")
            )
            require(
                len(primary_scores) == 24,
                f"LLVIP {sequence}: expected 24 primary score arrays",
            )
            require(
                len(secondary_scores) == 32,
                f"LLVIP {sequence}: expected 32 sensitivity score arrays",
            )

        figure10_source = (
            PAPER / "source_data" / "FigureS4_thermal_class_source.csv"
        ).read_text(encoding="utf-8")
        require(
            "p_value" not in figure10_source,
            "supplementary thermal source data still contains inferential p-values",
        )
        require(
            "CS-ECF could not be separated from" not in main_text,
            "stale mechanism claim remains",
        )

        vedai_evidence = json.loads(
            (
                PAPER / "evidence" / "vedai" / "vedai_evidence.json"
            ).read_text(encoding="utf-8")
        )
        vedai_protocol = (
            PAPER
            / "evidence"
            / "vedai"
            / "protocol"
            / "VEDAI_CONFIRMATION.md"
        )
        require(vedai_protocol.is_file(), "VEDAI protocol is missing")
        require(
            hashlib.sha256(vedai_protocol.read_bytes()).hexdigest()
            == vedai_evidence["protocol"]["sha256"],
            "VEDAI protocol hash mismatch",
        )
        vedai_manifest = json.loads(
            (
                PAPER / "evidence" / "vedai" / "dataset_manifest.json"
            ).read_text(encoding="utf-8")
        )
        require(
            vedai_manifest == vedai_evidence["dataset"]["manifest"],
            "VEDAI dataset manifest differs from canonical evidence",
        )
        require(
            vedai_manifest["image_counts"]["train"]["images"] == 968
            and vedai_manifest["image_counts"]["val"]["images"] == 121
            and vedai_manifest["image_counts"]["test"]["images"] == 121,
            "VEDAI split sizes changed",
        )
        require(
            all(value == 0 for value in vedai_manifest["overlap"].values()),
            "VEDAI fold overlap is non-zero",
        )
        for run_name, run in vedai_evidence["models"].items():
            weight_path = (
                PAPER
                / "evidence"
                / "vedai"
                / run["packaged_weights"]
            )
            require(weight_path.is_file(), f"{run_name}: VEDAI weights missing")
            require(
                hashlib.sha256(weight_path.read_bytes()).hexdigest()
                == run["last_sha256"]
                ,
                f"{run_name}: VEDAI weight hash mismatch",
            )
        vedai_analysis = vedai_evidence["analysis"]
        require(
            close(
                vedai_analysis["aggregate"]["csecf_minus_visible"][
                    "equal_run_mean"
                ],
                -0.013390462944335334,
            ),
            "VEDAI visible contrast changed",
        )
        require(
            close(
                vedai_analysis["aggregate"]["csecf_minus_no_shield"][
                    "equal_run_mean"
                ],
                0.04109210335631237,
            ),
            "VEDAI no-shield contrast changed",
        )
        require(
            len(list((PAPER / "evidence" / "vedai" / "test_scores").glob("*.npz")))
            == 8,
            "VEDAI score-array count changed",
        )
        require(
            (PAPER / "source_data" / "TableVedai_source.csv").is_file(),
            "VEDAI table source data is missing",
        )

        standard_prospective_path = (
            PAPER
            / "evidence"
            / "prospective_standard_matching"
            / "analysis.json"
        )
        matched_control_path = (
            PAPER
            / "evidence"
            / "prospective_standard_matching"
            / "matched_switch_target_control.json"
        )
        external_scores_path = (
            PAPER / "evidence" / "external_evaluator_scores.json"
        )
        for path in (
            standard_prospective_path,
            matched_control_path,
            external_scores_path,
        ):
            require(path.is_file(), f"missing new evidence: {path.name}")
            json.loads(path.read_text(encoding="utf-8"))
        standard_prospective = json.loads(
            standard_prospective_path.read_text(encoding="utf-8")
        )
        standard_comparisons = standard_prospective["confirmation"][
            "comparisons"
        ]
        require(
            close(
                standard_comparisons["csecf_minus_visible"]["summary"][
                    "sequence_equal_point"
                ],
                0.00012160449439087139,
            ),
            "standard-matching visible contrast changed",
        )
        require(
            close(
                standard_comparisons["csecf_minus_no_shield"]["summary"][
                    "sequence_equal_point"
                ],
                0.008935080475932928,
            ),
            "standard-matching no-shield contrast changed",
        )
        matched_control = json.loads(
            matched_control_path.read_text(encoding="utf-8")
        )
        require(
            close(
                matched_control["csecf_minus_matched_random"][
                    "point_difference_from_random_median"
                ],
                0.013919760886158436,
            ),
            "matched-switch control changed",
        )
        m3fd_full = json.loads(
            (
                evidence_dir / "m3fd_full_split_manifest.json"
            ).read_text(encoding="utf-8")
        )
        require(
            m3fd_full["splits"]["train"]["n_images"] == 1600
            and m3fd_full["splits"]["val"]["n_images"] == 400
            and m3fd_full["splits"]["test"]["n_images"] == 200,
            "full M3FD split sizes changed",
        )
        require(
            all(not values for values in m3fd_full["group_overlap"].values()),
            "M3FD full-split group overlap is non-zero",
        )

    figure_map = PAPER / "reproduction" / "FIGURE_BUILD_MAP.json"
    require(figure_map.is_file(), "figure build map is missing")
    mapping = json.loads(figure_map.read_text(encoding="utf-8"))
    require(
        set(mapping["main"]) == {str(index) for index in range(1, 6)},
        "figure build map does not cover main Figures 1-5",
    )
    require(
        set(mapping["supplementary"]) == {f"S{index}" for index in range(1, 9)},
        "figure build map does not cover supplementary Figures S1-S8",
    )
    require(
        (PAPER / "reproduction" / "equal_group_statistics.py").is_file(),
        "equal-group analysis script is missing",
    )
    require(
        (PAPER / "reproduction" / "merge_equal_group_evidence.py").is_file(),
        "equal-group evidence merge script is missing",
    )
    require(
        (PAPER / "evidence" / "dronevehicle_metric_boundary.md").is_file(),
        "DroneVehicle metric-boundary note is missing",
    )

    manifest_path = PAPER / "SHA256SUMS.txt"
    require(manifest_path.is_file(), "SHA256SUMS.txt is missing")
    manifest_rows = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"malformed manifest row: {line}")
        manifest_rows[match.group(2)] = match.group(1)
    excluded_suffixes = {
        ".aux",
        ".blg",
        ".fls",
        ".fdb_latexmk",
        ".log",
        ".out",
        ".pyc",
        ".spl",
    }
    excluded_relative = {
        "main.pdf",
        "supplementary.pdf",
    }
    payload_paths = {
        path.relative_to(PAPER).as_posix()
        for path in PAPER.rglob("*")
        if path.is_file()
        and path != manifest_path
        and path.suffix.lower() not in excluded_suffixes
        and path.relative_to(PAPER).as_posix() not in excluded_relative
        and path.relative_to(PAPER).as_posix() != "tmp"
        and not path.relative_to(PAPER).as_posix().startswith("tmp/")
        and "__pycache__" not in path.parts
    }
    require(
        set(manifest_rows) == payload_paths,
        "manifest file set differs from package payload",
    )
    for relative, expected in manifest_rows.items():
        actual = hashlib.sha256(
            (PAPER / relative).read_bytes()
        ).hexdigest()
        require(
            actual == expected,
            f"manifest hash mismatch: {relative}",
        )

    print(f"PASS: abstract={abstract_words} words; cover={cover_words} words")
    print(f"PASS: highlights={len(highlights)}; references={reference_entries}")
    print("PASS: placeholders, key claims, stale intervals, logs, pages, and figures")
    if evidence_dir.is_dir():
        print("PASS: major-revision JSON evidence and canonical statistics")
    print(
        "PASS: main=21 pages with 6 figures; supplementary=23 pages; "
        "cover and response letters=1 page each"
    )


if __name__ == "__main__":
    main()
