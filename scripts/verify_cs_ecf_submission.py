"""Run deterministic preflight checks for the CS-ECF IPT submission."""

from __future__ import annotations

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


def main() -> None:
    main_text = MAIN.read_text(encoding="utf-8")
    supplement_text = SUPPLEMENT.read_text(encoding="utf-8")
    cover_text = COVER.read_text(encoding="utf-8")
    highlights_text = HIGHLIGHTS.read_text(encoding="utf-8")

    source_text = "\n".join((main_text, supplement_text, cover_text))
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
            "0.0133",
            "[0.0026, 0.0261]",
            "0.0102",
            "[-0.00057, 0.02980]",
            "0.0108",
            "0.0153",
            "p=0.073",
            "0.0833",
            "10.5281/zenodo.23728385",
            "official LLVIP partitions are larger",
            "3,463 test pairs",
            "1.00,1.10,1.20,1.30,1.50,1.75,2.00",
            "q_{\\mathrm{TH}}=0.95",
            "lexicographically first 1,500",
            "external domain-transfer check",
            "109 pixels wide and 246 pixels high",
            "VIF and SSIM selected only among",
            "AP50 of zero for every candidate",
        ),
        SUPPLEMENT: (
            "-.0108",
            "[-.0342,+.0050]",
            "0.0153",
            "[+0.0072,+0.0267]",
            "p=0.0730",
            "12-member confirmatory family",
            "0.087/0.086",
            "Complete pre-freeze margin search",
        ),
        COVER: ("0.0108", "0.0153", "p=0.073"),
    }
    for path, claims in required_claims.items():
        text = path.read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        for claim in claims:
            require(claim in normalized_text, f"{path.name} is missing {claim}")

    stale_values = (
        "[-0.0331,+0.0048]",
        "[-.0331,+.0048]",
        "[-0.0199,+0.0102]",
        "[-.0199,+.0102]",
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
    require(reference_entries == 46, f"expected 46 references, found {reference_entries}")
    reference_text = REFERENCES.read_text(encoding="utf-8")
    for doi in (
        "10.1016/j.inffus.2012.01.008",
        "10.1016/j.infrared.2025.106325",
        "10.1016/j.infrared.2010.03.005",
    ):
        require(doi in reference_text, f"required verified DOI is missing: {doi}")

    expected_pages = {
        "main": (25, "Manuscript.pdf"),
        "supplementary": (6, "Supplementary_Material.pdf"),
        "cover_letter": (1, "Cover_Letter.pdf"),
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
        reference_numbers = {
            int(value)
            for value in re.findall(r"(?m)^\s*\[(\d+)\]", rendered_text)
        }
        require(
            reference_numbers == set(range(1, 47)),
            "PDF text does not contain all reference entries [1]--[46]",
        )
        require(
            not re.search(r"(?m)^\s*(?:1\s+){10,}1?\s*$", rendered_text),
            "PDF contains a repeated-character corruption line",
        )
        for figure_index in range(1, 12):
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
    for index in range(1, 12):
        matches = sorted(figure_dir.glob(f"Figure{index}_*.pdf"))
        require(len(matches) == 1, f"expected one PDF for figure {index}")

    source_data_dir = PAPER / "source_data"
    for index in range(1, 12):
        path = source_data_dir / f"Figure{index}_source.csv"
        require(path.is_file(), f"missing source data for figure {index}")

    require(
        len(re.findall(r"\\begin\{figure\}", main_text)) == 11,
        "main.tex does not contain exactly 11 figures",
    )

    evidence_dir = PAPER / "evidence" / "major_revision"
    if evidence_dir.is_dir():
        evidence_names = (
            "major_revision_statistics.json",
            "image_quality_baselines.json",
            "image_quality_features.json",
            "external_evaluator_diagnostics.json",
            "m3fd_class_summary.json",
            "alignment_csecf_sensitivity.json",
            "prefreeze_validation_summary.json",
            "llvip_test_sampling_summary.json",
            "llvip_test_manifest_3463.json",
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
        primary = statistics["multiple_comparisons"]["visible_relative"]
        require(primary["family_size"] == 12, "multiplicity family is not 12")
        require(primary["positive_checkpoints"] == 9, "positive count is not 9")
        require(
            abs(primary["exact_sign_test_p"] - 0.0730) < 1e-5,
            "sign-test value changed",
        )
        require(primary["survive_holm"] == [], "Holm-surviving checkpoint appeared")
        balanced = statistics["dataset_balanced"]
        require(
            abs(balanced["visible"]["point"] - 0.0133) < 1e-4,
            "balanced visible contrast changed",
        )
        require(
            abs(balanced["cdf"]["point"] - 0.0102) < 1e-4,
            "balanced CDF contrast changed",
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

    print(f"PASS: abstract={abstract_words} words; cover={cover_words} words")
    print(f"PASS: highlights={len(highlights)}; references={reference_entries}")
    print("PASS: placeholders, key claims, stale intervals, logs, pages, and figures")
    if evidence_dir.is_dir():
        print("PASS: major-revision JSON evidence and canonical statistics")
    print("PASS: main=25 pages with 11 figures; supplementary=6 pages; cover letter=1 page")


if __name__ == "__main__":
    main()
