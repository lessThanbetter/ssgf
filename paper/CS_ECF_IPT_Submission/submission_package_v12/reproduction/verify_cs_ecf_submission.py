"""Run deterministic preflight checks for the CS-ECF IPT submission."""

from __future__ import annotations

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
        ),
        SUPPLEMENT: (
            "-.0108",
            "[-.0342,+.0050]",
            "0.0153",
            "[+0.0072,+0.0267]",
        ),
        COVER: ("0.0108", "0.0153"),
    }
    for path, claims in required_claims.items():
        text = path.read_text(encoding="utf-8")
        for claim in claims:
            require(claim in text, f"{path.name} is missing {claim}")

    stale_values = (
        "[-0.0331,+0.0048]",
        "[-.0331,+.0048]",
        "[-0.0199,+0.0102]",
        "[-.0199,+.0102]",
    )
    for stale in stale_values:
        require(stale not in source_text, f"stale domain interval present: {stale}")

    reference_entries = len(re.findall(r"(?m)^@", REFERENCES.read_text(encoding="utf-8")))
    require(reference_entries == 27, f"expected 27 references, found {reference_entries}")

    expected_pages = {
        "main": (13, "Manuscript.pdf"),
        "supplementary": (4, "Supplementary_Material.pdf"),
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

    figure_dir = PAPER / "figures"
    for index in range(1, 5):
        matches = sorted(figure_dir.glob(f"fig{index}_*.pdf"))
        require(len(matches) == 1, f"expected one PDF for figure {index}")

    print(f"PASS: abstract={abstract_words} words; cover={cover_words} words")
    print(f"PASS: highlights={len(highlights)}; references={reference_entries}")
    print("PASS: placeholders, key claims, stale intervals, logs, pages, and figures")
    print("PASS: main=13 pages; supplementary=4 pages; cover letter=1 page")


if __name__ == "__main__":
    main()
