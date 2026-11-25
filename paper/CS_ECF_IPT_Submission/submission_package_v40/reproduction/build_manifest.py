"""Create a deterministic SHA-256 manifest for the submission package."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
MANIFEST = PACKAGE / "SHA256SUMS.txt"
EXCLUDED_SUFFIXES = {
    ".aux",
    ".blg",
    ".fls",
    ".fdb_latexmk",
    ".log",
    ".out",
    ".pyc",
    ".spl",
}
EXCLUDED_RELATIVE = {
    "main.pdf",
    "supplementary.pdf",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file():
            continue
        if path == MANIFEST or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE).as_posix()
        if relative == "tmp" or relative.startswith("tmp/"):
            continue
        if relative in EXCLUDED_RELATIVE:
            continue
        rows.append(f"{digest(path)}  {relative}")
    MANIFEST.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"WROTE {MANIFEST} ({len(rows)} files)")


if __name__ == "__main__":
    main()
