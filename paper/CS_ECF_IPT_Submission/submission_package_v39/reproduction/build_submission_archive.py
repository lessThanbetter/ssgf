"""Build the clean v39 ZIP from the manifest-listed payload."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
MANIFEST = PACKAGE / "SHA256SUMS.txt"
ARCHIVE = PACKAGE.parent / "CS_ECF_IPT_submission_2026-11-26_v39.zip"
SIDECAR = ARCHIVE.with_name(ARCHIVE.name + ".sha256")


def main() -> None:
    names = [
        line.split("  ", 1)[1]
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with zipfile.ZipFile(
        ARCHIVE,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in sorted(["SHA256SUMS.txt", *names]):
            path = PACKAGE / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, arcname=relative)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    SIDECAR.write_text(
        f"{digest}  {ARCHIVE.name}\n",
        encoding="utf-8",
    )
    print(f"WROTE {ARCHIVE}")
    print(f"WROTE {SIDECAR}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
