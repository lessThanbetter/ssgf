"""Fetch the fixed 300-frame LLVIP sample for PIA confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE = SCRIPT_PATH.parents[1]
REPO_ROOT = next(
    (
        path
        for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
        if (path / "scripts" / "fetch_llvip_holdout_040.py").is_file()
    ),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("Repository root containing LLVIP helpers was not found")
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from extract_llvip_sources import (  # noqa: E402
    LLVIP_URL,
    fetch_member,
    resolve_url,
)


SEQUENCES = ("080", "090", "100")
LIMIT = 300
LABEL_SOURCE = Path(
    r"D:\Code\_ssgf_external\llvip_fido_ann\AnnotationsTxt"
)
DATA_ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf\data")


def sha256_lines(values: list[str]) -> str:
    payload = ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nonempty_stems(sequence: str) -> list[str]:
    stems = []
    for path in sorted(LABEL_SOURCE.glob(f"{sequence}*.txt")):
        if any(line.strip() for line in path.read_text(encoding="utf-8").splitlines()):
            stems.append(path.stem)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=LLVIP_URL)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    selected = {
        sequence: nonempty_stems(sequence)[:LIMIT]
        for sequence in SEQUENCES
    }
    for sequence, stems in selected.items():
        if len(stems) != LIMIT:
            raise RuntimeError(
                f"{sequence}: expected {LIMIT} stems, found {len(stems)}"
            )
    wanted = set().union(*map(set, selected.values()))

    print("[fetch-pia] reading remote ZIP directory", flush=True)
    with RemoteZip(args.url, support_suffix_range=True) as archive:
        members = {}
        for info in archive.infolist():
            if not (
                info.filename.startswith("LLVIP/visible/train/")
                or info.filename.startswith("LLVIP/infrared/train/")
            ):
                continue
            stem = Path(info.filename).stem
            if stem not in wanted:
                continue
            modality = info.filename.split("/")[1]
            members[(stem, modality)] = info
    expected = len(wanted) * 2
    if len(members) != expected:
        found = {(stem, modality) for stem, modality in members}
        missing = [
            (stem, modality)
            for stem in sorted(wanted)
            for modality in ("visible", "infrared")
            if (stem, modality) not in found
        ]
        raise RuntimeError(
            f"found {len(members)} source members, expected {expected}; "
            f"missing={missing[:10]}"
        )

    final_url = resolve_url(args.url)
    jobs = []
    for sequence, stems in selected.items():
        output = args.data_root / f"prospective_pia_{sequence}"
        for stem in stems:
            for modality in ("visible", "infrared"):
                destination = output / modality / f"{stem}.jpg"
                jobs.append((members[(stem, modality)], destination))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(fetch_member, final_url, info, destination): destination
            for info, destination in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"[fetch-pia] {index}/{len(futures)}", flush=True)

    for sequence, stems in selected.items():
        output = args.data_root / f"prospective_pia_{sequence}"
        for stem in stems:
            for modality in ("visible", "infrared"):
                path = output / modality / f"{stem}.jpg"
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"missing fetched image: {path}")
            label = LABEL_SOURCE / f"{stem}.txt"
            destination = output / "labels" / label.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                destination.write_bytes(label.read_bytes())
        payload = {
            "name": f"LLVIP-sequence-{sequence}-PIA-confirmation",
            "source_url": args.url,
            "sequence": sequence,
            "selection_rule": (
                "First 300 lexicographically ordered stems with non-empty "
                "labels, fixed before candidate or detector scoring."
            ),
            "counts": len(stems),
            "stems": stems,
            "stem_list_sha256": sha256_lines(stems),
        }
        (output / "manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    (PACKAGE / "evidence" / "prospective_pia_confirmation"
     / "sample_manifest.json").write_text(
        json.dumps(
            {
                "sequences": list(SEQUENCES),
                "limit_per_sequence": LIMIT,
                "total_images": LIMIT * len(SEQUENCES),
                "selection_rule": (
                    "First 300 lexicographically ordered stems with non-empty "
                    "labels in each sequence."
                ),
                "stems": selected,
                "stem_list_sha256": {
                    sequence: sha256_lines(stems)
                    for sequence, stems in selected.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("WROTE sample_manifest.json", flush=True)


if __name__ == "__main__":
    main()
