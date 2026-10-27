"""Fetch LLVIP sequence 040 as an unseen cross-sequence detection holdout."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from remotezip import RemoteZip


URL = "https://hf-mirror.com/datasets/jsonhash/LLVIP/resolve/main/LLVIP.zip"
ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
LABEL_SOURCE = Path(r"D:\Code\_ssgf_external\llvip_fido_ann\AnnotationsTxt")
VISIBLE_SOURCE = Path(
    r"D:\Code\_ssgf_external\r2_clean\clean\raw_visible_train"
)
MANIFEST = (
    ROOT
    / "_deposit/ssgf-llvip-fusion-detection-v1/data/splits/clean_split_manifest.json"
)


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=None
    )
    parser.add_argument("--url", default=URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sequence", default="040")
    parser.add_argument("--fetch-visible", action="store_true")
    args = parser.parse_args()

    split = json.loads(MANIFEST.read_text(encoding="utf-8"))
    excluded = {
        stem
        for key in ("train", "validation", "test")
        for stem in split["split_stems"][key]
    }
    stems = [
        path.stem
        for path in sorted(LABEL_SOURCE.glob(f"{args.sequence}*.txt"))
        if any(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    ]
    overlap = sorted(set(stems) & excluded)
    if overlap:
        raise RuntimeError(f"holdout overlaps clean split: {overlap[:10]}")
    for stem in stems:
        if not (VISIBLE_SOURCE / f"{stem}.jpg").is_file() and not args.fetch_visible:
            raise FileNotFoundError(VISIBLE_SOURCE / f"{stem}.jpg")

    output = (
        args.out
        if args.out is not None
        else ROOT / f"data/erf_holdout_llvip{args.sequence}"
    ).resolve()
    for name in ("visible", "infrared", "labels"):
        (output / name).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        visible_source = VISIBLE_SOURCE / f"{stem}.jpg"
        if visible_source.is_file():
            link_or_copy(
                visible_source,
                output / "visible" / f"{stem}.jpg",
            )
        link_or_copy(
            LABEL_SOURCE / f"{stem}.txt",
            output / "labels" / f"{stem}.txt",
        )

    def fetch_member(archive, member, destination):
        temporary = destination.with_suffix(".jpg.part")
        for attempt in range(6):
            try:
                with archive.open(member) as source:
                    with temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                temporary.replace(destination)
                return
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 16))

    def fetch_chunk(worker_index, chunk):
        with RemoteZip(args.url) as archive:
            names = set(archive.namelist())
            missing = [
                stem
                for stem in chunk
                if f"LLVIP/infrared/train/{stem}.jpg" not in names
                or (
                    args.fetch_visible
                    and f"LLVIP/visible/train/{stem}.jpg" not in names
                )
            ]
            if missing:
                raise FileNotFoundError(
                    f"missing remote IR images: {missing[:10]}"
                )
            completed = 0
            for stem in chunk:
                for modality in (
                    ("visible",) if args.fetch_visible else ()
                ) + ("infrared",):
                    destination = output / modality / f"{stem}.jpg"
                    if destination.is_file() and destination.stat().st_size > 0:
                        continue
                    if destination.exists():
                        destination.unlink()
                    fetch_member(
                        archive,
                        f"LLVIP/{modality}/train/{stem}.jpg",
                        destination,
                    )
                completed += 1
                if completed % 10 == 0:
                    print(
                        f"worker {worker_index}: {completed}/{len(chunk)}",
                        flush=True,
                    )

    chunks = [
        stems[index:: args.workers] for index in range(max(1, args.workers))
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = [
            executor.submit(fetch_chunk, index, chunk)
            for index, chunk in enumerate(chunks)
            if chunk
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    for stem in stems:
        for modality in ("visible", "infrared", "labels"):
            suffix = ".txt" if modality == "labels" else ".jpg"
            destination = output / modality / f"{stem}{suffix}"
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError(f"image was not fetched: {destination}")

    payload = {
        "name": f"LLVIP-sequence-{args.sequence}-holdout",
        "source_url": args.url,
        "source_members": "LLVIP/infrared/train/*.jpg",
        "sequence": args.sequence,
        "counts": len(stems),
        "nonempty_labels": len(stems),
        "overlap_with_clean_split": 0,
        "stem_list_sha256": hashlib.sha256(
            ("\n".join(stems) + "\n").encode("utf-8")
        ).hexdigest(),
        "stems": stems,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print("WROTE", output / "manifest.json")


if __name__ == "__main__":
    main()
