"""Fetch a grouped 1,200-pair M3FD pilot for an external CDF evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from remotezip import RemoteZip


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
IR_URL = (
    "https://huggingface.co/datasets/nonameplease/M3FD_Detecion/"
    "resolve/main/ir.zip"
)
VI_URL = (
    "https://huggingface.co/datasets/nonameplease/M3FD_Detecion/"
    "resolve/main/vi.zip"
)
LABEL_SOURCE = Path(
    r"D:\Code\_ssgf_external\m3fd_source\m3fd_labels\labels"
)


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def one_class_label(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "0":
            rows.append("0 " + " ".join(parts[1:5]))
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def fetch_member(archive, member: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    if destination.exists():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".png.part")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\m3fd_erf_pilot"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.out.resolve()

    group_stats = defaultdict(lambda: {"stems": [], "people": 0})
    for label_path in sorted(LABEL_SOURCE.glob("*.txt")):
        group = int(label_path.stem) // 100
        group_stats[group]["stems"].append(label_path.stem)
        group_stats[group]["people"] += sum(
            1
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and line.split()[0] == "0"
        )
    high = [group for group, value in group_stats.items() if value["people"] >= 130]
    low = [group for group, value in group_stats.items() if value["people"] < 130]
    rng = random.Random(2028)
    train_groups = rng.sample(high, 6) + rng.sample(low, 2)
    validation_groups = rng.sample(
        [group for group in high if group not in train_groups], 1
    ) + rng.sample([group for group in low if group not in train_groups], 1)
    test_groups = rng.sample(
        [
            group
            for group in high
            if group not in train_groups + validation_groups
        ],
        1,
    ) + rng.sample(
        [
            group
            for group in low
            if group not in train_groups + validation_groups
        ],
        1,
    )
    splits = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    stems_by_split = {
        split: [
            stem
            for group in sorted(groups)
            for stem in sorted(group_stats[group]["stems"])
        ]
        for split, groups in splits.items()
    }
    all_stems = [
        stem for split in ("train", "validation", "test")
        for stem in stems_by_split[split]
    ]
    if len(all_stems) != 1200:
        raise RuntimeError(f"expected 1200 stems, got {len(all_stems)}")

    for directory in (
        "source/visible",
        "source/infrared",
        "source/labels",
        "det/images/train",
        "det/images/val",
        "det/images/test",
        "det/labels/train",
        "det/labels/val",
        "det/labels/test",
        "test_sources/visible",
        "test_sources/infrared",
        "test_sources/labels",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for stem in all_stems:
        link_or_copy(
            LABEL_SOURCE / f"{stem}.txt",
            output / "source/labels" / f"{stem}.txt",
        )

    def fetch_chunk(worker_index: int, chunk: list[str]) -> None:
        with RemoteZip(IR_URL) as infrared, RemoteZip(VI_URL) as visible:
            for index, stem in enumerate(chunk, 1):
                fetch_member(
                    infrared,
                    f"ir/{stem}.png",
                    output / "source/infrared" / f"{stem}.png",
                )
                fetch_member(
                    visible,
                    f"vi/{stem}.png",
                    output / "source/visible" / f"{stem}.png",
                )
                if index % 10 == 0:
                    print(
                        f"worker {worker_index}: {index}/{len(chunk)}",
                        flush=True,
                    )

    chunks = [
        all_stems[index:: args.workers] for index in range(args.workers)
    ]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
        futures = [
            executor.submit(fetch_chunk, index, chunk)
            for index, chunk in enumerate(chunks)
            if chunk
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    split_directory = {"train": "train", "validation": "val", "test": "test"}
    for split, stems in stems_by_split.items():
        target_split = split_directory[split]
        for stem in stems:
            link_or_copy(
                output / "source/visible" / f"{stem}.png",
                output / f"det/images/{target_split}/{stem}.png",
            )
            link_or_copy(
                output / "source/labels" / f"{stem}.txt",
                output / f"det/labels/{target_split}/{stem}.txt",
            )
    for stem in stems_by_split["test"]:
        link_or_copy(
            output / "source/visible" / f"{stem}.png",
            output / "test_sources/visible" / f"{stem}.png",
        )
        link_or_copy(
            output / "source/infrared" / f"{stem}.png",
            output / "test_sources/infrared" / f"{stem}.png",
        )
        one_class_label(
            output / "source/labels" / f"{stem}.txt",
            output / "test_sources/labels" / f"{stem}.txt",
        )

    data_yaml = (
        f"path: {(output / 'det').as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: people\n"
        "  1: car\n"
        "  2: bus\n"
        "  3: motorcycle\n"
        "  4: lamp\n"
        "  5: truck\n"
    )
    (output / "det/data.yaml").write_text(data_yaml, encoding="utf-8")
    payload = {
        "dataset": "M3FD",
        "url_ir": IR_URL,
        "url_visible": VI_URL,
        "grouping": "int(stem) // 100",
        "random_seed": 2028,
        "split_groups": splits,
        "counts": {
            split: len(stems) for split, stems in stems_by_split.items()
        },
        "people": {
            split: sum(
                group_stats[group]["people"] for group in groups
            )
            for split, groups in splits.items()
        },
        "stems": stems_by_split,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    print(json.dumps(payload["people"], indent=2))
    print("WROTE", manifest)


if __name__ == "__main__":
    main()
