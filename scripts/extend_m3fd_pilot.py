"""Extend the M3FD pilot without changing its held-out test groups."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from remotezip import RemoteZip


ROOT = Path(r"D:\Code\_ssgf_external\m3fd_erf_pilot")
LABEL_SOURCE = Path(
    r"D:\Code\_ssgf_external\m3fd_source\m3fd_labels\labels"
)
IR_URL = (
    "https://huggingface.co/datasets/nonameplease/M3FD_Detecion/"
    "resolve/main/ir.zip"
)
VI_URL = (
    "https://huggingface.co/datasets/nonameplease/M3FD_Detecion/"
    "resolve/main/vi.zip"
)
WORKERS = 16


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def one_class_label(source: Path, destination: Path) -> None:
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
        "--rebuild-only",
        action="store_true",
        help="Refresh labels and manifests without opening remote archives.",
    )
    args = parser.parse_args()
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    already_extended = manifest.get("name") == "M3FD-CDF-standard-2200"
    current_groups = {
        split: list(manifest["split_groups"][split])
        for split in ("train", "validation", "test")
    }
    all_groups = {
        int(path.stem) // 100 for path in LABEL_SOURCE.glob("*.txt")
    }
    people_by_group = defaultdict(int)
    stems_by_group = defaultdict(list)
    for path in LABEL_SOURCE.glob("*.txt"):
        group = int(path.stem) // 100
        stems_by_group[group].append(path.stem)
        people_by_group[group] += sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and line.split()[0] == "0"
        )
    if already_extended:
        new_train, new_validation = [], []
    else:
        remaining = sorted(
            all_groups
            - set(current_groups["train"])
            - set(current_groups["validation"])
            - set(current_groups["test"])
        )
        high = [group for group in remaining if people_by_group[group] >= 130]
        low = [group for group in remaining if people_by_group[group] < 130]
        rng = random.Random(2029)
        new_train = rng.sample(high, 5) + rng.sample(low, 3)
        high_left = [group for group in high if group not in new_train]
        low_left = [group for group in low if group not in new_train]
        new_validation = rng.sample(high_left, 1) + rng.sample(low_left, 1)
        current_groups["train"].extend(new_train)
        current_groups["validation"].extend(new_validation)

    split_stems = {}
    for split, groups in current_groups.items():
        split_stems[split] = [
            stem
            for group in sorted(groups)
            for stem in sorted(stems_by_group[group])
        ]
    new_stems = [] if already_extended else [
        stem
        for group in sorted(new_train + new_validation)
        for stem in sorted(stems_by_group[group])
    ]
    for stem in new_stems:
        link_or_copy(
            LABEL_SOURCE / f"{stem}.txt",
            ROOT / "source/labels" / f"{stem}.txt",
        )

    def fetch_chunk(worker_index: int, chunk: list[str]) -> None:
        with RemoteZip(IR_URL) as infrared, RemoteZip(VI_URL) as visible:
            for index, stem in enumerate(chunk, 1):
                fetch_member(
                    infrared,
                    f"ir/{stem}.png",
                    ROOT / "source/infrared" / f"{stem}.png",
                )
                fetch_member(
                    visible,
                    f"vi/{stem}.png",
                    ROOT / "source/visible" / f"{stem}.png",
                )
                if index % 10 == 0:
                    print(
                        f"worker {worker_index}: {index}/{len(chunk)}",
                        flush=True,
                    )

    if not args.rebuild_only:
        chunks = [
            new_stems[index::WORKERS] for index in range(WORKERS)
        ]
        with concurrent.futures.ThreadPoolExecutor(WORKERS) as executor:
            futures = [
                executor.submit(fetch_chunk, index, chunk)
                for index, chunk in enumerate(chunks)
                if chunk
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    targets = {"train": "train", "validation": "val", "test": "test"}
    for split, stems in split_stems.items():
        target = targets[split]
        for stem in stems:
            link_or_copy(
                ROOT / "source/visible" / f"{stem}.png",
                ROOT / f"det/images/{target}/{stem}.png",
            )
            shutil.copy2(
                ROOT / "source/labels" / f"{stem}.txt",
                ROOT / f"det/labels/{target}/{stem}.txt",
            )

    data_yaml = (
        f"path: {(ROOT / 'det').as_posix()}\n"
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
    (ROOT / "det/data.yaml").write_text(data_yaml, encoding="utf-8")

    payload = {
        **manifest,
        "name": "M3FD-CDF-standard-2200",
        "split_groups": current_groups,
        "counts": {
            split: len(stems) for split, stems in split_stems.items()
        },
        "people": {
            split: sum(people_by_group[group] for group in groups)
            for split, groups in current_groups.items()
        },
        "stems": split_stems,
        "extension_seed": 2029,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    print(json.dumps(payload["people"], indent=2))
    print("WROTE", manifest_path)


if __name__ == "__main__":
    main()
