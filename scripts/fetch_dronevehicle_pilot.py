"""Fetch a grouped DroneVehicle subset for external CDF validation."""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from remotezip import RemoteZip


TRAIN_URL = (
    "https://huggingface.co/datasets/McCheng/DroneVehicle/"
    "resolve/main/train.zip"
)
VAL_URL = (
    "https://huggingface.co/datasets/McCheng/DroneVehicle/"
    "resolve/main/val.zip"
)
OUTPUT = Path(r"D:\Code\_ssgf_external\dronevehicle_erf_pilot")
TRAIN_COUNT = 1200
WORKERS = 16
CLASSES = ["car", "truck", "bus", "van", "freight car"]


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def fetch_member(archive, member: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    if destination.exists():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
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


def xml_to_yolo(source: Path, destination: Path) -> None:
    root = ET.parse(source).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    rows = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name == "feright car":
            name = "freight car"
        if name not in CLASSES:
            continue
        polygon = obj.find("polygon")
        if polygon is not None:
            xs = [
                float(polygon.findtext(f"x{index}"))
                for index in range(1, 5)
            ]
            ys = [
                float(polygon.findtext(f"y{index}"))
                for index in range(1, 5)
            ]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        cls = CLASSES.index(name)
        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        rows.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with RemoteZip(TRAIN_URL) as train_archive:
        train_names = set(train_archive.namelist())
    with RemoteZip(VAL_URL) as val_archive:
        val_names = set(val_archive.namelist())
    train_stems = sorted(
        Path(name).stem
        for name in train_names
        if name.startswith("train/trainimg/") and name.endswith(".jpg")
    )
    val_stems = sorted(
        Path(name).stem
        for name in val_names
        if name.startswith("val/valimg/") and name.endswith(".jpg")
    )
    rng = random.Random(2031)
    selected_train_remote = sorted(rng.sample(train_stems, TRAIN_COUNT))
    val_groups = {}
    for stem in val_stems:
        group = (int(stem) - 1) // 100
        val_groups.setdefault(group, []).append(stem)
    calibration_groups = rng.sample(sorted(val_groups), 2)
    calibration_remote = [
        stem for group in sorted(calibration_groups)
        for stem in val_groups[group]
    ]
    test_remote = [
        stem for group, stems in sorted(val_groups.items())
        if group not in calibration_groups
        for stem in stems
    ]

    for directory in (
        "source/visible",
        "source/infrared",
        "source/xml",
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
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            "train",
            stem,
            f"train_{stem}",
            "train/trainimg",
            "train/trainimgr",
            "train/trainlabel",
        )
        for stem in selected_train_remote
    ] + [
        (
            "val",
            stem,
            f"val_{stem}",
            "val/valimg",
            "val/valimgr",
            "val/vallabel",
        )
        for stem in [*calibration_remote, *test_remote]
    ]

    def fetch_chunk(worker_index: int, chunk):
        with RemoteZip(TRAIN_URL) as train_archive, RemoteZip(VAL_URL) as val_archive:
            for index, (
                split,
                remote_stem,
                output_stem,
                visible_prefix,
                infrared_prefix,
                label_prefix,
            ) in enumerate(chunk, 1):
                archive = train_archive if split == "train" else val_archive
                fetch_member(
                    archive,
                    f"{visible_prefix}/{remote_stem}.jpg",
                    OUTPUT / "source/visible" / f"{output_stem}.jpg",
                )
                fetch_member(
                    archive,
                    f"{infrared_prefix}/{remote_stem}.jpg",
                    OUTPUT / "source/infrared" / f"{output_stem}.jpg",
                )
                fetch_member(
                    archive,
                    f"{label_prefix}/{remote_stem}.xml",
                    OUTPUT / "source/xml" / f"{output_stem}.xml",
                )
                if index % 10 == 0:
                    print(
                        f"worker {worker_index}: {index}/{len(chunk)}",
                        flush=True,
                    )

    chunks = [jobs[index::WORKERS] for index in range(WORKERS)]
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as executor:
        futures = [
            executor.submit(fetch_chunk, index, chunk)
            for index, chunk in enumerate(chunks)
            if chunk
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    for _, _, output_stem, _, _, _ in jobs:
        xml_to_yolo(
            OUTPUT / "source/xml" / f"{output_stem}.xml",
            OUTPUT / "source/labels" / f"{output_stem}.txt",
        )
    split_stems = {
        "train": [f"train_{stem}" for stem in selected_train_remote],
        "validation": [f"val_{stem}" for stem in calibration_remote],
        "test": [f"val_{stem}" for stem in test_remote],
    }
    targets = {"train": "train", "validation": "val", "test": "test"}
    for split, stems in split_stems.items():
        target = targets[split]
        for stem in stems:
            link_or_copy(
                OUTPUT / "source/visible" / f"{stem}.jpg",
                OUTPUT / f"det/images/{target}/{stem}.jpg",
            )
            link_or_copy(
                OUTPUT / "source/labels" / f"{stem}.txt",
                OUTPUT / f"det/labels/{target}/{stem}.txt",
            )
    for stem in test_stems:
        link_or_copy(
            OUTPUT / "source/visible" / f"{stem}.jpg",
            OUTPUT / "test_sources/visible" / f"{stem}.jpg",
        )
        link_or_copy(
            OUTPUT / "source/infrared" / f"{stem}.jpg",
            OUTPUT / "test_sources/infrared" / f"{stem}.jpg",
        )
        link_or_copy(
            OUTPUT / "source/labels" / f"{stem}.txt",
            OUTPUT / "test_sources/labels" / f"{stem}.txt",
        )
    data_yaml = (
        f"path: {(OUTPUT / 'det').as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES))
    )
    (OUTPUT / "det/data.yaml").write_text(data_yaml, encoding="utf-8")
    manifest = {
        "dataset": "DroneVehicle",
        "train_url": TRAIN_URL,
        "val_url": VAL_URL,
        "sampling_seed": 2031,
        "classes": CLASSES,
        "calibration_groups": calibration_groups,
        "counts": {
            split: len(stems) for split, stems in split_stems.items()
        },
        "stems": split_stems,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
