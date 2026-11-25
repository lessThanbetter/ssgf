"""Prepare an official-fold VEDAI visible/IR object-detection subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


CLASS_MAP = {
    1: ("car", 0),
    2: ("truck", 1),
    4: ("tractor", 2),
    5: ("camping car", 3),
    9: ("van", 4),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fold(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"empty fold list: {path}")
    return values


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        target.write_bytes(source.read_bytes())


def parse_label(path: Path) -> tuple[list[str], Counter]:
    rows = []
    counts = Counter()
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 14:
            continue
        class_id = int(float(values[3]))
        if class_id not in CLASS_MAP:
            continue
        class_name, mapped_id = CLASS_MAP[class_id]
        xs = [float(value) for value in values[6:10]]
        ys = [float(value) for value in values[10:14]]
        points = []
        for x, y in zip(xs, ys):
            x = min(512.0, max(0.0, x))
            y = min(512.0, max(0.0, y))
            points.extend((x / 512.0, y / 512.0))
        rows.append(
            f"{mapped_id} "
            + " ".join(f"{value:.8f}" for value in points)
        )
        counts[class_name] += 1
    return rows, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vedai-root",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\vedai_official"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\vedai_fold01"),
    )
    args = parser.parse_args()
    root = args.vedai_root.resolve()
    out = args.out.resolve()
    image_root = root / "images" / "Vehicules512"
    annotation_root = root / "annotations" / "Annotations512"

    fold_test = {
        fold: read_fold(annotation_root / f"fold{fold:02d}test.txt")
        for fold in range(1, 11)
    }
    splits = {
        "train": [
            stem
            for fold in range(3, 11)
            for stem in fold_test[fold]
        ],
        "val": fold_test[2],
        "test": fold_test[1],
    }
    split_sets = {name: set(values) for name, values in splits.items()}
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        raise ValueError("official VEDAI fold partitions overlap")

    class_counts = {}
    image_counts = {}
    for split, stems in splits.items():
        counts = Counter()
        populated = 0
        for stem in stems:
            visible = image_root / f"{stem}_co.png"
            infrared = image_root / f"{stem}_ir.png"
            annotation = annotation_root / f"{stem}.txt"
            for path in (visible, infrared, annotation):
                if not path.is_file():
                    raise FileNotFoundError(path)
            link_or_copy(visible, out / "images" / split / f"{stem}.png")
            link_or_copy(infrared, out / "sources" / split / "infrared" / f"{stem}.png")
            link_or_copy(visible, out / "sources" / split / "visible" / f"{stem}.png")
            rows, row_counts = parse_label(annotation)
            label_path = out / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                "\n".join(rows) + ("\n" if rows else ""),
                encoding="utf-8",
            )
            counts.update(row_counts)
            populated += bool(rows)
        class_counts[split] = dict(counts)
        image_counts[split] = {
            "images": len(stems),
            "images_with_targets": populated,
        }

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: car",
                "  1: truck",
                "  2: tractor",
                "  3: camping car",
                "  4: van",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "source": "official VEDAI 512 multi-spectral dataset",
        "source_root": str(root),
        "split_rule": {
            "train": "union of fold03test through fold10test",
            "validation": "fold02test",
            "test": "fold01test",
            "selection": "lexicographically first official test fold",
        },
        "class_map": {
            str(key): {"name": value[0], "yolo_id": value[1]}
            for key, value in CLASS_MAP.items()
        },
        "image_counts": image_counts,
        "class_counts": class_counts,
        "split_stems": {
            split: sorted(values)
            for split, values in splits.items()
        },
        "fold_file_hashes": {
            name: sha256(annotation_root / name)
            for name in (
                [f"fold{fold:02d}test.txt" for fold in range(1, 11)]
            )
        },
        "hashes": {
            "Vehicules512.tar": sha256(root / "Vehicules512.tar"),
            "Annotations512.tar": sha256(root / "Annotations512.tar"),
            "data.yaml": sha256(data_yaml),
        },
        "overlap": {
            "train_val": len(split_sets["train"] & split_sets["val"]),
            "train_test": len(split_sets["train"] & split_sets["test"]),
            "val_test": len(split_sets["val"] & split_sets["test"]),
        },
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
