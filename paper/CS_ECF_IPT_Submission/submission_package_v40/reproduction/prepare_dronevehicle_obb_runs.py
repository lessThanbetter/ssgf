"""Convert the held-out DroneVehicle groups to native OBB training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


CLASSES = ("car", "truck", "bus", "van", "freight car")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        destination.write_bytes(source.read_bytes())


def polygon_label(source: Path) -> str:
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
        if polygon is None:
            continue
        xs = [
            float(polygon.findtext(f"x{index}")) / width
            for index in range(1, 5)
        ]
        ys = [
            float(polygon.findtext(f"y{index}")) / height
            for index in range(1, 5)
        ]
        coordinates = " ".join(
            f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys)
        )
        rows.append(f"{CLASSES.index(name)} {coordinates}")
    return "\n".join(rows) + ("\n" if rows else "")


def stable_seed(stems: list[str]) -> int:
    digest = hashlib.sha256(
        ("\n".join(sorted(stems)) + "\n").encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\dronevehicle_val300"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\dronevehicle_val300\det_obb"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()

    groups: dict[int, list[str]] = {}
    for path in sorted((root / "visible").glob("*.jpg")):
        match = re.fullmatch(r"val_(\d+)", path.stem)
        if match is None:
            raise ValueError(f"unexpected stem: {path.stem}")
        group = (int(match.group(1)) - 1) // 100
        groups.setdefault(group, []).append(path.stem)
    if sorted(groups) != [4, 12, 13]:
        raise ValueError(f"unexpected DroneVehicle groups: {sorted(groups)}")

    splits = {
        "run1": {
            "train": [4, 12],
            "val": [13],
        },
        "run2": {
            "train": [4, 13],
            "val": [12],
        },
    }
    for run, spec in splits.items():
        for split, selected_groups in spec.items():
            for group in selected_groups:
                for stem in groups[group]:
                    link_or_copy(
                        root / "visible" / f"{stem}.jpg",
                        out / "images" / f"{split}_{run}" / f"{stem}.jpg",
                    )
                    label_path = (
                        out / "labels" / f"{split}_{run}" / f"{stem}.txt"
                    )
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text(
                        polygon_label(root / "xml" / f"{stem}.xml"),
                        encoding="utf-8",
                    )
        data_yaml = (
            f"path: {out.as_posix()}\n"
            f"train: images/train_{run}\n"
            f"val: images/val_{run}\n"
            "names:\n"
            + "".join(
                f"  {index}: {name}\n"
                for index, name in enumerate(CLASSES)
            )
        )
        (out / f"data_{run}.yaml").write_text(data_yaml, encoding="utf-8")

    manifest = {
        "source": str(root),
        "groups": {
            str(group): len(stems) for group, stems in groups.items()
        },
        "splits": splits,
        "seeds": {
            run: stable_seed(
                [
                    stem
                    for split in ("train", "val")
                    for group in spec[split]
                    for stem in groups[group]
                ]
            )
            for run, spec in splits.items()
        },
        "classes": list(CLASSES),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
