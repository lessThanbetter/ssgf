"""Extract three fixed official DroneVehicle test groups from the local archive."""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


ARCHIVE = Path(r"D:\Code\_ssgf_external\dronevehicle_hf\test.zip")
ARCHIVE_SHA256 = (
    "94BF47E493E7A57FD5D900439F614F339E7B6C7181E4D781BCFEBAEB963FFD8E"
)
OUTPUT = Path(r"D:\Code\_ssgf_external\dronevehicle_test300")
CLASSES = ["car", "truck", "bus", "van", "freight car"]
GROUP_SEED = 2037
GROUP_COUNT = 3


def parse_xml(payload: bytes, collapse_classes: bool):
    root = ET.fromstring(payload)
    width = float(root.find("size").findtext("width"))
    height = float(root.find("size").findtext("height"))
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
        xs = [float(polygon.findtext(f"x{index}")) for index in range(1, 5)]
        ys = [float(polygon.findtext(f"y{index}")) for index in range(1, 5)]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        class_id = 0 if collapse_classes else CLASSES.index(name)
        rows.append(
            "{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}".format(
                cls=class_id,
                xc=(xmin + xmax) / 2 / width,
                yc=(ymin + ymax) / 2 / height,
                w=(xmax - xmin) / width,
                h=(ymax - ymin) / height,
            )
        )
    return "\n".join(rows) + ("\n" if rows else "")


def main():
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        stems = sorted(
            Path(name).stem
            for name in names
            if name.startswith("test/testimg/") and name.endswith(".jpg")
        )
        groups = {}
        for stem in stems:
            group = (int(stem) - 1) // 100
            groups.setdefault(group, []).append(stem)
        selected_groups = sorted(
            random.Random(GROUP_SEED).sample(sorted(groups), GROUP_COUNT)
        )
        selected_stems = [
            stem for group in selected_groups for stem in groups[group]
        ]

        for directory in (
            "visible",
            "infrared",
            "xml",
            "labels",
            "labels_vehicle",
        ):
            (OUTPUT / directory).mkdir(parents=True, exist_ok=True)

        for index, remote_stem in enumerate(selected_stems, 1):
            stem = f"test_{remote_stem}"
            members = {
                "visible": f"test/testimg/{remote_stem}.jpg",
                "infrared": f"test/testimgr/{remote_stem}.jpg",
                "xml": f"test/testlabel/{remote_stem}.xml",
            }
            for directory, member in members.items():
                destination = OUTPUT / directory / (
                    f"{stem}.xml" if directory == "xml" else f"{stem}.jpg"
                )
                destination.write_bytes(archive.read(member))
            xml_payload = archive.read(members["xml"])
            (OUTPUT / "labels" / f"{stem}.txt").write_text(
                parse_xml(xml_payload, collapse_classes=False),
                encoding="utf-8",
            )
            (OUTPUT / "labels_vehicle" / f"{stem}.txt").write_text(
                parse_xml(xml_payload, collapse_classes=True),
                encoding="utf-8",
            )
            if index % 25 == 0 or index == len(selected_stems):
                print(f"{index}/{len(selected_stems)}", flush=True)

    manifest = {
        "dataset": "DroneVehicle",
        "source_url": (
            "https://huggingface.co/datasets/McCheng/DroneVehicle/"
            "resolve/main/test.zip"
        ),
        "local_archive": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": ARCHIVE_SHA256,
        "official_split": "test",
        "selection": {
            "unit": "consecutive 100-frame group",
            "seed": GROUP_SEED,
            "groups": selected_groups,
        },
        "classes": CLASSES,
        "counts": {"test": len(selected_stems)},
        "stems": {
            "test": [f"test_{stem}" for stem in selected_stems]
        },
    }
    output_manifest = OUTPUT / "manifest.json"
    output_manifest.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"WROTE {len(selected_stems)} pairs; groups={selected_groups}; "
        f"manifest={output_manifest}"
    )


if __name__ == "__main__":
    main()
