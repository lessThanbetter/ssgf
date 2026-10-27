"""Fetch three official DroneVehicle test groups for external testing."""

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


URL = (
    "https://hf-mirror.com/datasets/McCheng/DroneVehicle/"
    "resolve/main/test.zip"
)
OUTPUT = Path(r"D:\Code\_ssgf_external\dronevehicle_test300")
CLASSES = ["car", "truck", "bus", "van", "freight car"]
WORKERS = 12


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


def open_remote_zip(url: str) -> RemoteZip:
    for attempt in range(6):
        try:
            return RemoteZip(url, support_suffix_range=False)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 16))


def xml_to_yolo(source: Path, destination: Path) -> None:
    root = ET.parse(source).getroot()
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
        rows.append(
            "{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}".format(
                cls=CLASSES.index(name),
                xc=(xmin + xmax) / 2 / width,
                yc=(ymin + ymax) / 2 / height,
                w=(xmax - xmin) / width,
                h=(ymax - ymin) / height,
            )
        )
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    with open_remote_zip(URL) as archive:
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
        selected_groups = random.Random(2037).sample(sorted(groups), 3)
        remote_stems = [
            stem for group in sorted(selected_groups)
            for stem in groups[group]
        ]

    for directory in ("visible", "infrared", "xml", "labels"):
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)

    def fetch_chunk(worker_index: int, chunk: list[str]) -> None:
        with open_remote_zip(URL) as worker_archive:
            for index, remote_stem in enumerate(chunk, 1):
                stem = f"test_{remote_stem}"
                fetch_member(
                    worker_archive,
                    f"test/testimg/{remote_stem}.jpg",
                    OUTPUT / "visible" / f"{stem}.jpg",
                )
                fetch_member(
                    worker_archive,
                    f"test/testimgr/{remote_stem}.jpg",
                    OUTPUT / "infrared" / f"{stem}.jpg",
                )
                fetch_member(
                    worker_archive,
                    f"test/testlabel/{remote_stem}.xml",
                    OUTPUT / "xml" / f"{stem}.xml",
                )
                if index % 10 == 0:
                    print(
                        f"worker {worker_index}: {index}/{len(chunk)}",
                        flush=True,
                    )

    chunks = [remote_stems[index::WORKERS] for index in range(WORKERS)]
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as executor:
        futures = [
            executor.submit(fetch_chunk, index, chunk)
            for index, chunk in enumerate(chunks)
            if chunk
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    test_stems = [f"test_{stem}" for stem in remote_stems]
    for stem in test_stems:
        xml_to_yolo(
            OUTPUT / "xml" / f"{stem}.xml",
            OUTPUT / "labels" / f"{stem}.txt",
        )
    manifest = {
        "dataset": "DroneVehicle",
        "source_url": URL,
        "official_split": "test",
        "selected_groups": selected_groups,
        "classes": CLASSES,
        "counts": {"test": len(test_stems)},
        "stems": {"test": test_stems},
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"WROTE {len(test_stems)} pairs")


if __name__ == "__main__":
    main()
