"""Prepare the fixed LLVIP 1350/150 clean split used by every evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np


def select_stems(label_dir: Path, count: int) -> list[str]:
    stems = []
    for path in sorted(label_dir.glob("*.txt")):
        if any(line.strip() for line in path.read_text(encoding="utf-8").splitlines()):
            stems.append(path.stem)
        if len(stems) == count:
            break
    if len(stems) != count:
        raise SystemExit(f"found only {len(stems)} non-empty labels, need {count}")
    return stems


def extract_images(archives: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename).name
                target = destination / name
                if target.exists():
                    continue
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_split_manifest(
    out: Path,
    selected: Sequence[str],
    train_stems: Sequence[str],
    val_stems: Sequence[str],
    test_stems: Sequence[str],
    annotation_dir: Path,
    test_label_dir: Path,
) -> None:
    selected_set = set(selected)
    train_set = set(train_stems)
    val_set = set(val_stems)
    test_set = set(test_stems)
    partition = {
        name: len(stems)
        for name, stems in (
            ("selected", selected),
            ("train", train_stems),
            ("validation", val_stems),
            ("test", test_stems),
        )
    }
    overlap = {
        "train_test": len(train_set & test_set),
        "validation_test": len(val_set & test_set),
        "train_validation": len(train_set & val_set),
    }
    label_counts: dict[str, int] = {}
    for name, stems, labels_root in (
        ("train", train_stems, annotation_dir),
        ("validation", val_stems, annotation_dir),
        ("test", test_stems, test_label_dir),
    ):
        label_counts[name] = sum(
            sum(bool(line.strip()) for line in (labels_root / f"{stem}.txt").read_text(
                encoding="utf-8"
            ).splitlines())
            for stem in stems
        )
    if selected_set != train_set | val_set:
        raise SystemExit("selected stems do not equal the train and validation union")
    if any(overlap.values()):
        raise SystemExit(f"non-independent split detected: {overlap}")
    manifest: dict[str, object] = {
        "dataset": "LLVIP",
        "selection": "lexicographically first 1500 non-empty LLVIP train labels",
        "counts": partition,
        "label_counts": label_counts,
        "overlap": overlap,
        "stem_list_sha256": {
            name: hashlib.sha256(
                ("\n".join(stems) + "\n").encode("utf-8")
            ).hexdigest()
            for name, stems in (
                ("train", train_stems),
                ("validation", val_stems),
                ("test", test_stems),
            )
        },
        "split_stems": {
            "train": list(train_stems),
            "validation": list(val_stems),
            "test": list(test_stems),
        },
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"manifest counts={partition} label_counts={label_counts} "
        f"overlap={overlap}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--archives", nargs="+", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--selected", type=int, default=1500)
    parser.add_argument("--train", type=int, default=1350)
    parser.add_argument("--test-stems-npz")
    parser.add_argument("--test-label-out")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Validate existing split directories and rewrite split_manifest.json",
    )
    args = parser.parse_args()

    annotations = Path(args.annotations)
    out = Path(args.out)
    if args.manifest_only:
        if not args.test_stems_npz or not args.test_label_out:
            raise SystemExit(
                "--test-stems-npz and --test-label-out are required with "
                "--manifest-only"
            )
        train_stems = sorted(path.stem for path in (out / "images" / "train").glob("*.jpg"))
        val_stems = sorted(path.stem for path in (out / "images" / "val").glob("*.jpg"))
        reference = np.load(args.test_stems_npz, allow_pickle=True)
        test_stems = [str(stem) for stem in reference["stems"]]
        if len(train_stems) != args.train:
            raise SystemExit(
                f"found {len(train_stems)} train images, expected {args.train}"
            )
        if len(val_stems) != args.selected - args.train:
            raise SystemExit(
                f"found {len(val_stems)} validation images, expected "
                f"{args.selected - args.train}"
            )
        if len(test_stems) != 300:
            raise SystemExit(f"found {len(test_stems)} test stems, expected 300")
        write_split_manifest(
            out,
            [*train_stems, *val_stems],
            train_stems,
            val_stems,
            test_stems,
            annotations,
            Path(args.test_label_out),
        )
        return

    raw_images = out / "raw_visible_train"
    if not args.archives:
        raise SystemExit("--archives is required unless --manifest-only is used")
    extract_images([Path(item) for item in args.archives], raw_images)

    selected = select_stems(annotations, args.selected)
    train_stems = selected[:args.train]
    val_stems = selected[args.train:]
    for split, stems in (("train", train_stems), ("val", val_stems)):
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            source_image = raw_images / f"{stem}.jpg"
            if not source_image.is_file():
                raise SystemExit(f"missing image {source_image}")
            link_or_copy(source_image, out / "images" / split / f"{stem}.jpg")
            shutil.copy2(
                annotations / f"{stem}.txt",
                out / "labels" / split / f"{stem}.txt",
            )

    yaml = (
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n"
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    test_stems = []
    if args.test_stems_npz:
        if not args.test_label_out:
            raise SystemExit("--test-label-out is required with --test-stems-npz")
        reference = np.load(args.test_stems_npz, allow_pickle=True)
        test_stems = [str(stem) for stem in reference["stems"]]
        test_label_out = Path(args.test_label_out)
        test_label_out.mkdir(parents=True, exist_ok=True)
        for stem in test_stems:
            shutil.copy2(
                annotations / f"{stem}.txt",
                test_label_out / f"{stem}.txt",
            )
    write_split_manifest(
        out,
        selected,
        train_stems,
        val_stems,
        test_stems,
        annotations,
        Path(args.test_label_out) if args.test_label_out else annotations,
    )
    print(
        f"selected={len(selected)} train={len(train_stems)} "
        f"val={len(val_stems)} first={selected[0]} last={selected[-1]}"
    )


if __name__ == "__main__":
    main()
