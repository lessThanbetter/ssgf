"""Document and verify the complete M3FD train/validation/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_lines(lines: list[str]) -> str:
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\m3fd_erf_pilot\det"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            r"D:\Code\_server_handoff_v2\ssgf\paper\CS_ECF_IPT_Submission"
            r"\submission_package_v24\evidence\major_revision"
            r"\m3fd_full_split_manifest.json"
        ),
    )
    args = parser.parse_args()

    split_records = {}
    split_stems = {}
    for split in ("train", "val", "test"):
        image_root = args.dataset_root / "images" / split
        label_root = args.dataset_root / "labels" / split
        images = sorted(
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(
            path for path in label_root.iterdir() if path.suffix == ".txt"
        )
        image_stems = [path.stem for path in images]
        label_stems = [path.stem for path in labels]
        if image_stems != label_stems:
            missing_labels = sorted(set(image_stems) - set(label_stems))
            missing_images = sorted(set(label_stems) - set(image_stems))
            raise ValueError(
                f"{split}: image/label stems differ; "
                f"missing labels={missing_labels[:10]}, "
                f"missing images={missing_images[:10]}"
            )

        group_members = {}
        for stem in image_stems:
            group = stem[:3]
            group_members.setdefault(group, []).append(stem)
        groups = {}
        for group in sorted(group_members):
            stems = sorted(group_members[group])
            groups[group] = {
                "n_images": len(stems),
                "stem_digest": digest_lines(stems),
                "stems": stems,
            }

        split_stems[split] = set(image_stems)
        split_records[split] = {
            "n_images": len(image_stems),
            "n_groups": len(groups),
            "image_suffixes": dict(Counter(path.suffix.lower() for path in images)),
            "image_stem_digest": digest_lines(image_stems),
            "label_stem_digest": digest_lines(label_stems),
            "groups": groups,
        }

    overlaps = {
        "train_val": sorted(split_stems["train"] & split_stems["val"]),
        "train_test": sorted(split_stems["train"] & split_stems["test"]),
        "val_test": sorted(split_stems["val"] & split_stems["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"split overlap detected: {overlaps}")

    group_sets = {
        split: set(record["groups"])
        for split, record in split_records.items()
    }
    group_overlaps = {
        "train_val": sorted(group_sets["train"] & group_sets["val"]),
        "train_test": sorted(group_sets["train"] & group_sets["test"]),
        "val_test": sorted(group_sets["val"] & group_sets["test"]),
    }

    payload = {
        "dataset": "M3FD",
        "purpose": (
            "Full stem and group manifest for the fixed 1,600/400/200 split "
            "used by the CS-ECF evaluators and the YOLOv11-RGBT reference."
        ),
        "group_definition": "first three characters of the image stem",
        "source_root": str(args.dataset_root.resolve()),
        "split_construction": (
            "Frames were first assigned to consecutive 100-frame groups. "
            "The training, validation, and held-out test collections were "
            "then constructed from disjoint groups. This artifact records "
            "the resulting paths and verifies stem and group overlap."
        ),
        "total_images": sum(
            record["n_images"] for record in split_records.values()
        ),
        "splits": split_records,
        "stem_overlap": overlaps,
        "group_overlap": group_overlaps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "counts": {
            split: record["n_images"]
            for split, record in split_records.items()
        },
        "groups": {
            split: sorted(record["groups"])
            for split, record in split_records.items()
        },
        "group_overlap": group_overlaps,
    }, indent=2))


if __name__ == "__main__":
    main()
