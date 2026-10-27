"""Fetch the 150 LLVIP validation infrared images used by the selector study.

The visible validation images already exist in the clean evaluator dataset.  This
script reads only the archived validation subset from the public LLVIP ZIP so that
the selector never uses the 300-image test set for fitting.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from extract_llvip_sources import LLVIP_URL, fetch_member, resolve_url  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="ssgf/_deposit/ssgf-llvip-fusion-detection-v1/data/splits/clean_split_manifest.json",
    )
    parser.add_argument("--visible", default="D:/Code/_ssgf_external/r2_clean/clean/images/val")
    parser.add_argument("--out", default="ssgf/data/selective_fusion/validation_sources")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    stems = [str(value) for value in manifest["split_stems"]["validation"]]
    wanted = set(stems)
    visible_dir = Path(args.visible)
    out = Path(args.out)
    visible_out = out / "visible"
    infrared_out = out / "infrared"
    visible_out.mkdir(parents=True, exist_ok=True)
    infrared_out.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        source = visible_dir / f"{stem}.jpg"
        if not source.is_file():
            raise SystemExit(f"missing visible validation image: {source}")
        target = visible_out / f"{stem}.jpg"
        if not target.exists():
            shutil.copy2(source, target)

    with RemoteZip(LLVIP_URL, support_suffix_range=True) as archive:
        members = {
            Path(info.filename).stem: info
            for info in archive.infolist()
            if info.filename.startswith("LLVIP/infrared/train/")
            and Path(info.filename).stem in wanted
        }
    missing = sorted(wanted - set(members))
    if missing:
        raise SystemExit(f"missing infrared validation images: {missing[:10]}")

    final_url = resolve_url(LLVIP_URL)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                fetch_member,
                final_url,
                members[stem],
                infrared_out / f"{stem}.jpg",
            ): stem
            for stem in stems
        }
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 25 == 0 or index == len(futures):
                print(f"downloaded {index}/{len(futures)}", flush=True)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "n": len(stems),
                "stems": stems,
                "test_overlap": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
