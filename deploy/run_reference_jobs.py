"""Run the reference-VIF scorer for several datasets from one JSON config."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for job in config["jobs"]:
        name = job["name"]
        cmd = [
            args.python, str(root / "analysis" / "vif_reference_rescore.py"),
            "--vis", job["vis"], "--ir", job["ir"],
            "--limit", str(job.get("limit", 0)),
            "--max_side", str(job.get("max_side", 0)),
            "--out", str(outdir / name),
        ]
        if job.get("target_hw"):
            cmd += ["--target_hw", job["target_hw"]]
        if job.get("fields"):
            cmd += ["--fields", *job["fields"]]
        for spec in job["fused"]:
            cmd += ["--fused", spec]
        print("RUN", name, flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        log = outdir / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                  text=True)
        if proc.returncode:
            raise SystemExit(f"{name} failed; see {log}")
        print("DONE", name, log, flush=True)
    print("REFERENCE_JOBS_DONE", flush=True)


if __name__ == "__main__":
    main()
