"""Run reference-VIF jobs with one resumable subprocess per fused variant."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def completed(path: Path, limit: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        if not rows:
            return False
        return not limit or all(row["n"] == limit for row in rows)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_task(task: dict, python: str) -> dict:
    command = [
        python,
        str(task["script"]),
        "--vis", task["vis"],
        "--ir", task["ir"],
        "--fused", f'{task["variant"]}={task["fused"]}',
        "--limit", str(task["limit"]),
        "--max_side", str(task["max_side"]),
        "--fields", *task["fields"],
        "--out", str(task["output"]),
    ]
    if task["target_hw"]:
        command += ["--target_hw", task["target_hw"]]
    task["log"].parent.mkdir(parents=True, exist_ok=True)
    with task["log"].open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                text=True)
    if result.returncode:
        raise RuntimeError(f'{task["dataset"]}/{task["variant"]} failed; '
                           f'see {task["log"]}')
    return task


def merge_dataset(dataset: dict, fields: list[str]) -> None:
    rows = []
    for variant, path in dataset["variant_outputs"].items():
        result_path = path.with_suffix(".json")
        if not result_path.exists():
            raise RuntimeError(f"missing result {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    base = dataset["output"]
    base.parent.mkdir(parents=True, exist_ok=True)
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "n", *fields))
        writer.writeheader()
        writer.writerows(rows)
    json_path = base if base.suffix.lower() == ".json" else base.with_suffix(".json")
    json_path.write_text(
        json.dumps({
            "limit": dataset["limit"],
            "max_side": dataset["max_side"],
            "fields": fields,
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    script = root / "analysis" / "vif_reference_rescore.py"
    tasks = []
    datasets = []
    for job in config["jobs"]:
        output = Path(job["output"]) if job.get("output") else (
            root / "results" / "reference_vif" / job["name"]
        )
        fields = list(dict.fromkeys(job.get("fields", ["reported", "reference"])))
        variant_outputs = {
            spec.split("=", 1)[0]: output.with_name(
                output.name + "__" + spec.split("=", 1)[0]
            )
            for spec in job["fused"]
        }
        dataset = {
            "output": output,
            "limit": int(job.get("limit", 0)),
            "max_side": int(job.get("max_side", 0)),
            "fields": fields,
            "variant_outputs": variant_outputs,
        }
        datasets.append(dataset)
        for spec in job["fused"]:
            variant, fused = spec.split("=", 1)
            variant_output = variant_outputs[variant]
            if completed(variant_output.with_suffix(".json"), dataset["limit"]):
                continue
            tasks.append({
                "dataset": job["name"],
                "variant": variant,
                "fused": fused,
                "vis": job["vis"],
                "ir": job["ir"],
                "limit": dataset["limit"],
                "max_side": dataset["max_side"],
                "target_hw": job.get("target_hw", ""),
                "fields": fields,
                "script": script,
                "output": variant_output,
                "log": variant_output.with_suffix(".log"),
            })

    print(f"[reference-vif] {len(tasks)} tasks, workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, task, args.python): task for task in tasks}
        completed_count = 0
        for future in as_completed(futures):
            task = future.result()
            completed_count += 1
            print(f'[reference-vif] {completed_count}/{len(tasks)} '
                  f'{task["dataset"]}/{task["variant"]}', flush=True)
    for dataset in datasets:
        merge_dataset(dataset, dataset["fields"])
    print("[reference-vif] complete", flush=True)


if __name__ == "__main__":
    main()
