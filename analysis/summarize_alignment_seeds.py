"""Summarize best-aligned contrasts across independent detector seeds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def seed_family(name: str) -> str:
    return re.sub(r"_s\d+$", "", name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mean", "tophat", "probe", "ensemble", "ifcnn",
                 "densefuse", "piafusion", "spgfusion"],
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged = {}
    for path in map(Path, args.files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for evaluator, groups in payload["evaluators"].items():
            family = seed_family(evaluator)
            merged.setdefault(family, {})
            for group, result in groups.items():
                merged[family].setdefault(group, {})[evaluator] = result

    output = {}
    for family, groups in merged.items():
        output[family] = {}
        for group, runs in groups.items():
            inputs = {}
            for name in args.methods:
                deltas, maps = [], []
                for result in runs.values():
                    if name not in result["scores"]:
                        continue
                    value = result["scores"][name]["mAP50"]
                    maps.append(value)
                    if name != "vis":
                        if "vis" in result["scores"]:
                            baseline = result["scores"]["vis"]["mAP50"]
                            deltas.append(value - baseline)
                        elif "delta50" in result["scores"][name]:
                            deltas.append(result["scores"][name]["delta50"])
                if maps:
                    inputs[name] = {
                        "mAP50_mean": float(sum(maps) / len(maps)),
                        "mAP50_min": float(min(maps)),
                        "mAP50_max": float(max(maps)),
                    }
                    if deltas:
                        inputs[name].update({
                            "delta50_mean": float(sum(deltas) / len(deltas)),
                            "delta50_min": float(min(deltas)),
                            "delta50_max": float(max(deltas)),
                        })
            output[family][group] = {"runs": len(runs), "inputs": inputs}

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for family, groups in output.items():
        print(f"== {family} ==")
        for group, result in groups.items():
            values = result["inputs"]
            fields = (
                f"spg={values['spgfusion']['delta50_mean']:+.4f} "
                f"[{values['spgfusion']['delta50_min']:+.4f},"
                f"{values['spgfusion']['delta50_max']:+.4f}]"
            )
            print(f"  {group:9s} runs={result['runs']} {fields}")


if __name__ == "__main__":
    main()
