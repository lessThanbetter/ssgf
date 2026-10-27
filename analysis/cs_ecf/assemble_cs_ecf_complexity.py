"""Assemble the measured CS-ECF latency components."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    cdf = load("cs_ecf_cdf_runtime_512.json")
    sota = load("cs_ecf_sota_runtime.json")
    detector = load("cs_ecf_detector_runtime.json")

    fusion_components = {
        "cdf_pipeline": cdf["latency_ms"]["end_to_end"]["mean"],
        "piafusion": sota["methods"]["piafusion"]["latency_ms"],
        "spgfusion": sota["methods"]["spgfusion"]["latency_ms"],
    }
    fusion_total = sum(fusion_components.values())
    detector_once = detector["latency_ms"]
    detector_total = detector_once * 4
    payload = {
        "device": detector["device"],
        "standardized_fusion_input_hw": [384, 512],
        "detector_input": detector["input"],
        "batch": 1,
        "execution_model": (
            "serial GPU stages; models assumed loaded or swapped outside "
            "the reported latency"
        ),
        "fusion_components_ms": {
            name: round(value, 3)
            for name, value in fusion_components.items()
        },
        "fusion_generation_total_ms": round(fusion_total, 3),
        "detector_once_ms": round(detector_once, 3),
        "four_detector_forwards_ms": round(detector_total, 3),
        "estimated_serial_total_ms": round(fusion_total + detector_total, 3),
        "parameters": {
            "ifcnn_frozen": cdf["parameters"]["ifcnn_frozen"],
            "cdf_trainable": cdf["parameters"]["cdf_trainable"],
            "piafusion_m": sota["methods"]["piafusion"]["params_m"],
            "spgfusion_m": sota["methods"]["spgfusion"]["params_m"],
        },
        "per_stage_extra_peak_mib": {
            "piafusion": sota["methods"]["piafusion"]["extra_peak_mib"],
            "spgfusion": sota["methods"]["spgfusion"]["extra_peak_mib"],
            "detector_once": detector["extra_peak_mib"],
        },
        "notes": [
            "Four detector forwards are counted serially; batching was not assumed.",
            "Per-stage memory is not additive because models were benchmarked independently.",
            "End-to-end wall time including data loading and model swapping was not measured.",
        ],
    }
    output = RESULTS / "cs_ecf_complexity.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("WROTE", output)


if __name__ == "__main__":
    main()
