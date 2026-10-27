"""Paired risk comparison between CS-ECF and the strongest fixed baseline.

The script reconstructs the frozen CS-ECF decisions from the released per-image
detection scores, checks that its point estimates agree with the published
aggregate table, and reports paired image, group and hierarchical bootstrap
intervals for CS-ECF minus Always CDF.

Run from the repository root:

    python scripts/analyze_csecf_vs_cdf.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from local_ap_reference import ap_per_class, load, pooled


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "selective_fusion"
DEPOSIT = ROOT / "_deposit" / "ssgf-llvip-fusion-detection-v1" / "data" / "per_image"
OUT = RESULTS / "csecf_vs_cdf_paired.json"

TOP_K = 5
VISIBLE_MARGIN = 1.5
CHECKPOINT_BOOTSTRAP_DRAWS = 500
AGGREGATE_BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 2219


@dataclass(frozen=True)
class Checkpoint:
    dataset: str
    name: str
    run_cluster: str
    candidates: dict[str, Path]


def _llvip_paths(vis: Path, cdf: Path, pia: Path, spg: Path) -> dict[str, Path]:
    return {"vis": vis, "cdf": cdf, "pia": pia, "spg": spg}


def _candidate_paths() -> list[Checkpoint]:
    pixel = RESULTS / "test_scores_pixelmean_cdf"
    missing = RESULTS / "test_scores_sota_missing"
    cdf_name = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"

    checkpoints = [
        Checkpoint(
            "llvip",
            "yolo11",
            "llvip-yolo11-visible-seed0",
            _llvip_paths(
                pixel / "yolo11__vis.npz",
                pixel / f"yolo11__{cdf_name}.npz",
                DEPOSIT / "vis_split__piafusion.npz",
                DEPOSIT / "vis_split__spgfusion.npz",
            ),
        ),
        Checkpoint(
            "llvip",
            "agnostic",
            "llvip-yolo11-agnostic-seed0",
            _llvip_paths(
                pixel / "agnostic__vis.npz",
                pixel / f"agnostic__{cdf_name}.npz",
                DEPOSIT / "agn_split__piafusion.npz",
                DEPOSIT / "agn_split__spgfusion.npz",
            ),
        ),
        Checkpoint(
            "llvip",
            "yolo26_s0",
            "llvip-yolo26-seed0",
            _llvip_paths(
                pixel / "yolo26__vis.npz",
                pixel / f"yolo26__{cdf_name}.npz",
                DEPOSIT / "yolo26n_clean__piafusion.npz",
                DEPOSIT / "yolo26n_clean__spgfusion.npz",
            ),
        ),
        Checkpoint(
            "llvip",
            "rtdetr",
            "llvip-rtdetr-seed0",
            _llvip_paths(
                pixel / "rtdetr__vis.npz",
                pixel / f"rtdetr__{cdf_name}.npz",
                DEPOSIT / "rtdetr_l_clean__piafusion.npz",
                DEPOSIT / "rtdetr_l_clean__spgfusion.npz",
            ),
        ),
        Checkpoint(
            "llvip",
            "yolo26_s1",
            "llvip-yolo26-seed1",
            _llvip_paths(
                pixel / "yolo26_s1__vis.npz",
                pixel / f"yolo26_s1__{cdf_name}.npz",
                missing / "yolo26_s1__piafusion.npz",
                missing / "yolo26_s1__spgfusion.npz",
            ),
        ),
        Checkpoint(
            "llvip",
            "yolo26_s2",
            "llvip-yolo26-seed2",
            _llvip_paths(
                pixel / "yolo26_s2__vis.npz",
                pixel / f"yolo26_s2__{cdf_name}.npz",
                missing / "yolo26_s2__piafusion.npz",
                missing / "yolo26_s2__spgfusion.npz",
            ),
        ),
        Checkpoint(
            "llvip",
            "yolo8_s0",
            "llvip-yolo8-seed0",
            _llvip_paths(
                pixel / "yolo8n_s0__vis.npz",
                pixel / f"yolo8n_s0__{cdf_name}.npz",
                missing / "yolo8n_s0__piafusion.npz",
                missing / "yolo8n_s0__spgfusion.npz",
            ),
        ),
        Checkpoint(
            "llvip",
            "yolo8_s1",
            "llvip-yolo8-seed1",
            _llvip_paths(
                pixel / "yolo8n_s1__vis.npz",
                pixel / f"yolo8n_s1__{cdf_name}.npz",
                missing / "yolo8n_s1__piafusion.npz",
                missing / "yolo8n_s1__spgfusion.npz",
            ),
        ),
    ]

    m3fd = RESULTS / "m3fd_sota_scores"
    m3fd_cdf = RESULTS / "m3fd_6cls_multiclass"
    for detector in ("yolo26_m3fd6", "yolo8n_m3fd6"):
        checkpoints.append(
            Checkpoint(
                "m3fd",
                detector,
                f"m3fd-{detector}",
                _llvip_paths(
                    m3fd / f"{detector}__vis.npz",
                    m3fd_cdf / f"{detector}__{cdf_name}.npz",
                    m3fd / f"{detector}__piafusion.npz",
                    m3fd / f"{detector}__spgfusion.npz",
                ),
            )
        )

    for suffix, name in (("", "best"), ("_last", "last")):
        base = RESULTS / f"dronevehicle_test_scores{suffix}"
        stem = "yolo11x_obb" + ("_last" if suffix else "")
        checkpoints.append(
            Checkpoint(
                "dronevehicle_test",
                name,
                "dronevehicle-yolo11x-obb",
                _llvip_paths(
                    base / f"{stem}__vis.npz",
                    base / f"{stem}__{cdf_name}.npz",
                    base / f"{stem}__piafusion.npz",
                    base / f"{stem}__spgfusion.npz",
                ),
            )
        )

    return checkpoints


def _load_checkpoint(spec: Checkpoint) -> dict[str, dict]:
    states = {}
    stems = None
    for candidate, path in spec.candidates.items():
        if not path.exists():
            raise FileNotFoundError(path)
        state = load(path)
        if stems is None:
            stems = state["stems"]
        elif state["stems"] != stems:
            raise ValueError(f"stem order differs for {spec.name}: {candidate}")
        states[candidate] = state
    return states


def _topk(values: np.ndarray, k: int = TOP_K) -> float:
    if len(values) == 0:
        return -1.0
    return float(np.sort(values)[-k:].mean())


def _decisions(states: dict[str, dict]) -> tuple[list[str], list[str]]:
    n = states["vis"]["n"]
    shielded, unshielded = [], []
    for image in range(n):
        q = {
            candidate: _topk(state["cf"][image])
            for candidate, state in states.items()
        }
        best = max(("vis", "cdf", "pia", "spg"), key=lambda name: q[name])
        unshielded.append(best)
        if (
            best != "vis"
            and q[best] >= VISIBLE_MARGIN * q["vis"]
        ):
            shielded.append(best)
        else:
            shielded.append("vis")
    return shielded, unshielded


def _policy_stats(
    states: dict[str, dict], selected: list[str]
) -> dict[str, np.ndarray | int | list[str]]:
    reference = states["vis"]
    tp, conf, pred_cls, gt = [], [], [], []
    det_counts, gt_counts = [], []
    for image, candidate in enumerate(selected):
        state = states[candidate]
        tp.append(state["tp"][image])
        conf.append(state["cf"][image])
        pred_cls.append(state["cl"][image])
        gt.append(state["gt"][image])
        det_counts.append(len(state["cf"][image]))
        gt_counts.append(len(state["gt"][image]))

    det_counts_array = np.asarray(det_counts, dtype=np.int64)
    gt_counts_array = np.asarray(gt_counts, dtype=np.int64)
    return {
        "n": reference["n"],
        "stems": reference["stems"],
        "tp_all": np.concatenate(tp),
        "cf_all": np.concatenate(conf),
        "cl_all": np.concatenate(pred_cls),
        "gt_all": np.concatenate(gt),
        "det_counts": det_counts_array,
        "det_offsets": np.concatenate(([0], np.cumsum(det_counts_array)[:-1])),
        "gt_counts": gt_counts_array,
        "gt_offsets": np.concatenate(([0], np.cumsum(gt_counts_array)[:-1])),
    }


def _ap(stats: dict, image_indices: np.ndarray) -> float:
    return float(pooled(stats, np.asarray(image_indices, dtype=np.int64))[0])


def _groups(dataset: str, stems: list[str]) -> list[np.ndarray]:
    labels = []
    for stem in stems:
        if dataset == "llvip":
            labels.append(stem[:3])
        elif dataset == "m3fd":
            labels.append(stem[:3])
        else:
            match = re.match(r"test_(\d{3})", stem)
            if match is None:
                raise ValueError(f"unrecognised DroneVehicle stem: {stem}")
            labels.append(str(int(match.group(1))))
    unique = sorted(set(labels), key=lambda value: int(value))
    return [
        np.flatnonzero(np.asarray(labels) == label)
        for label in unique
    ]


def _draw_by_group(
    rng: np.random.Generator, groups: list[np.ndarray]
) -> np.ndarray:
    selected = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[index] for index in selected])


def _interval(draws: np.ndarray) -> list[float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    checkpoints = _candidate_paths()
    published = json.loads(
        (RESULTS / "confirmatory_policy_comparison.json").read_text(
            encoding="utf-8"
        )
    )["per_checkpoint"]
    rows = {}
    dataset_images: dict[str, dict] = {}
    dataset_groups: dict[str, list[np.ndarray]] = {}

    for spec in checkpoints:
        states = _load_checkpoint(spec)
        n = states["vis"]["n"]
        if len(states["vis"]["stems"]) != n:
            raise ValueError(f"invalid image count for {spec.name}")

        shielded_names, unshielded_names = _decisions(states)
        shielded_stats = _policy_stats(states, shielded_names)
        cdf_stats = states["cdf"]
        all_images = np.arange(n)
        cdf_ap = _ap(cdf_stats, all_images)
        shielded_ap = _ap(shielded_stats, all_images)
        published_key = f"{spec.dataset}/{spec.name}"
        expected = published[published_key]
        expected_cdf = (
            expected["visible_ap50"] + expected["always_cdf"]
        )
        expected_csecf = (
            expected["visible_ap50"] + expected["cs_ecf"]
        )
        if abs(cdf_ap - expected_cdf) > 5e-4:
            raise ValueError(
                f"{spec.name}: CDF AP {cdf_ap:.6f} != "
                f"published {expected_cdf:.6f}"
            )
        if abs(shielded_ap - expected_csecf) > 5e-4:
            raise ValueError(
                f"{spec.name}: CS-ECF AP {shielded_ap:.6f} != "
                f"published {expected_csecf:.6f}"
            )
        groups = _groups(spec.dataset, states["vis"]["stems"])

        image_draws = []
        group_draws = []
        for _ in range(CHECKPOINT_BOOTSTRAP_DRAWS):
            image_indices = rng.integers(0, n, size=n)
            group_indices = _draw_by_group(rng, groups)
            image_draws.append(
                _ap(shielded_stats, image_indices)
                - _ap(cdf_stats, image_indices)
            )
            group_draws.append(
                _ap(shielded_stats, group_indices)
                - _ap(cdf_stats, group_indices)
            )
        image_draws = np.asarray(image_draws)
        group_draws = np.asarray(group_draws)

        rows[spec.name] = {
            "dataset": spec.dataset,
            "run_cluster": spec.run_cluster,
            "visible_ap50": _ap(states["vis"], all_images),
            "always_cdf_ap50": cdf_ap,
            "csecf_ap50": shielded_ap,
            "csecf_minus_cdf": shielded_ap - cdf_ap,
            "csecf_minus_cdf_image_ci95": _interval(image_draws),
            "csecf_minus_cdf_group_ci95": _interval(group_draws),
            "selected_counts": {
                candidate: shielded_names.count(candidate)
                for candidate in ("vis", "cdf", "pia", "spg")
            },
            "unshielded_counts": {
                candidate: unshielded_names.count(candidate)
                for candidate in ("vis", "cdf", "pia", "spg")
            },
        }
        group_level_deltas = np.asarray(
            [
                _ap(shielded_stats, group) - _ap(cdf_stats, group)
                for group in groups
            ]
        )
        group_level_visible_gains = np.asarray(
            [
                _ap(shielded_stats, group)
                - _ap(states["vis"], group)
                for group in groups
            ]
        )
        dataset_images.setdefault(spec.dataset, {})[spec.name] = {
            "run_cluster": spec.run_cluster,
            "group_level_deltas": group_level_deltas,
            "group_level_visible_gains": group_level_visible_gains,
        }
        dataset_groups.setdefault(spec.dataset, groups)

        print(
            f"{spec.dataset:18s} {spec.name:16s} "
            f"CDF={cdf_ap:.4f} CS-ECF={shielded_ap:.4f} "
            f"delta={shielded_ap - cdf_ap:+.4f} "
            f"imageCI={_interval(image_draws)} "
            f"groupCI={_interval(group_draws)}",
            flush=True,
        )

    dataset_rows = {}
    for dataset, checkpoints_for_dataset in dataset_images.items():
        names = list(checkpoints_for_dataset)
        point = float(
            np.mean(
                [
                    rows[name]["csecf_minus_cdf"]
                    for name in names
                ]
            )
        )
        visible_point = float(
            np.mean(
                [
                    rows[name]["csecf_ap50"] - rows[name]["visible_ap50"]
                    for name in names
                ]
            )
        )
        groups = dataset_groups[dataset]
        fixed_model_group_draws = []
        fixed_model_visible_draws = []
        clusters: dict[str, list[str]] = {}
        for name in names:
            cluster = checkpoints_for_dataset[name]["run_cluster"]
            clusters.setdefault(cluster, []).append(name)

        for _ in range(AGGREGATE_BOOTSTRAP_DRAWS):
            sampled_groups = rng.integers(
                0, len(groups), size=len(groups)
            )
            fixed_model_group_draws.append(
                np.mean(
                    [
                        np.mean(
                            checkpoints_for_dataset[name][
                                "group_level_deltas"
                            ][sampled_groups]
                        )
                        for name in names
                    ]
                )
            )
            fixed_model_visible_draws.append(
                np.mean(
                    [
                        np.mean(
                            checkpoints_for_dataset[name][
                                "group_level_visible_gains"
                            ][sampled_groups]
                        )
                        for name in names
                    ]
                )
            )
        dataset_rows[dataset] = {
            "point_csecf_minus_cdf": point,
            "point_csecf_minus_visible": visible_point,
            "fixed_models_group_ci95": _interval(
                np.asarray(fixed_model_group_draws)
            ),
            "csecf_minus_visible_fixed_models_group_ci95": _interval(
                np.asarray(fixed_model_visible_draws)
            ),
            "checkpoints": names,
            "run_clusters": sorted(clusters),
        }

        # Keep the clustered draws for the cross-dataset hierarchical interval.
        checkpoints_for_dataset["_groups"] = groups
        checkpoints_for_dataset["_clusters"] = clusters

    balanced_draws = []
    balanced_visible_draws = []
    datasets = ["llvip", "m3fd", "dronevehicle_test"]
    for _ in range(AGGREGATE_BOOTSTRAP_DRAWS):
        per_dataset = []
        per_dataset_visible = []
        for dataset in datasets:
            checkpoints_for_dataset = dataset_images[dataset]
            groups = checkpoints_for_dataset["_groups"]
            clusters = checkpoints_for_dataset["_clusters"]
            sampled_groups = rng.integers(
                0, len(groups), size=len(groups)
            )
            cluster_names = sorted(clusters)
            sampled_clusters = rng.integers(
                0, len(cluster_names), size=len(cluster_names)
            )
            cluster_contrasts = []
            cluster_visible_contrasts = []
            for cluster_index in sampled_clusters:
                names = clusters[cluster_names[cluster_index]]
                cluster_contrasts.append(
                    np.mean(
                        [
                            np.mean(
                                checkpoints_for_dataset[name][
                                    "group_level_deltas"
                                ][sampled_groups]
                            )
                            for name in names
                        ]
                    )
                )
                cluster_visible_contrasts.append(
                    np.mean(
                        [
                            np.mean(
                                checkpoints_for_dataset[name][
                                    "group_level_visible_gains"
                                ][sampled_groups]
                            )
                            for name in names
                        ]
                    )
                )
            per_dataset.append(float(np.mean(cluster_contrasts)))
            per_dataset_visible.append(
                float(np.mean(cluster_visible_contrasts))
            )
        balanced_draws.append(float(np.mean(per_dataset)))
        balanced_visible_draws.append(float(np.mean(per_dataset_visible)))

    balanced_point = float(
        np.mean(
            [
                dataset_rows[dataset]["point_csecf_minus_cdf"]
                for dataset in datasets
            ]
        )
    )
    balanced_visible_point = float(
        np.mean(
            [
                dataset_rows[dataset]["point_csecf_minus_visible"]
                for dataset in datasets
            ]
        )
    )
    summary = {
        "comparison": "CS-ECF minus Always CDF, mAP50",
        "policy": {
            "top_k": TOP_K,
            "visible_margin": VISIBLE_MARGIN,
            "candidates": ["vis", "cdf", "pia", "spg"],
        },
        "bootstrap": {
            "checkpoint_draws": CHECKPOINT_BOOTSTRAP_DRAWS,
            "aggregate_draws": AGGREGATE_BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%",
            "hierarchical_units": (
                "training-run clusters resampled within dataset; "
                "scene/frame groups resampled independently"
            ),
        },
        "per_checkpoint": rows,
        "per_dataset": dataset_rows,
        "dataset_balanced": {
            "point_csecf_minus_cdf": balanced_point,
            "run_cluster_by_group_ci95": _interval(
                np.asarray(balanced_draws)
            ),
            "point_csecf_minus_visible": balanced_visible_point,
            "visible_run_cluster_by_group_ci95": _interval(
                np.asarray(balanced_visible_draws)
            ),
        },
    }

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDataset-balanced CS-ECF minus CDF: {balanced_point:+.4f}")
    print(
        "run-cluster x group CI: "
        f"{summary['dataset_balanced']['run_cluster_by_group_ci95']}"
    )
    print(
        "Dataset-balanced CS-ECF minus visible: "
        f"{balanced_visible_point:+.4f}"
    )
    print(
        "visible run-cluster x group CI: "
        f"{summary['dataset_balanced']['visible_run_cluster_by_group_ci95']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
