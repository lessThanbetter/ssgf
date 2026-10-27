"""Evaluator-robust consensus detail fusion.

The rule injects only high-frequency structure supported by two independent
fusion candidates.  Low frequencies, chroma, and unsupported detail remain from
the visible source.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ConsensusDetailConfig:
    threshold: float = 0.95
    visible_quantile: float = 0.75
    alpha: float = 0.20
    sigma: float = 5.0
    infrared_floor: int = 120
    kernel_size: int = 31
    minimum_component: int = 64
    min_magnitude: float = 0.0


DEFAULT_CONSENSUS_DETAIL = ConsensusDetailConfig()


def thermal_validity_mask(
    visible_bgr: np.ndarray,
    infrared: np.ndarray,
    config: ConsensusDetailConfig = DEFAULT_CONSENSUS_DETAIL,
) -> np.ndarray:
    """Return a smooth [0, 1] mask for thermally informative visible-dark regions."""

    visible_gray = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.kernel_size, config.kernel_size)
    )
    targetness = cv2.morphologyEx(infrared, cv2.MORPH_TOPHAT, kernel)
    raw = (
        (targetness >= np.quantile(targetness, config.threshold))
        & (infrared >= config.infrared_floor)
        & (
            visible_gray
            < np.quantile(visible_gray, config.visible_quantile)
        )
    ).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(raw, 8)
    cleaned = np.zeros_like(raw)
    for label in range(1, count):
        if statistics[label, cv2.CC_STAT_AREA] >= config.minimum_component:
            cleaned[labels == label] = 255
    cleaned = cv2.dilate(
        cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    cleaned = cv2.GaussianBlur(cleaned, (31, 31), 0)
    return cleaned.astype(np.float32) / 255.0


def highpass(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    image = image.astype(np.float32)
    return image - cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def pixel_mean_fusion(
    visible_bgr: np.ndarray,
    infrared: np.ndarray,
) -> np.ndarray:
    visible = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2GRAY)
    return ((visible.astype(np.float32) + infrared.astype(np.float32)) / 2.0).astype(
        np.uint8
    )


def consensus_intersection(
    first: np.ndarray,
    second: np.ndarray,
    min_magnitude: float = 0.0,
) -> np.ndarray:
    """Keep the smaller magnitude only when both candidates agree in sign."""

    first = first.astype(np.float32)
    second = second.astype(np.float32)
    magnitude = np.minimum(np.abs(first), np.abs(second))
    return np.where(
        (first * second > 0) & (magnitude >= min_magnitude),
        np.sign(first + second) * magnitude,
        0.0,
    )


def inject_luminance_detail(
    visible_bgr: np.ndarray,
    detail: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.20,
) -> np.ndarray:
    """Inject detail into LAB-L while preserving visible chroma."""

    lab = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] += alpha * mask * detail
    return cv2.cvtColor(
        np.clip(lab, 0.0, 255.0).astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def fuse_consensus_detail(
    visible_bgr: np.ndarray,
    infrared: np.ndarray,
    mean_candidate: np.ndarray,
    ifcnn_candidate: np.ndarray,
    config: ConsensusDetailConfig = DEFAULT_CONSENSUS_DETAIL,
):
    """Fuse one registered pair and return output, mask, and consensus detail."""

    mask = thermal_validity_mask(visible_bgr, infrared, config)
    first = highpass(mean_candidate, config.sigma)
    second = highpass(ifcnn_candidate, config.sigma)
    detail = consensus_intersection(
        first, second, config.min_magnitude
    )
    output = inject_luminance_detail(visible_bgr, detail, mask, config.alpha)
    return output, mask, detail


__all__ = [
    "ConsensusDetailConfig",
    "DEFAULT_CONSENSUS_DETAIL",
    "consensus_intersection",
    "fuse_consensus_detail",
    "highpass",
    "inject_luminance_detail",
    "pixel_mean_fusion",
    "thermal_validity_mask",
]
