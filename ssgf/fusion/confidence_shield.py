"""Evaluator-conditioned confidence shielding for fusion input selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ConfidenceShieldConfig:
    top_k: int = 5
    visible_margin: float = 1.5


DEFAULT_CONFIDENCE_SHIELD = ConfidenceShieldConfig()


def topk_confidence(values: np.ndarray, top_k: int = 5) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return -1.0
    return float(np.sort(values)[-top_k:].mean())


def select_candidate(
    candidate_confidences: Mapping[str, np.ndarray],
    config: ConfidenceShieldConfig = DEFAULT_CONFIDENCE_SHIELD,
) -> tuple[str, dict[str, float]]:
    """Select a fused input only when its detector confidence clears the shield."""

    if "visible" not in candidate_confidences:
        raise KeyError("candidate_confidences must contain 'visible'")
    scores = {
        candidate: topk_confidence(values, config.top_k)
        for candidate, values in candidate_confidences.items()
    }
    selected = max(scores, key=scores.get)
    if (
        selected != "visible"
        and scores[selected] < config.visible_margin * scores["visible"]
    ):
        selected = "visible"
    return selected, scores


__all__ = [
    "ConfidenceShieldConfig",
    "DEFAULT_CONFIDENCE_SHIELD",
    "select_candidate",
    "topk_confidence",
]
