import numpy as np

from ssgf.fusion import ConfidenceShieldConfig, select_candidate, topk_confidence


def test_topk_confidence_uses_top_five():
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
    assert abs(topk_confidence(values, 5) - (0.9 + 0.5 + 0.4 + 0.3 + 0.2) / 5) < 1e-9


def test_confidence_shield_falls_back_to_visible():
    selected, _ = select_candidate(
        {
            "visible": np.array([0.6, 0.5, 0.4]),
            "fused": np.array([0.7, 0.6, 0.5]),
        },
        ConfidenceShieldConfig(top_k=3, visible_margin=1.5),
    )
    assert selected == "visible"


def test_confidence_shield_switches_on_clear_margin():
    selected, _ = select_candidate(
        {
            "visible": np.array([0.3, 0.2, 0.1]),
            "fused": np.array([0.9, 0.8, 0.7]),
        },
        ConfidenceShieldConfig(top_k=3, visible_margin=1.5),
    )
    assert selected == "fused"
