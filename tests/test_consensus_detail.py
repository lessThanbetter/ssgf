import numpy as np

from ssgf.fusion import (
    ConsensusDetailConfig,
    consensus_intersection,
    fuse_consensus_detail,
)


def test_consensus_intersection_requires_same_sign():
    first = np.array([[2.0, -4.0, 0.0, 3.0]], np.float32)
    second = np.array([[1.0, -2.0, 5.0, -3.0]], np.float32)
    result = consensus_intersection(first, second)
    np.testing.assert_allclose(result, [[1.0, -2.0, 0.0, 0.0]])


def test_fuse_consensus_detail_preserves_shape():
    rng = np.random.default_rng(4)
    visible = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    infrared = rng.integers(0, 256, (64, 80), dtype=np.uint8)
    mean_candidate = rng.integers(0, 256, (64, 80), dtype=np.uint8)
    ifcnn_candidate = rng.integers(0, 256, (64, 80), dtype=np.uint8)
    config = ConsensusDetailConfig(kernel_size=9, minimum_component=1)
    output, mask, detail = fuse_consensus_detail(
        visible,
        infrared,
        mean_candidate,
        ifcnn_candidate,
        config,
    )
    assert output.shape == visible.shape
    assert output.dtype == np.uint8
    assert mask.shape == infrared.shape
    assert detail.shape == infrared.shape
