"""Off-the-shelf semantic prior via a pretrained salient-object detector.

Wraps a U2-Net ONNX export (``u2netp.onnx``, ~4.7 MB) so we can turn any
grayscale source image into a per-pixel ``[0, 1]`` importance/saliency map
without training. This is the "pretrained prior" used by the A-direction trial.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class U2NetSaliency:
    def __init__(self, onnx_path: str, size: int = 320, output_index: int = 4):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path)
        self.size = size
        self.output_index = output_index
        # ImageNet stats used by the U2-Net preprocessing.
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """``img``: ``(H, W)`` float in ``[0, 1]`` -> saliency ``(H, W)`` in ``[0, 1]``."""
        h, w = img.shape
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGB").resize((self.size, self.size))
        x = np.asarray(pil, dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]            # (1,3,S,S)
        x = (x - self.mean) / self.std
        out = self.sess.run(None, {"input.1": x})[self.output_index][0, 0]
        out = np.clip(out, 0, 1)
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        out = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize((w, h)), dtype=np.float32) / 255.0
        return out


class TophatTargetPrior:
    """Target-aware prior via morphological top-hat (small thermal targets).

    Generic saliency detectors (U2-Net) and COCO detectors both fail on
    low-contrast military scenes because they miss small thermal targets. This
    prior is built on the classic white/black top-hat operator, which is exactly
    designed to pop small bright/dark targets (a soldier, a vehicle) that blend
    into the terrain. It is the "stronger semantic/targetness" prior.
    """

    def __init__(self, kernel: int = 9, sigma: float = 3.0):
        import cv2

        self.kernel = kernel
        self.sigma = sigma
        self._cv2 = cv2

    def __call__(self, img) -> "np.ndarray":
        x = np.clip(img, 0, 1).astype(np.float32)
        k = self._cv2.getStructuringElement(self._cv2.MORPH_RECT, (self.kernel, self.kernel))
        wth = self._cv2.morphologyEx(x, self._cv2.MORPH_TOPHAT, k)
        bth = self._cv2.morphologyEx(1.0 - x, self._cv2.MORPH_TOPHAT, k)
        t = np.maximum(wth, bth)
        t = self._cv2.GaussianBlur(t, (0, 0), self.sigma)
        rng = t.max() - t.min()
        return (t - t.min()) / (rng + 1e-8)


def build_prior(spec: str):
    """Build a semantic prior from a spec string.

    ``u2net:/path/to/model.onnx`` -> U2NetSaliency,
    ``tophat`` -> TophatTargetPrior.
    """
    if spec == "tophat":
        return TophatTargetPrior()
    if spec.startswith("u2net:"):
        return U2NetSaliency(spec.split(":", 1)[1])
    # bare path -> treat as a U2Net ONNX export
    return U2NetSaliency(spec)
