"""Fallback to CPU NMS when the local torchvision build lacks CUDA NMS."""

from __future__ import annotations

import torch
import torchvision


_ORIGINAL_NMS = torchvision.ops.nms


def _nms(boxes, scores, iou_threshold):
    if boxes.is_cuda:
        keep = _ORIGINAL_NMS(
            boxes.detach().cpu(),
            scores.detach().cpu(),
            float(iou_threshold),
        )
        return keep.to(device=boxes.device)
    return _ORIGINAL_NMS(boxes, scores, iou_threshold)


torchvision.ops.nms = _nms
