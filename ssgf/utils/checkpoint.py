"""Checkpoint save / load for the SSGF networks."""

from __future__ import annotations

import torch


def save_ckpt(path: str, encoder, decoder, probe=None, classifier=None, cfg=None, meta: dict | None = None) -> None:
    payload = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "probe": probe.state_dict() if probe is not None else None,
        "classifier": classifier.state_dict() if classifier is not None else None,
        "cfg": cfg.to_dict() if cfg is not None else None,
    }
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def load_ckpt(path: str, encoder=None, decoder=None, probe=None, classifier=None, strict: bool = False) -> dict:
    """Load shared weights into the provided modules and return the payload."""
    payload = torch.load(path, map_location="cpu")
    for module, key in ((encoder, "encoder"), (decoder, "decoder"), (probe, "probe"), (classifier, "classifier")):
        if module is not None and key in payload and payload[key] is not None:
            module.load_state_dict(payload[key], strict=strict)
    return payload
