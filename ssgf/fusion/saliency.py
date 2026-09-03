"""Path-based attribution used by all fusion rules.

This is the PyTorch analogue of the loop in ``generate.py``: build an
interpolation sequence between the two registered source images, encode each
step, differentiate a chosen score w.r.t. the feature volume, and accumulate
the absolute gradients with a geometric decay.
"""

from __future__ import annotations

import torch


def interpolation_sequences(vis_n: torch.Tensor, ir_n: torch.Tensor, samples: int):
    """Mean-match the sources and build two interpolation paths.

    ``var_ir`` travels from mean-matched visible toward infrared; ``var_vis``
    travels from infrared toward mean-matched visible (CSF ``generate.py``).
    Each returns a stacked ``(samples, 1, H, W)`` tensor.
    """
    vis = vis_n
    ir = ir_n
    vis_matched = vis - vis.mean() + ir.mean()
    diff_vis_ir = ir - vis_matched      # direction toward IR
    diff_ir_vis = vis_matched - ir      # direction toward VIS

    alpha = torch.arange(1, samples + 1, device=vis.device).float() / samples  # (samples,)
    alpha = alpha.view(samples, 1, 1, 1)

    var_ir = vis_matched + alpha * diff_vis_ir
    var_vis = ir + alpha * diff_ir_vis
    return var_ir, var_vis


def path_attribution(encoder, score_fn, interp: torch.Tensor, samples: int, decay: float, target_idx: int):
    """Accumulate ``|grad logit_target| * decay**i`` over the path.

    ``score_fn(feats) -> (B, n_classes)``. The gradient of the selected class
    w.r.t. the feature volume is the per-pixel contribution.
    """
    acc = None
    for i in range(samples):
        x = interp[i : i + 1].detach().clone().requires_grad_(True)
        feats = encoder(x)
        logits = score_fn(feats)
        score = logits[:, target_idx].sum()
        grad = torch.autograd.grad(score, feats)[0]     # (1, C, H, W)
        g = grad.abs().squeeze(0) * (decay ** i)        # (C, H, W)
        acc = g if acc is None else acc + g
    return acc / samples
