from __future__ import annotations

import torch


def dice_score_binary(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """prob, target: (N,1,H,W) in [0,1]."""
    p = prob.contiguous().view(prob.size(0), -1)
    t = target.contiguous().view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    return ((2 * inter + eps) / (union + eps)).mean()


def iou_binary(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = (prob >= 0.5).float().view(prob.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()
