"""
Losses for Part 3: fixed-slot multi-object detection.

Directly mirrors the reference implementation's compute_loss().

KEY INSIGHT — why fixed-slot loss is simpler and more stable:
  Because slot i always corresponds to class i, we know exactly which
  prediction to compare to which ground-truth box. No assignment problem,
  no Hungarian matching, no grid distance calculation.

Loss:
  - Bbox (SmoothL1): only on slots where gt class != background.
  - Classification (CrossEntropy): on ALL slots (background slots train
    the model to output class 0 when nothing is there).
  - Optional GIoU term on foreground slots for better localization.
"""

import torch
import torch.nn.functional as F

from utils import box_cxcywh_to_xyxy_normalized

BACKGROUND_CLASS_ID = 0


def _giou(pred_cxcywh: torch.Tensor, gt_cxcywh: torch.Tensor) -> torch.Tensor:
    """Per-element GIoU in [-1,1]. Inputs: (..., 4) normalised cxcywh."""
    p = box_cxcywh_to_xyxy_normalized(pred_cxcywh, size=1.0).clamp(0.0, 1.0)
    g = box_cxcywh_to_xyxy_normalized(gt_cxcywh,   size=1.0).clamp(0.0, 1.0)

    ix1 = torch.max(p[..., 0], g[..., 0])
    iy1 = torch.max(p[..., 1], g[..., 1])
    ix2 = torch.min(p[..., 2], g[..., 2])
    iy2 = torch.min(p[..., 3], g[..., 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    ap    = (p[..., 2] - p[..., 0]).clamp(0) * (p[..., 3] - p[..., 1]).clamp(0)
    ag    = (g[..., 2] - g[..., 0]).clamp(0) * (g[..., 3] - g[..., 1]).clamp(0)
    union = ap + ag - inter
    iou   = inter / union.clamp(min=1e-6)

    ex1 = torch.min(p[..., 0], g[..., 0])
    ey1 = torch.min(p[..., 1], g[..., 1])
    ex2 = torch.max(p[..., 2], g[..., 2])
    ey2 = torch.max(p[..., 3], g[..., 3])
    enc  = ((ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0)).clamp(min=1e-6)

    return iou - (enc - union) / enc


def compute_loss(
    pred_boxes:   torch.Tensor,    # (B, num_slots, 4)   Sigmoid output
    pred_logits:  torch.Tensor,    # (B, num_slots, C+1) raw logits
    gt_boxes:     torch.Tensor,    # (B, num_slots, 4)   normalised cxcywh; bg slots = 0
    gt_class_ids: torch.Tensor,    # (B, num_slots)      long; 0=bg, 1..C=fg
    *,
    bbox_loss_weight: float = 1.0,
    giou_loss_weight: float = 1.0,
    slot_loss_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    """
    Fixed-slot loss. Mirrors reference compute_loss() exactly.

    - Bbox (SmoothL1) + GIoU: only on foreground slots (gt_class_ids != 0)
    - CrossEntropy: on ALL slots (teaches background slots to predict class 0)

    Returns: (total_loss, bbox_loss_scalar, class_loss_scalar)
    """
    fg_mask = (gt_class_ids != BACKGROUND_CLASS_ID)   # (B, num_slots)
    n_fg    = fg_mask.sum().clamp(min=1)

    # ── Bbox loss (foreground only) ───────────────────────────────────────────
    pred_fg = pred_boxes[fg_mask]    # (N_fg, 4)
    gt_fg   = gt_boxes[fg_mask]      # (N_fg, 4)

    if pred_fg.numel() > 0:
        loc_weights: torch.Tensor
        if slot_loss_weights is not None:
            _, num_slots, _ = pred_boxes.shape
            slot_ids = torch.arange(num_slots, device=pred_boxes.device).unsqueeze(0).expand_as(gt_class_ids)
            loc_weights = slot_loss_weights.to(pred_boxes.device)[slot_ids[fg_mask]]
        else:
            loc_weights = torch.ones(pred_fg.size(0), device=pred_boxes.device)

        norm = loc_weights.sum().clamp(min=1e-6)
        l1_per_slot = F.smooth_l1_loss(pred_fg, gt_fg, reduction="none").sum(dim=-1)
        giou_per_slot = 1.0 - _giou(pred_fg, gt_fg)
        loss_bbox = (l1_per_slot * loc_weights).sum() / norm
        loss_giou = (giou_per_slot * loc_weights).sum() / norm
    else:
        loss_bbox = pred_boxes.sum() * 0.0
        loss_giou = pred_boxes.sum() * 0.0

    # ── Classification loss (all slots) ──────────────────────────────────────
    B, S, C1 = pred_logits.shape
    loss_class = F.cross_entropy(
        pred_logits.view(B * S, C1),
        gt_class_ids.view(B * S),
        reduction="mean",
    )

    total = (bbox_loss_weight * loss_bbox
             + giou_loss_weight * loss_giou
             + loss_class)

    return total, float(loss_bbox.detach()), float(loss_class.detach())


def mean_iou(
    pred_boxes:   torch.Tensor,   # (B, num_slots, 4) normalised cxcywh
    gt_boxes:     torch.Tensor,   # (B, num_slots, 4) normalised cxcywh
    gt_class_ids: torch.Tensor,   # (B, num_slots) long
) -> torch.Tensor:
    """Mean IoU over foreground slots only."""
    fg = gt_class_ids != BACKGROUND_CLASS_ID
    if fg.sum() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    p = box_cxcywh_to_xyxy_normalized(pred_boxes, size=1.0).clamp(0.0, 1.0)
    g = box_cxcywh_to_xyxy_normalized(gt_boxes,   size=1.0).clamp(0.0, 1.0)

    pf = p.view(-1, 4)[fg.view(-1)]
    gf = g.view(-1, 4)[fg.view(-1)]

    ix1 = torch.max(pf[:, 0], gf[:, 0])
    iy1 = torch.max(pf[:, 1], gf[:, 1])
    ix2 = torch.min(pf[:, 2], gf[:, 2])
    iy2 = torch.min(pf[:, 3], gf[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    ap    = (pf[:, 2] - pf[:, 0]).clamp(0) * (pf[:, 3] - pf[:, 1]).clamp(0)
    ag    = (gf[:, 2] - gf[:, 0]).clamp(0) * (gf[:, 3] - gf[:, 1]).clamp(0)
    union = ap + ag - inter
    iou   = inter / union.clamp(min=1e-6)
    return iou.mean()