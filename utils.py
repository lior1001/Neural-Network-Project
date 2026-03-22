"""Shared utilities for Part 3 (box format, IoU, matching)."""
from typing import Tuple

import torch


def xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
    """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
    x, y, w, h = box_xywh.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def cxcywh_to_xyxy(box_cxcywh: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = box_cxcywh.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = box_xyxy.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return torch.stack([cx, cy, w, h], dim=-1)


def clamp_box(box: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    return torch.clamp(box, min=min_val, max=max_val)


def iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of boxes in (x1, y1, x2, y2)."""
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    inter_area = inter_w * inter_h
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter_area
    return inter_area / union.clamp(min=1e-6)


def denormalize_box(box_cxcywh: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Scale normalized (cx, cy, w, h) to pixel units."""
    width, height = image_size
    scale = torch.tensor([width, height, width, height], dtype=box_cxcywh.dtype, device=box_cxcywh.device)
    return box_cxcywh * scale


def box_cxcywh_to_xyxy_normalized(box_cxcywh: torch.Tensor, size: float = 1.0) -> torch.Tensor:
    """Convert normalized (cx, cy, w, h) in [0,1] to (x1, y1, x2, y2) in [0, size] for IoU."""
    cx, cy, w, h = box_cxcywh.unbind(-1)
    x1 = (cx - w / 2.0) * size
    y1 = (cy - h / 2.0) * size
    x2 = (cx + w / 2.0) * size
    y2 = (cy + h / 2.0) * size
    return torch.stack([x1, y1, x2, y2], dim=-1)
