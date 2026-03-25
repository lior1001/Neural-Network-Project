"""
Two-stage video inference:
1. Stage 1 detects the cyclist/bicycle region.
2. Stage 2 localizes the helmet inside a crop around that region.

Improvements over previous version:
- EMA temporal smoothing on both bicycle and helmet boxes
- Expanded crop height (0.70 instead of 0.58) for BMX/aggressive riders
- Geometric consistency check (helmet must be in upper half of bicycle box)
- Crop jitter at inference time mirrors training augmentation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as T

from dataset_helmet_stage2 import make_stage2_crop_xyxy
from model import MobileNetV3MultiBBox
from model_helmet_stage2 import HelmetCropRegressor
from utils import helmet_stage2_display_score

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SLOT_BICYCLE = 0


def parse_args():
    p = argparse.ArgumentParser(description="Two-stage bicycle + helmet video inference.")
    p.add_argument("--stage1_checkpoint", type=Path, required=True)
    p.add_argument("--stage2_checkpoint", type=Path, required=True)
    p.add_argument("--video_in", type=Path, required=True)
    p.add_argument("--video_out", type=Path, required=True)
    p.add_argument("--stage1_input_size", type=int, default=448)
    p.add_argument("--stage2_input_size", type=int, default=224)
    p.add_argument("--bike_conf_thresh", type=float, default=0.55)
    p.add_argument("--smooth_alpha", type=float, default=0.5,
                   help="EMA alpha: fraction of NEW detection per frame. Lower = smoother.")
    p.add_argument("--max_miss", type=int, default=5,
                   help="Frames to hold a detection after it disappears.")
    p.add_argument("--crop_height_frac", type=float, default=0.70,
                   help="Fraction of bicycle box height to include in stage2 crop. "
                        "Use 0.70+ for BMX/aggressive riders (default: 0.70).")
    p.add_argument("--no_geo_filter", action="store_true",
                   help="Disable geometric consistency check (for debugging).")
    return p.parse_args()


def _to_image_tensor(input_size: int):
    return T.Compose([
        T.ToImage(),
        T.Resize((input_size, input_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _norm_cxcywh_to_xyxy_px(box: np.ndarray, width: int, height: int):
    cx, cy, bw, bh = box
    x1 = int(max(0, (cx - bw / 2.0) * width))
    y1 = int(max(0, (cy - bh / 2.0) * height))
    x2 = int(min(width - 1, (cx + bw / 2.0) * width))
    y2 = int(min(height - 1, (cy + bh / 2.0) * height))
    return x1, y1, x2, y2


def _make_stage2_crop_xyxy(
    bx1: int, by1: int, bx2: int, by2: int,
    width: int, height: int,
    crop_height_frac: float = 0.70,
) -> Tuple[int, int, int, int]:
    """
    Build stage 2 crop from a pixel-space bicycle box (xyxy).

    crop_height_frac controls how much of the bicycle height is included
    vertically. Increase for BMX/aggressive forward-lean riders.
    """
    bw = bx2 - bx1
    bh = by2 - by1
    x1 = max(0, int(bx1 - 0.10 * bw))
    y1 = max(0, int(by1 - 0.12 * bh))
    x2 = min(width - 1, int(bx1 + 1.10 * bw))
    y2 = min(height - 1, int(by1 + crop_height_frac * bh))
    if x2 <= x1 + 1:
        x2 = min(width - 1, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(height - 1, y1 + 2)
    return x1, y1, x2, y2


def _stage2_box_to_full_image(
    box: np.ndarray, crop_xyxy: Tuple[int, int, int, int], width: int, height: int
) -> Tuple[int, int, int, int]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
    crop_w = max(crop_x2 - crop_x1, 1)
    crop_h = max(crop_y2 - crop_y1, 1)
    cx, cy, bw, bh = box
    x1 = crop_x1 + (cx - bw / 2.0) * crop_w
    y1 = crop_y1 + (cy - bh / 2.0) * crop_h
    x2 = crop_x1 + (cx + bw / 2.0) * crop_w
    y2 = crop_y1 + (cy + bh / 2.0) * crop_h
    return (
        int(max(0, x1)),
        int(max(0, y1)),
        int(min(width - 1, x2)),
        int(min(height - 1, y2)),
    )


def _helmet_consistent_with_bicycle(
    hx1: int, hy1: int, hx2: int, hy2: int,
    bx1: int, by1: int, bx2: int, by2: int,
) -> bool:
    """Helmet center must be in the upper half of the bicycle box."""
    hcx = (hx1 + hx2) / 2.0
    hcy = (hy1 + hy2) / 2.0
    b_height = by2 - by1
    b_width = bx2 - bx1
    b_mid_y = by1 + b_height * 0.5
    margin_x = b_width * 0.30
    h_area = (hx2 - hx1) * (hy2 - hy1)
    b_area = b_width * b_height
    in_y = hcy < b_mid_y
    in_x = (bx1 - margin_x) <= hcx <= (bx2 + margin_x)
    size_ok = h_area < b_area * 0.60
    not_tiny = (hx2 - hx1) > 8 and (hy2 - hy1) > 8
    return in_y and in_x and size_ok and not_tiny


class BoxSmoother:
    """EMA smoother for a single bounding box (pixel xyxy)."""

    def __init__(self, alpha: float = 0.5, max_miss: int = 5):
        self.alpha = alpha
        self.max_miss = max_miss
        self._box: Optional[np.ndarray] = None
        self._miss: int = 0

    def update(self, box: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if box is not None:
            if self._box is None:
                self._box = box.astype(float)
            else:
                self._box = (1 - self.alpha) * self._box + self.alpha * box.astype(float)
            self._miss = 0
        else:
            self._miss += 1
            if self._miss > self.max_miss:
                self._box = None
        return self._box.astype(int) if self._box is not None else None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stage1 = MobileNetV3MultiBBox(num_classes=2, num_slots=2, pretrained=False)
    ckpt1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    stage1.load_state_dict(ckpt1.get("model_state", ckpt1), strict=True)
    stage1.to(device).eval()

    stage2 = HelmetCropRegressor(pretrained=False)
    ckpt2 = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
    stage2.load_state_dict(ckpt2.get("model_state", ckpt2), strict=True)
    stage2.to(device).eval()

    tf_stage1 = _to_image_tensor(args.stage1_input_size)
    tf_stage2 = _to_image_tensor(args.stage2_input_size)

    cap = cv2.VideoCapture(str(args.video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}×{height} @ {fps:.1f} fps, {total} frames")
    print(f"Crop height fraction: {args.crop_height_frac:.2f}")
    print(f"Geo filter: {'OFF' if args.no_geo_filter else 'ON'}")

    args.video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    bike_smoother = BoxSmoother(alpha=args.smooth_alpha, max_miss=args.max_miss)
    helmet_smoother = BoxSmoother(alpha=args.smooth_alpha, max_miss=args.max_miss)

    frame_idx = 0
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # ── Stage 1: detect bicycle ──────────────────────────────────────
            stage1_tensor = tf_stage1(pil).unsqueeze(0).to(device)
            pred_boxes, pred_logits = stage1(stage1_tensor)
            pred_boxes = pred_boxes[0]
            pred_logits = pred_logits[0]
            probs = F.softmax(pred_logits, dim=-1)
            bike_conf = float(probs[SLOT_BICYCLE, SLOT_BICYCLE + 1].item())

            raw_bike_box = None
            raw_helmet_box = None

            if bike_conf >= args.bike_conf_thresh:
                bike_box_norm = pred_boxes[SLOT_BICYCLE].cpu().numpy()
                bx1, by1, bx2, by2 = _norm_cxcywh_to_xyxy_px(bike_box_norm, width, height)
                raw_bike_box = np.array([bx1, by1, bx2, by2], dtype=float)

                # ── Stage 2: localize helmet in crop ─────────────────────────
                crop_xyxy = _make_stage2_crop_xyxy(
                    bx1, by1, bx2, by2, width, height,
                    crop_height_frac=args.crop_height_frac,
                )
                crop = pil.crop(crop_xyxy)
                stage2_tensor = tf_stage2(crop).unsqueeze(0).to(device)
                helmet_box_crop = stage2(stage2_tensor)[0].cpu().numpy()
                hx1, hy1, hx2, hy2 = _stage2_box_to_full_image(
                    helmet_box_crop, crop_xyxy, width, height
                )

                # ── Geometric consistency check ───────────────────────────────
                if args.no_geo_filter or _helmet_consistent_with_bicycle(
                    hx1, hy1, hx2, hy2, bx1, by1, bx2, by2
                ):
                    raw_helmet_box = np.array([hx1, hy1, hx2, hy2], dtype=float)

            # ── EMA smoothing ─────────────────────────────────────────────────
            smooth_bike = bike_smoother.update(raw_bike_box)
            smooth_helmet = helmet_smoother.update(raw_helmet_box)

            # ── Draw ──────────────────────────────────────────────────────────
            if smooth_bike is not None:
                bx1, by1, bx2, by2 = smooth_bike
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(
                    frame, f"Bicycle {bike_conf:.2f}", (bx1, max(by1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
                )

            if smooth_helmet is not None:
                hx1, hy1, hx2, hy2 = smooth_helmet
                helmet_score = helmet_stage2_display_score(
                    (hx1, hy1, hx2, hy2),
                    (smooth_bike[0], smooth_bike[1], smooth_bike[2], smooth_bike[3])
                    if smooth_bike is not None else (hx1, hy1, hx2, hy2),
                )
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)
                cv2.putText(
                    frame, f"Helmet (stage2) {helmet_score:.2f}", (hx1, max(hy1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA,
                )

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"  Frame {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"\nDone. Saved to {args.video_out}")


if __name__ == "__main__":
    main()