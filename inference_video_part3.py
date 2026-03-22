"""
Part 3 video inference: fixed-slot multi-object detector (Bicycle + Helmet).

Key features:
- Fixed-slot inference: slot 0 = Bicycle, slot 1 = Helmet
- Geometric consistency check: helmet box center must lie within the
  upper half of the bicycle box, otherwise the helmet detection is
  discarded as a false positive.
- Per-slot EMA temporal smoothing applied AFTER the consistency check.
- Confidence label shown on each box.

Usage:
    python -m your_package.inference_video_part3 \
        --checkpoint  outputs/part3/best.pt \
        --video_in    cyclist_video.mp4 \
        --video_out   outputs/part3/result.mp4 \
        --class_names "Bicycle,Helmet" \
        --conf_thresh 0.40 \
        --smooth_alpha 0.5
"""

import argparse
import colorsys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as T

from model import MobileNetV3MultiBBox

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
BACKGROUND_CLASS_ID = 0

# Slot indices — must match --class_names order
SLOT_BICYCLE = 0
SLOT_HELMET  = 1


def _make_colors(n: int) -> List[Tuple[int, int, int]]:
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


# ── Geometric consistency check ───────────────────────────────────────────────

def _helmet_consistent_with_bicycle(
    helmet_box: np.ndarray,
    bicycle_box: np.ndarray,
) -> bool:
    """
    Return True if the helmet detection is geometrically consistent.

    The helmet/head sits AT OR ABOVE the top of the bicycle bounding box,
    not inside it. Previous version required helmet center to be inside
    the bicycle box which filtered out nearly all correct detections.

    Rules:
    1. Helmet center must be ABOVE the bicycle box bottom (hy < by2).
       This allows it to be above, at the top, or in the upper portion.
    2. Helmet center must NOT be below the bicycle midpoint.
       If the helmet is in the lower half of the bicycle box it is on
       the body/legs, not the head.
    3. Horizontally within bicycle box with 30% margin (generous, since
       the head can be slightly outside the bicycle bounding box).
    4. Helmet area < 60% of bicycle box area (not larger than the bicycle).
    5. Helmet box not tiny (filters pure noise).
    """
    hx = (helmet_box[0] + helmet_box[2]) / 2.0
    hy = (helmet_box[1] + helmet_box[3]) / 2.0

    bx1, by1, bx2, by2 = bicycle_box
    b_height = by2 - by1
    b_width  = bx2 - bx1
    b_mid_y  = by1 + b_height * 0.5

    # Rule 1+2: helmet center must be above bicycle midpoint
    # (allows it to be above the bicycle box entirely, or in upper half)
    in_y = hy < b_mid_y

    # Rule 3: horizontally near the bicycle box
    margin_x = b_width * 0.30
    in_x = (bx1 - margin_x) <= hx <= (bx2 + margin_x)

    # Rule 4: helmet not bigger than the bicycle
    h_area = (helmet_box[2] - helmet_box[0]) * (helmet_box[3] - helmet_box[1])
    b_area = b_width * b_height
    size_ok = h_area < b_area * 0.60

    # Rule 5: not a tiny noise box
    h_w = helmet_box[2] - helmet_box[0]
    h_h = helmet_box[3] - helmet_box[1]
    not_tiny = h_w > 0.02 and h_h > 0.02

    return in_y and in_x and size_ok and not_tiny


# ── Temporal smoother ─────────────────────────────────────────────────────────

class SlotSmoother:
    """
    Per-slot EMA box smoother applied AFTER geometric filtering.

    alpha: fraction of NEW detection blended in each frame.
      0.5 = balanced responsiveness and smoothness (recommended)
      0.3 = very smooth, lags on fast motion
      1.0 = no smoothing
    max_miss: frames to hold a detection after it disappears.
    """

    def __init__(self, num_slots: int, alpha: float = 0.5, max_miss: int = 5):
        self.num_slots = num_slots
        self.alpha     = alpha
        self.max_miss  = max_miss
        self._boxes:  List[Optional[np.ndarray]] = [None] * num_slots
        self._confs:  List[float]                 = [0.0]  * num_slots
        self._misses: List[int]                   = [0]    * num_slots

    def update(
        self,
        slot_boxes: List[Optional[np.ndarray]],  # normalised xyxy or None per slot
        slot_confs: List[float],
    ) -> Tuple[List[Optional[np.ndarray]], List[float]]:
        out_boxes: List[Optional[np.ndarray]] = [None] * self.num_slots
        out_confs: List[float]                = [0.0]  * self.num_slots

        for s in range(self.num_slots):
            if slot_boxes[s] is not None:
                if self._boxes[s] is None:
                    self._boxes[s] = slot_boxes[s].copy()
                else:
                    self._boxes[s] = (
                        (1 - self.alpha) * self._boxes[s]
                        + self.alpha * slot_boxes[s]
                    )
                self._confs[s]  = slot_confs[s]
                self._misses[s] = 0
            else:
                self._misses[s] += 1
                if self._misses[s] > self.max_miss:
                    self._boxes[s] = None

            if self._boxes[s] is not None:
                out_boxes[s] = self._boxes[s]
                out_confs[s] = self._confs[s]

        return out_boxes, out_confs


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Part 3 fixed-slot video inference.")
    p.add_argument("--checkpoint",      type=Path,  required=True,
        help="Path to best.pt from training.")
    p.add_argument("--video_in",        type=Path,  required=True)
    p.add_argument("--video_out",       type=Path,  required=True)
    p.add_argument("--input_size",      type=int,   default=320)
    p.add_argument("--class_names",     type=str,   default="Bicycle,Helmet",
        help="Comma-separated class names IN SAME ORDER AS TRAINING.")
    p.add_argument("--conf_thresh",     type=float, default=0.55,
        help="Min confidence per slot. Try 0.45-0.65.")
    p.add_argument("--smooth_alpha",    type=float, default=0.65,
        help="EMA alpha: fraction of NEW detection per frame. 0=freeze, 1=raw.")
    p.add_argument("--max_miss",        type=int,   default=3,
        help="Frames to hold a detection after it disappears.")
    p.add_argument("--no_geo_filter",   action="store_true",
        help="Disable the geometric consistency check (for debugging).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    class_names = [s.strip() for s in args.class_names.split(",") if s.strip()]
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Need at least 2 class names.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MobileNetV3MultiBBox(
        num_classes=num_classes, num_slots=num_classes, pretrained=False
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd   = ckpt.get("model_state", ckpt)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(f"Loaded: {args.checkpoint}")

    tf = T.Compose([
        T.ToImage(),
        T.Resize((args.input_size, args.input_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    cap = cv2.VideoCapture(str(args.video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video_in}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}×{H} @ {fps:.1f} fps, {total} frames")
    if not args.no_geo_filter:
        print("Geometric consistency check: ON  "
              "(helmet center must be in upper half of bicycle box)")
    else:
        print("Geometric consistency check: OFF")

    args.video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )

    colors   = _make_colors(num_classes)
    smoother = SlotSmoother(
        num_slots=num_classes, alpha=args.smooth_alpha, max_miss=args.max_miss
    )

    frame_idx = 0
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ── Preprocess ────────────────────────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)

            # ── Inference ─────────────────────────────────────────────────────
            pred_boxes, pred_logits = model(tensor)  # (1,S,4), (1,S,C+1)
            pred_boxes  = pred_boxes[0]               # (S, 4) normalised cxcywh
            pred_logits = pred_logits[0]              # (S, C+1)

            probs = F.softmax(pred_logits, dim=-1)    # (S, C+1)

            # ── Per-slot confidence check ─────────────────────────────────────
            slot_boxes: List[Optional[np.ndarray]] = [None] * num_classes
            slot_confs: List[float]                = [0.0]  * num_classes

            for s in range(num_classes):
                expected_cls = s + 1   # 1-based foreground label for this slot
                conf = float(probs[s, expected_cls].item())
                if conf < args.conf_thresh:
                    continue
                cx, cy, bw, bh = pred_boxes[s].cpu().tolist()
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0
                slot_boxes[s] = np.array([x1, y1, x2, y2], dtype=np.float32)
                slot_confs[s] = conf

            # ── Geometric consistency check ───────────────────────────────────
            # If helmet is detected but its center is not in the upper half of
            # the bicycle box, discard the helmet detection.
            # This filters out false positives caused by the global-pooling head
            # losing precise spatial information.
            if (not args.no_geo_filter
                    and slot_boxes[SLOT_HELMET]  is not None
                    and slot_boxes[SLOT_BICYCLE] is not None):
                if not _helmet_consistent_with_bicycle(
                    slot_boxes[SLOT_HELMET],
                    slot_boxes[SLOT_BICYCLE],
                ):
                    slot_boxes[SLOT_HELMET] = None
                    slot_confs[SLOT_HELMET] = 0.0

            # ── Temporal smoothing ────────────────────────────────────────────
            smooth_boxes, smooth_confs = smoother.update(slot_boxes, slot_confs)

            # ── Draw ──────────────────────────────────────────────────────────
            for s in range(num_classes):
                if smooth_boxes[s] is None:
                    continue
                color = colors[s]
                b     = smooth_boxes[s]
                x1 = int(max(0,   b[0] * W))
                y1 = int(max(0,   b[1] * H))
                x2 = int(min(W-1, b[2] * W))
                y2 = int(min(H-1, b[3] * H))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_names[s]}  {smooth_confs[s]:.2f}"
                (tw, th), bl = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                ty = max(y1 - 4, th + 4)
                cv2.rectangle(
                    frame, (x1, ty - th - bl - 2), (x1 + tw, ty + 2), color, -1
                )
                cv2.putText(
                    frame, label, (x1, ty - bl),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
                )

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"  Frame {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"\nDone. Saved to {args.video_out}")
    print(f"Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()