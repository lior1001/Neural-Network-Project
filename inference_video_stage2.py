"""
Two-stage video inference:
1. Stage 1 detects the cyclist/bicycle region.
2. Stage 2 localizes the helmet inside a crop around that region.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as T

from dataset_helmet_stage2 import make_stage2_crop_xyxy
from model import MobileNetV3MultiBBox
from model_helmet_stage2 import HelmetCropRegressor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SLOT_BICYCLE = 0


def parse_args():
    p = argparse.ArgumentParser(description="Two-stage bicycle + helmet video inference.")
    p.add_argument("--stage1_checkpoint", type=Path, required=True,
                   help="Path to the full-image detector checkpoint.")
    p.add_argument("--stage2_checkpoint", type=Path, required=True,
                   help="Path to the helmet crop regressor checkpoint.")
    p.add_argument("--video_in", type=Path, required=True)
    p.add_argument("--video_out", type=Path, required=True)
    p.add_argument("--stage1_input_size", type=int, default=448)
    p.add_argument("--stage2_input_size", type=int, default=224)
    p.add_argument("--bike_conf_thresh", type=float, default=0.55)
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


def _stage2_box_to_full_image(box: np.ndarray, crop_xyxy, width: int, height: int):
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

    args.video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = 0
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            stage1_tensor = tf_stage1(pil).unsqueeze(0).to(device)
            pred_boxes, pred_logits = stage1(stage1_tensor)
            pred_boxes = pred_boxes[0]
            pred_logits = pred_logits[0]
            probs = F.softmax(pred_logits, dim=-1)
            bike_conf = float(probs[SLOT_BICYCLE, SLOT_BICYCLE + 1].item())

            if bike_conf >= args.bike_conf_thresh:
                bike_box = pred_boxes[SLOT_BICYCLE].cpu().numpy()
                bx1, by1, bx2, by2 = _norm_cxcywh_to_xyxy_px(bike_box, width, height)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(
                    frame, f"Bicycle {bike_conf:.2f}", (bx1, max(by1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA
                )

                crop_xyxy = make_stage2_crop_xyxy(
                    (bx1, by1, max(bx2 - bx1, 1), max(by2 - by1, 1)), width, height
                )
                crop = pil.crop(crop_xyxy)
                stage2_tensor = tf_stage2(crop).unsqueeze(0).to(device)
                helmet_box = stage2(stage2_tensor)[0].cpu().numpy()
                hx1, hy1, hx2, hy2 = _stage2_box_to_full_image(helmet_box, crop_xyxy, width, height)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)
                cv2.putText(
                    frame, "Helmet (stage2)", (hx1, max(hy1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA
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
