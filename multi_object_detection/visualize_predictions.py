"""Visualize Part 3 predictions on still images."""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import v2 as T

from dataset_coco import COCOMultiObjectDataset, default_coco_paths
from model import MobileNetV3MultiBBox

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SLOT_BICYCLE = 0
SLOT_HELMET = 1


def _helmet_consistent_with_bicycle(helmet_box, bicycle_box) -> bool:
    hx = (helmet_box[0] + helmet_box[2]) / 2.0
    hy = (helmet_box[1] + helmet_box[3]) / 2.0

    bx1, by1, bx2, by2 = bicycle_box
    b_height = by2 - by1
    b_width = bx2 - bx1
    b_mid_y = by1 + b_height * 0.5

    in_y = hy < b_mid_y
    margin_x = b_width * 0.30
    in_x = (bx1 - margin_x) <= hx <= (bx2 + margin_x)

    h_area = (helmet_box[2] - helmet_box[0]) * (helmet_box[3] - helmet_box[1])
    b_area = b_width * b_height
    size_ok = h_area < b_area * 0.60

    h_w = helmet_box[2] - helmet_box[0]
    h_h = helmet_box[3] - helmet_box[1]
    not_tiny = h_w > 0.02 and h_h > 0.02
    return in_y and in_x and size_ok and not_tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Part 3 predictions on still images.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--category_name", type=str, default="Bicycle,Helmet")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input_size", type=int, default=320)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/part3_pred_samples"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draw_gt", action="store_true", help="Draw ground-truth boxes in green.")
    parser.add_argument("--conf_thresh", type=float, default=0.55)
    parser.add_argument("--image_subset_dir", type=Path, default=None,
                        help="Optional directory of images whose filenames define a visualization subset.")
    parser.add_argument("--require_all_classes", action="store_true",
                        help="Only visualize images whose ground truth contains all requested classes.")
    parser.add_argument("--apply_geo_filter", action="store_true",
                        help="Apply the same helmet-vs-bicycle geometric filter used in video inference.")
    return parser.parse_args()


def _norm_cxcywh_to_px_xyxy(box, orig_w: int, orig_h: int):
    cx, cy, bw, bh = box
    x1 = max(0, int((cx - bw / 2.0) * orig_w))
    y1 = max(0, int((cy - bh / 2.0) * orig_h))
    x2 = min(orig_w - 1, int((cx + bw / 2.0) * orig_w))
    y2 = min(orig_h - 1, int((cy + bh / 2.0) * orig_h))
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    category_names = [s.strip() for s in args.category_name.split(",") if s.strip()]
    if len(category_names) < 2:
        raise ValueError("Need at least 2 categories.")

    ann_path, img_dir = default_coco_paths(args.data_dir, args.split)
    dataset = COCOMultiObjectDataset(
        annotations_path=ann_path,
        images_dir=img_dir,
        category_names=category_names,
        input_size=args.input_size,
        train=False,
        include_filenames_from_dir=args.image_subset_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3MultiBBox(
        num_classes=len(category_names),
        num_slots=len(category_names),
        pretrained=False,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state", ckpt)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    preprocess = T.Compose([
        T.ToImage(),
        T.Resize((args.input_size, args.input_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    candidate_indices = list(range(len(dataset)))
    if args.require_all_classes:
        candidate_indices = [
            idx for idx in candidate_indices
            if all(ann is not None for ann in dataset._samples[idx]["best"])
        ]
        if not candidate_indices:
            raise RuntimeError(
                f"No {args.split} images contain all requested classes: {category_names}"
            )

    sample_count = min(args.num_samples, len(candidate_indices))
    random.seed(args.seed)
    picked = random.sample(candidate_indices, sample_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx in picked:
        sample = dataset._samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        orig_w, orig_h = image.size

        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            pred_boxes, pred_logits = model(input_tensor)
        pred_boxes = pred_boxes[0].cpu()
        pred_logits = pred_logits[0].cpu()
        probs = F.softmax(pred_logits, dim=-1)

        slot_boxes = [None] * len(category_names)
        slot_confs = [0.0] * len(category_names)
        for s in range(len(category_names)):
            expected_cls = s + 1
            conf = float(probs[s, expected_cls].item())
            if conf < args.conf_thresh:
                continue
            slot_boxes[s] = pred_boxes[s].tolist()
            slot_confs[s] = conf

        if (args.apply_geo_filter
                and len(category_names) >= 2
                and slot_boxes[SLOT_BICYCLE] is not None
                and slot_boxes[SLOT_HELMET] is not None):
            bicycle_xyxy = [
                pred_boxes[SLOT_BICYCLE][0].item() - pred_boxes[SLOT_BICYCLE][2].item() / 2.0,
                pred_boxes[SLOT_BICYCLE][1].item() - pred_boxes[SLOT_BICYCLE][3].item() / 2.0,
                pred_boxes[SLOT_BICYCLE][0].item() + pred_boxes[SLOT_BICYCLE][2].item() / 2.0,
                pred_boxes[SLOT_BICYCLE][1].item() + pred_boxes[SLOT_BICYCLE][3].item() / 2.0,
            ]
            helmet_xyxy = [
                pred_boxes[SLOT_HELMET][0].item() - pred_boxes[SLOT_HELMET][2].item() / 2.0,
                pred_boxes[SLOT_HELMET][1].item() - pred_boxes[SLOT_HELMET][3].item() / 2.0,
                pred_boxes[SLOT_HELMET][0].item() + pred_boxes[SLOT_HELMET][2].item() / 2.0,
                pred_boxes[SLOT_HELMET][1].item() + pred_boxes[SLOT_HELMET][3].item() / 2.0,
            ]
            if not _helmet_consistent_with_bicycle(helmet_xyxy, bicycle_xyxy):
                slot_boxes[SLOT_HELMET] = None
                slot_confs[SLOT_HELMET] = 0.0

        draw = ImageDraw.Draw(image)

        if args.draw_gt:
            for slot, ann in enumerate(sample["best"]):
                if ann is None:
                    continue
                x, y, w, h = ann.bbox_xywh
                draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)
                draw.text((x, max(0, y - 14)), f"GT {category_names[slot]}", fill="lime")

        for s, box in enumerate(slot_boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = _norm_cxcywh_to_px_xyxy(box, orig_w, orig_h)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = f"{category_names[s]} {slot_confs[s]:.2f}"
            draw.text((x1, max(0, y1 - 14)), label, fill="red")

        out_path = args.out_dir / sample["path"].name
        image.save(out_path)

    qualifier = " (all classes present only)" if args.require_all_classes else ""
    print(f"Saved {sample_count} prediction samples{qualifier} to {args.out_dir}")


if __name__ == "__main__":
    main()
