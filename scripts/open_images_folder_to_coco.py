"""
Build a COCO split from Open Images–style downloaded images + train/labels/detections.csv.

- Filenames must be like <ImageID>.jpg (Open Images image id + extension).
- Streams the huge detections.csv and keeps rows for your images only.
- Keeps images that have exactly one Bicycle and one Helmet box (LabelName /m/0199g and /m/0zvk5).
- Output categories: Bicycle, Helmet, No_helmet (ids 1,2,3) with only Bicycle+Helmet annotated.

Example:
  python scripts/open_images_folder_to_coco.py ^
    --images_dir "data/open-images-v7/train/images" ^
    --detections_csv "data/open-images-v7/train/labels/detections.csv" ^
    --output_dir "data/open_images_clean/train"
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

# From Open Images v7 classes.csv (see train/metadata/classes.csv)
LABEL_BICYCLE = "/m/0199g"
LABEL_HELMET = "/m/0zvk5"

CANONICAL_CATEGORIES = [
    {"id": 1, "name": "Bicycle", "supercategory": "none"},
    {"id": 2, "name": "Helmet", "supercategory": "none"},
    {"id": 3, "name": "No_helmet", "supercategory": "none"},
]
LABEL_TO_CAT_ID = {LABEL_BICYCLE: 1, LABEL_HELMET: 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open Images folder + CSV → COCO (1 bicycle + 1 helmet per image).")
    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--detections_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk when reading detections.csv.",
    )
    return p.parse_args()


def image_ids_in_folder(images_dir: Path) -> dict[str, Path]:
    """Map Open Images ImageID -> path for common extensions."""
    out: dict[str, Path] = {}
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        for p in images_dir.glob(f"*{ext}"):
            stem = p.stem
            out[stem] = p
    return out


def main() -> None:
    args = parse_args()
    id_to_path = image_ids_in_folder(args.images_dir)
    if not id_to_path:
        raise RuntimeError(f"No images found under {args.images_dir}")

    want_ids = set(id_to_path.keys())
    # image_id -> label -> list of rows (dict)
    boxes: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    print(f"[scan] {args.detections_csv} for {len(want_ids)} local image ids …")
    for chunk in pd.read_csv(args.detections_csv, chunksize=args.chunksize):
        chunk = chunk[chunk["ImageID"].isin(want_ids)]
        chunk = chunk[chunk["LabelName"].isin([LABEL_BICYCLE, LABEL_HELMET])]
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            img_id = row["ImageID"]
            lab = row["LabelName"]
            boxes[img_id][lab].append(
                {
                    "XMin": float(row["XMin"]),
                    "YMin": float(row["YMin"]),
                    "XMax": float(row["XMax"]),
                    "YMax": float(row["YMax"]),
                }
            )

    out_images: list[dict] = []
    out_anns: list[dict] = []
    next_iid = 1
    next_aid = 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_id, path in sorted(id_to_path.items(), key=lambda x: x[0]):
        per = boxes.get(image_id, {})
        b_list = per.get(LABEL_BICYCLE, [])
        h_list = per.get(LABEL_HELMET, [])
        if len(b_list) != 1 or len(h_list) != 1:
            continue

        with Image.open(path) as im:
            w, h = im.size

        def row_to_bbox_xywh(r: dict) -> list[float]:
            xmin, ymin, xmax, ymax = r["XMin"] * w, r["YMin"] * h, r["XMax"] * w, r["YMax"] * h
            bw, bh = xmax - xmin, ymax - ymin
            return [float(xmin), float(ymin), float(bw), float(bh)]

        out_name = path.name
        dst = args.output_dir / out_name
        if not dst.exists():
            shutil.copy2(path, dst)

        out_images.append({"id": next_iid, "width": w, "height": h, "file_name": out_name})

        for lab, rows in ((LABEL_BICYCLE, b_list), (LABEL_HELMET, h_list)):
            bbox = row_to_bbox_xywh(rows[0])
            area = bbox[2] * bbox[3]
            out_anns.append(
                {
                    "id": next_aid,
                    "image_id": next_iid,
                    "category_id": LABEL_TO_CAT_ID[lab],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            next_aid += 1
        next_iid += 1

    coco = {"images": out_images, "annotations": out_anns, "categories": list(CANONICAL_CATEGORIES)}
    ann_path = args.output_dir / "_annotations.coco.json"
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"[ok] {len(out_images)} images, {len(out_anns)} annotations → {args.output_dir}")


if __name__ == "__main__":
    main()
