import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset visualization."""
    parser = argparse.ArgumentParser(description="Visualize COCO samples with boxes.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--category_name", type=str, default="bicycle,Bicycle")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/part2_samples"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Draw bounding boxes on random single-object samples and save them."""
    args = parse_args()
    category_names = [name.strip() for name in args.category_name.split(",") if name.strip()]
    ann_path = args.data_dir / args.split / "_annotations.coco.json"
    img_dir = args.data_dir / args.split

    # This script is only for visual sanity checks (no training).
    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    category_ids = {
        cat["id"] for cat in coco.get("categories", []) if cat.get("name") in category_names
    }
    if not category_ids:
        raise ValueError(f"Categories {category_names} not found in COCO JSON.")

    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") not in category_ids:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Keep only images with exactly one target annotation (matches training setup).
    candidates = [
        (image_id, ann_list[0])
        for image_id, ann_list in anns_by_image.items()
        if len(ann_list) == 1
    ]
    if not candidates:
        raise RuntimeError("No single-object samples found for the chosen category.")

    random.seed(args.seed)
    sample_count = min(args.num_samples, len(candidates))
    picked = random.sample(candidates, sample_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for image_id, ann in picked:
        img_meta = images_by_id.get(image_id)
        if img_meta is None:
            continue
        image_path = img_dir / img_meta["file_name"]
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        out_path = args.out_dir / image_path.name
        image.save(out_path)

    print(f"Saved {sample_count} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
