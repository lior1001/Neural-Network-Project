"""
Keep only images where annotations match strict "one cyclist + helmet" rules:

  - Exactly **one** Bicycle box
  - Exactly **one** Helmet box
  - **Zero** No_helmet boxes (if that category exists in the JSON)

This matches common filtering goals:
  - drop multi-rider / group shots (multiple bicycle boxes)
  - drop "no helmet" scenes (No_helmet label, if present)
  - drop missing or duplicate helmet boxes

Outputs a new COCO split directory (images copied + _annotations.coco.json).

Usage (from project `scripts/` folder):

  python filter_coco_strict_single_cyclist.py ^
    --split_dir "..\\data\\bicycle_helmet_merged\\train" ^
    --output_dir "..\\data\\bicycle_helmet_filtered\\train"
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from coco_utils import (
    annotations_by_image,
    category_name_to_id,
    copy_image,
    ensure_unique_name,
    load_coco_split,
    save_coco,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter COCO: exactly 1 bicycle, 1 helmet, 0 no-helmet.")
    p.add_argument("--split_dir", type=Path, required=True,
                   help="Directory with _annotations.coco.json and images.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print counts only; do not copy images or write JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    coco, images_dir = load_coco_split(args.split_dir)
    name_to_id = category_name_to_id(coco)

    need = ["Bicycle", "Helmet"]
    for n in need:
        if n not in name_to_id:
            raise ValueError(f"Category '{n}' not found. Available: {list(name_to_id.keys())}")

    id_bicycle = name_to_id["Bicycle"]
    id_helmet = name_to_id["Helmet"]
    id_no_helmet = name_to_id.get("No_helmet")

    allowed_ids = {id_bicycle, id_helmet}
    if id_no_helmet is not None:
        allowed_ids.add(id_no_helmet)

    anns_by_image = annotations_by_image(coco)
    images_meta = {img["id"]: img for img in coco.get("images", [])}

    kept_images: list[dict] = []
    kept_anns: list[dict] = []
    next_iid = 1
    next_aid = 1
    used_names: set[str] = set()
    skipped = 0

    for old_img_id, meta in images_meta.items():
        anns = anns_by_image.get(old_img_id, [])
        c = Counter()
        for a in anns:
            c[a["category_id"]] += 1

        if any(cid not in allowed_ids for cid in c):
            skipped += 1
            continue

        n_b = c.get(id_bicycle, 0)
        n_h = c.get(id_helmet, 0)
        n_nh = c.get(id_no_helmet, 0) if id_no_helmet is not None else 0

        if n_b != 1 or n_h != 1 or n_nh != 0:
            skipped += 1
            continue

        src = images_dir / meta["file_name"]
        if not src.exists():
            skipped += 1
            continue

        out_name = ensure_unique_name(meta["file_name"], used_names)
        if not args.dry_run:
            copy_image(src, args.output_dir, out_name)

        new_meta = dict(meta)
        new_meta["id"] = next_iid
        new_meta["file_name"] = out_name
        kept_images.append(new_meta)

        for a in anns:
            if a["category_id"] not in (id_bicycle, id_helmet):
                continue
            na = dict(a)
            na["id"] = next_aid
            na["image_id"] = next_iid
            kept_anns.append(na)
            next_aid += 1

        next_iid += 1

    out_coco = {
        "images": kept_images,
        "annotations": kept_anns,
        "categories": coco.get("categories", []),
    }

    if args.dry_run:
        total = len(images_meta)
        print(f"[dry_run] Would keep {len(kept_images)} / {total} images (skip {skipped}).")
        return

    save_coco(out_coco, args.output_dir)
    print(
        f"Saved {len(kept_images)} images, {len(kept_anns)} annotations to {args.output_dir} "
        f"(skipped {skipped})."
    )


if __name__ == "__main__":
    main()
