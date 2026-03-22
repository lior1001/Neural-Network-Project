from __future__ import annotations

import argparse
from pathlib import Path

from coco_utils import copy_image, ensure_unique_name, load_coco_split, save_coco


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple COCO split directories into one split.")
    p.add_argument("--input_dir", action="append", type=Path, required=True,
                   help="Input COCO split directory. Repeat once per dataset to merge.")
    p.add_argument("--output_dir", type=Path, required=True)
    return p.parse_args()


def categories_key(categories: list[dict]) -> list[tuple[int, str]]:
    return [(int(cat["id"]), str(cat["name"])) for cat in categories]


def main() -> None:
    args = parse_args()
    loaded = [load_coco_split(input_dir) for input_dir in args.input_dir]
    base_categories = loaded[0][0].get("categories", [])
    base_key = categories_key(base_categories)

    for coco, _ in loaded[1:]:
        if categories_key(coco.get("categories", [])) != base_key:
            raise ValueError("All input datasets must have the same categories in the same order.")

    used_names: set[str] = set()
    out_images: list[dict] = []
    out_annotations: list[dict] = []
    next_image_id = 1
    next_ann_id = 1

    for dataset_idx, (coco, images_dir) in enumerate(loaded):
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        prefix = f"d{dataset_idx}_"
        for image_meta in coco.get("images", []):
            src_path = images_dir / image_meta["file_name"]
            if not src_path.exists():
                continue

            out_name = ensure_unique_name(prefix + Path(image_meta["file_name"]).name, used_names)
            copy_image(src_path, args.output_dir, out_name)

            new_image = dict(image_meta)
            new_image["id"] = next_image_id
            new_image["file_name"] = out_name
            out_images.append(new_image)

            for ann in anns_by_image.get(image_meta["id"], []):
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                out_annotations.append(new_ann)
                next_ann_id += 1

            next_image_id += 1

    out_coco = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": base_categories,
    }
    save_coco(out_coco, args.output_dir)
    print(
        f"Merged {len(args.input_dir)} datasets into {args.output_dir} "
        f"with {len(out_images)} images"
    )


if __name__ == "__main__":
    main()
