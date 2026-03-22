from __future__ import annotations

import argparse
from pathlib import Path

from coco_utils import (
    annotations_by_image,
    category_name_to_id,
    copy_image,
    ensure_unique_name,
    load_coco,
    preserve_category_order,
    save_coco,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remap and filter a COCO split into a clean dataset.")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--class_map",
        action="append",
        required=True,
        help="Mapping in the form SourceName=TargetName. Repeat per class to keep.",
    )
    p.add_argument("--require_all_classes", action="store_true",
                   help="Keep only images that contain all mapped target classes.")
    p.add_argument("--max_instances_per_class", type=int, default=None,
                   help="Keep only images where each target class appears at most this many times.")
    p.add_argument("--image_prefix", type=str, default="",
                   help="Prefix added to copied image filenames.")
    p.add_argument(
        "--output_categories",
        type=str,
        default=None,
        help="Comma-separated names for the output JSON categories (ids 1..N in order). "
             "Use e.g. Bicycle,Helmet,No_helmet to match train_part3 / Roboflow. "
             "Only classes that appear in --class_map get annotations; others exist so "
             "merge_coco_splits can match your main dataset.",
    )
    return p.parse_args()


def parse_class_map(entries: list[str]) -> tuple[dict[str, str], list[str]]:
    source_to_target: dict[str, str] = {}
    ordered_targets: list[str] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --class_map '{entry}'. Expected Source=Target.")
        source, target = [part.strip() for part in entry.split("=", 1)]
        if not source or not target:
            raise ValueError(f"Invalid --class_map '{entry}'.")
        source_to_target[source] = target
        ordered_targets.append(target)
    return source_to_target, preserve_category_order(ordered_targets)


def main() -> None:
    args = parse_args()
    coco = load_coco(args.annotations)
    anns_by_image = annotations_by_image(coco)
    source_to_target, mapped_target_names = parse_class_map(args.class_map)
    source_name_to_id = category_name_to_id(coco)

    source_id_to_target: dict[int, str] = {}
    for source_name, target_name in source_to_target.items():
        if source_name not in source_name_to_id:
            raise ValueError(f"Source category '{source_name}' not found in {args.annotations}")
        source_id_to_target[source_name_to_id[source_name]] = target_name

    if args.output_categories is not None:
        output_names = [c.strip() for c in args.output_categories.split(",") if c.strip()]
        if not output_names:
            raise ValueError("--output_categories must list at least one name.")
        for t in mapped_target_names:
            if t not in output_names:
                raise ValueError(
                    f"Mapped target '{t}' from --class_map must appear in --output_categories."
                )
        target_names_for_ids = output_names
    else:
        target_names_for_ids = mapped_target_names

    target_name_to_new_id = {name: idx + 1 for idx, name in enumerate(target_names_for_ids)}
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "merged"}
        for idx, name in enumerate(target_names_for_ids)
    ]

    images_meta = {img["id"]: img for img in coco.get("images", [])}
    used_names: set[str] = set()
    out_images: list[dict] = []
    out_annotations: list[dict] = []
    next_image_id = 1
    next_ann_id = 1

    for old_image_id, image_meta in images_meta.items():
        image_anns = anns_by_image.get(old_image_id, [])
        kept_anns = [ann for ann in image_anns if ann["category_id"] in source_id_to_target]
        if not kept_anns:
            continue

        by_target: dict[str, list[dict]] = {name: [] for name in mapped_target_names}
        for ann in kept_anns:
            by_target[source_id_to_target[ann["category_id"]]].append(ann)

        if args.require_all_classes and any(len(by_target[name]) == 0 for name in mapped_target_names):
            continue
        if args.max_instances_per_class is not None and any(
            len(by_target[name]) > args.max_instances_per_class for name in mapped_target_names
        ):
            continue

        src_image_path = args.images_dir / image_meta["file_name"]
        if not src_image_path.exists():
            continue

        out_name = ensure_unique_name(f"{args.image_prefix}{Path(image_meta['file_name']).name}", used_names)
        copy_image(src_image_path, args.output_dir, out_name)

        new_image = dict(image_meta)
        new_image["id"] = next_image_id
        new_image["file_name"] = out_name
        out_images.append(new_image)

        for ann in kept_anns:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = next_image_id
            new_ann["category_id"] = target_name_to_new_id[source_id_to_target[ann["category_id"]]]
            out_annotations.append(new_ann)
            next_ann_id += 1

        next_image_id += 1

    out_coco = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": categories,
    }
    save_coco(out_coco, args.output_dir)
    print(
        f"Saved {len(out_images)} images and {len(out_annotations)} annotations "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
