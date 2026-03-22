from __future__ import annotations

import argparse
from pathlib import Path

from coco_utils import (
    annotations_by_image,
    copy_image,
    ensure_unique_name,
    hamming_distance,
    image_ahash,
    image_sha1,
    load_coco_split,
    save_coco,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove duplicate images from one COCO split against reference splits.")
    p.add_argument("--source_dir", type=Path, required=True,
                   help="COCO split directory containing images and _annotations.coco.json.")
    p.add_argument("--reference_dir", action="append", type=Path, required=True,
                   help="Reference COCO split directory. Repeat to compare against multiple splits.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--ahash_threshold", type=int, default=0,
                   help="Max Hamming distance for average-hash duplicate matching. 0 = exact aHash match only.")
    return p.parse_args()


def build_reference_signatures(reference_dirs: list[Path]) -> tuple[set[str], list[int]]:
    sha1s: set[str] = set()
    ahashes: list[int] = []
    for ref_dir in reference_dirs:
        ref_coco, ref_images_dir = load_coco_split(ref_dir)
        for image_meta in ref_coco.get("images", []):
            path = ref_images_dir / image_meta["file_name"]
            if not path.exists():
                continue
            sha1s.add(image_sha1(path))
            ahashes.append(image_ahash(path))
    return sha1s, ahashes


def is_duplicate(path: Path, ref_sha1s: set[str], ref_ahashes: list[int], threshold: int) -> bool:
    sha1 = image_sha1(path)
    if sha1 in ref_sha1s:
        return True

    candidate_hash = image_ahash(path)
    return any(hamming_distance(candidate_hash, ref_hash) <= threshold for ref_hash in ref_ahashes)


def main() -> None:
    args = parse_args()
    source_coco, source_images_dir = load_coco_split(args.source_dir)
    anns_by_image = annotations_by_image(source_coco)
    ref_sha1s, ref_ahashes = build_reference_signatures(args.reference_dir)

    used_names: set[str] = set()
    out_images: list[dict] = []
    out_annotations: list[dict] = []
    next_image_id = 1
    next_ann_id = 1
    skipped = 0

    for image_meta in source_coco.get("images", []):
        src_path = source_images_dir / image_meta["file_name"]
        if not src_path.exists():
            continue
        if is_duplicate(src_path, ref_sha1s, ref_ahashes, args.ahash_threshold):
            skipped += 1
            continue

        out_name = ensure_unique_name(Path(image_meta["file_name"]).name, used_names)
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
        "categories": source_coco.get("categories", []),
    }
    save_coco(out_coco, args.output_dir)
    print(
        f"Saved {len(out_images)} images to {args.output_dir} "
        f"after removing {skipped} duplicates"
    )


if __name__ == "__main__":
    main()
