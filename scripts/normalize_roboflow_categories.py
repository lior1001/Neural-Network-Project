"""
Rewrite _annotations.coco.json so categories match merge_coco_splits / train_part3:

  id 1 Bicycle, id 2 Helmet, id 3 No_helmet

Roboflow exports often include an extra id=0 combined label; annotations already use 1–3.
Run once per split directory (train / valid / test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CANONICAL = [
    {"id": 1, "name": "Bicycle", "supercategory": "none"},
    {"id": 2, "name": "Helmet", "supercategory": "none"},
    {"id": 3, "name": "No_helmet", "supercategory": "none"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize Roboflow COCO categories to Bicycle/Helmet/No_helmet (ids 1–3).")
    p.add_argument("--split_dir", type=Path, required=True, help="Directory with _annotations.coco.json and images.")
    p.add_argument("--dry_run", action="store_true", help="Print only; do not write.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ann_path = args.split_dir / "_annotations.coco.json"
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    coco["categories"] = list(CANONICAL)

    if args.dry_run:
        print(f"[dry_run] Would write {ann_path} with {len(coco['images'])} images")
        return

    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"Updated categories in {ann_path}")


if __name__ == "__main__":
    main()
