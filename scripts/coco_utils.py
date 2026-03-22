from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image


def load_coco(annotations_path: Path) -> dict:
    with annotations_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(coco: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "_annotations.coco.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f)
    return out_path


def load_coco_split(split_dir: Path) -> tuple[dict, Path]:
    split_dir = Path(split_dir)
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations: {ann_path}")
    return load_coco(ann_path), split_dir


def ensure_unique_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name

    base = Path(name).stem
    suffix = Path(name).suffix
    idx = 1
    while True:
        candidate = f"{base}_{idx}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def copy_image(src: Path, dst_dir: Path, dst_name: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / dst_name)


def annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def image_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def image_ahash(path: Path, size: int = 8) -> int:
    with Image.open(path) as img:
        gray = img.convert("L").resize((size, size))
        pixels = list(gray.getdata())
    mean = sum(pixels) / max(len(pixels), 1)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | int(value >= mean)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def category_name_to_id(coco: dict) -> dict[str, int]:
    return {cat["name"]: cat["id"] for cat in coco.get("categories", [])}


def preserve_category_order(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
