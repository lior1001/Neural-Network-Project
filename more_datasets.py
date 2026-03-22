from __future__ import annotations

import json
import tarfile
import time
import requests
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse


# =========================
# Configuration
# =========================

# Choose: "val" or "train"
SPLIT = "train"

ROOT = Path("Objects365_bicycle_helmet_coco")
TARGET_CLASSES = ["Helmet", "Bicycle"]

# Optional cap on number of matching images to keep
# Set to None for all matches
MAX_IMAGES = 300

BASE_URL = (
    "https://dorc.ks3-cn-beijing.ksyun.com/data-set/"
    "2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/"
)

if SPLIT not in {"val", "train"}:
    raise ValueError("SPLIT must be 'val' or 'train'")

SPLIT_ROOT = ROOT / SPLIT
PATCHES_DIR = SPLIT_ROOT / "patches"
IMAGES_DIR = SPLIT_ROOT / "images"
ANNOTATIONS_DIR = SPLIT_ROOT / "annotations"

PATCHES_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

if SPLIT == "val":
    ANNOTATIONS_DOWNLOAD_URL = f"{BASE_URL}val/zhiyuan_objv2_val.json"
    RAW_ANNOTATIONS_PATH = SPLIT_ROOT / "zhiyuan_objv2_val.json"
    EXTRACTED_JSON_PATH = RAW_ANNOTATIONS_PATH
else:
    ANNOTATIONS_DOWNLOAD_URL = f"{BASE_URL}train/zhiyuan_objv2_train.tar.gz"
    RAW_ANNOTATIONS_PATH = SPLIT_ROOT / "zhiyuan_objv2_train.tar.gz"
    EXTRACTED_JSON_PATH = SPLIT_ROOT / "zhiyuan_objv2_train.json"

OUTPUT_JSON_PATH = ANNOTATIONS_DIR / "annotations.json"


def download_file(url: str, dest: Path, timeout: int = 120, retries: int = 5) -> None:
    # If file already exists and is non-empty, keep it
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] Already exists: {dest}")
        return

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"[download attempt {attempt}/{retries}] {url}")

            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # success
            if dest.exists() and dest.stat().st_size > 0:
                return

            raise RuntimeError(f"Downloaded file is empty: {dest}")

        except Exception as e:
            last_error = e
            print(f"[download failed attempt {attempt}] {url} -> {e}")

            # Remove partial file before retrying
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass

            if attempt < retries:
                wait_seconds = 5 * attempt
                print(f"[retrying in {wait_seconds}s]")
                time.sleep(wait_seconds)

    raise last_error


def extract_train_json_if_needed(tar_path: Path, out_json_path: Path) -> None:
    if out_json_path.exists() and out_json_path.stat().st_size > 0:
        print(f"[skip] Already extracted: {out_json_path}")
        return

    print(f"[extract] {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        json_members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json")]
        if not json_members:
            raise FileNotFoundError("No JSON file found inside train annotations tar.gz")

        member = json_members[0]
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Failed to extract {member.name}")

        with open(out_json_path, "wb") as f:
            f.write(extracted.read())

    print(f"[ok] Extracted JSON to {out_json_path}")


def patch_url_from_file_name(file_name: str, split: str) -> str:
    """
    val example:
      images/v2/patch16/objects365_v2_00909034.jpg
      -> .../val/images/v2/patch16.tar.gz

    train example:
      patch6/objects365_v1_00320532.jpg
      or similar directory structure under train
      -> .../train/patch6.tar.gz
    """
    parts = Path(file_name).parts

    if split == "val":
        if len(parts) < 4:
            raise ValueError(f"Unexpected val file_name format: {file_name}")
        version_dir = parts[1]   # v1 or v2
        patch_dir = parts[2]     # patch16, ...
        return f"{BASE_URL}val/images/{version_dir}/{patch_dir}.tar.gz"

    # train
    patch_dir = None
    for p in parts:
        if p.startswith("patch"):
            patch_dir = p
            break

    if patch_dir is None:
        raise ValueError(f"Could not find patch directory in train file_name: {file_name}")

    return f"{BASE_URL}train/{patch_dir}.tar.gz"


def patch_local_name_from_url(url: str) -> str:
    path = urlparse(url).path
    return Path(path).name


def find_member_in_tar_by_basename(tar: tarfile.TarFile, basename: str):
    for member in tar.getmembers():
        if member.isfile() and Path(member.name).name == basename:
            return member
    return None


def main() -> None:
    # -------------------------
    # 1) Download annotations
    # -------------------------
    download_file(ANNOTATIONS_DOWNLOAD_URL, RAW_ANNOTATIONS_PATH)

    if SPLIT == "train":
        extract_train_json_if_needed(RAW_ANNOTATIONS_PATH, EXTRACTED_JSON_PATH)

    # -------------------------
    # 2) Load JSON
    # -------------------------
    print(f"[load] {EXTRACTED_JSON_PATH}")
    with open(EXTRACTED_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    image_id_to_image = {img["id"]: img for img in images}

    for cls in TARGET_CLASSES:
        if cls not in set(cat_id_to_name.values()):
            raise ValueError(f"Class '{cls}' not found in categories")

    target_cat_ids = {
        cat_id for cat_id, name in cat_id_to_name.items()
        if name in TARGET_CLASSES
    }

    # -------------------------
    # 3) Group target anns by image
    # -------------------------
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        if ann["category_id"] in target_cat_ids:
            anns_by_image[ann["image_id"]].append(ann)

    # -------------------------
    # 4) Keep exactly one Helmet and one Bicycle
    # -------------------------
    matching_image_ids: list[int] = []
    matching_annotations: dict[int, list[dict]] = {}

    for image_id, anns in anns_by_image.items():
        helmet_anns = []
        bicycle_anns = []

        for ann in anns:
            class_name = cat_id_to_name[ann["category_id"]]
            if class_name == "Helmet":
                helmet_anns.append(ann)
            elif class_name == "Bicycle":
                bicycle_anns.append(ann)

        if len(helmet_anns) == 1 and len(bicycle_anns) == 1:
            matching_image_ids.append(image_id)
            matching_annotations[image_id] = helmet_anns + bicycle_anns

    print(f"[match] Found {len(matching_image_ids)} matching images in {SPLIT}")

    if MAX_IMAGES is not None:
        matching_image_ids = matching_image_ids[:MAX_IMAGES]
        print(f"[limit] Using first {len(matching_image_ids)} matches")

    if not matching_image_ids:
        print("[done] No matching images found.")
        return

    # -------------------------
    # 5) Group matches by needed patch archive
    # -------------------------
    patch_to_items: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    for image_id in matching_image_ids:
        img = image_id_to_image[image_id]
        file_name = img["file_name"]
        patch_url = patch_url_from_file_name(file_name, SPLIT)
        patch_to_items[patch_url].append((image_id, img))

    print(f"[patches] Need {len(patch_to_items)} patch archive(s)")

    # -------------------------
    # 6) Download only needed patches
    # -------------------------
    for patch_url in patch_to_items:
        local_patch = PATCHES_DIR / patch_local_name_from_url(patch_url)
        download_file(patch_url, local_patch)

    # -------------------------
    # 7) Extract only matching images
    # -------------------------
    output_categories = [
        {"id": 1, "name": "Helmet", "supercategory": "object"},
        {"id": 2, "name": "Bicycle", "supercategory": "object"},
    ]
    category_name_to_new_id = {"Helmet": 1, "Bicycle": 2}

    downloaded_images = []
    downloaded_annotations = []
    new_ann_id = 1
    extracted_count = 0
    missing_in_tar = 0

    for patch_url, items in patch_to_items.items():
        local_patch = PATCHES_DIR / patch_local_name_from_url(patch_url)
        print(f"[extract] {local_patch}")

        with tarfile.open(local_patch, "r:gz") as tar:
            for image_id, img in items:
                basename = Path(img["file_name"]).name
                member = find_member_in_tar_by_basename(tar, basename)

                if member is None:
                    missing_in_tar += 1
                    print(f"[missing in tar] {basename}")
                    continue

                out_path = IMAGES_DIR / basename
                if not out_path.exists():
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        print(f"[extract failed] {basename}")
                        continue
                    with open(out_path, "wb") as f:
                        f.write(extracted.read())

                downloaded_images.append(
                    {
                        "id": image_id,
                        "width": img["width"],
                        "height": img["height"],
                        "file_name": basename,
                    }
                )

                for ann in matching_annotations[image_id]:
                    cls_name = cat_id_to_name[ann["category_id"]]
                    downloaded_annotations.append(
                        {
                            "id": new_ann_id,
                            "image_id": image_id,
                            "category_id": category_name_to_new_id[cls_name],
                            "bbox": ann["bbox"],
                            "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                            "iscrowd": ann.get("iscrowd", 0),
                        }
                    )
                    new_ann_id += 1

                extracted_count += 1
                print(f"[ok] {basename}")

    # -------------------------
    # 8) Save filtered COCO JSON
    # -------------------------
    output_coco = {
        "images": downloaded_images,
        "annotations": downloaded_annotations,
        "categories": output_categories,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_coco, f, indent=2)

    print("\n[summary]")
    print(f"Split: {SPLIT}")
    print(f"Matched in JSON: {len(matching_image_ids)}")
    print(f"Extracted images: {extracted_count}")
    print(f"Missing in tar: {missing_in_tar}")
    print(f"Patches downloaded: {len(patch_to_items)}")
    print(f"Images folder: {IMAGES_DIR}")
    print(f"COCO JSON: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()