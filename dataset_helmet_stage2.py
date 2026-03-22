"""
Stage 2 dataset: crop around cyclist/bicycle, regress helmet box in the crop.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


@dataclass(frozen=True)
class _Ann:
    image_id: int
    category_id: int
    bbox_xywh: Tuple[float, float, float, float]
    area: float


def make_stage2_crop_xyxy(
    bicycle_bbox_xywh: Tuple[float, float, float, float],
    img_w: float,
    img_h: float,
) -> Tuple[int, int, int, int]:
    """
    Build a crop focused on the rider's upper body and bike cockpit area.

    The crop intentionally keeps more horizontal context than vertical context
    because forward-leaning cyclists move the helmet toward the handlebars.
    """
    x, y, w, h = bicycle_bbox_xywh
    x1 = max(0.0, x - 0.10 * w)
    y1 = max(0.0, y - 0.12 * h)
    x2 = min(img_w, x + 1.10 * w)
    y2 = min(img_h, y + 0.58 * h)

    # Enforce at least a 2×2 crop to avoid degenerate boxes.
    if x2 <= x1 + 1:
        x2 = min(img_w, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(img_h, y1 + 2)
    return int(x1), int(y1), int(x2), int(y2)


class HelmetCropDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        annotations_path: Path,
        images_dir: Path,
        bicycle_name: str = "Bicycle",
        helmet_name: str = "Helmet",
        input_size: int = 224,
        train: bool = True,
        require_all_classes: bool = True,
        max_instances_per_class: Optional[int] = 1,
        blur_p: float = 0.0,
        include_filenames_from_dir: Optional[Path] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.input_size = input_size
        self.train = train
        self.require_all_classes = require_all_classes
        self.max_instances_per_class = max_instances_per_class
        self.blur_p = blur_p
        self.include_filenames_from_dir = (
            Path(include_filenames_from_dir) if include_filenames_from_dir is not None else None
        )
        self._included_filenames: Optional[set[str]] = None
        if self.include_filenames_from_dir is not None:
            self._included_filenames = {
                p.name for p in self.include_filenames_from_dir.iterdir() if p.is_file()
            }

        self._bicycle_name = bicycle_name
        self._helmet_name = helmet_name
        self._samples: List[dict] = []
        self._coco_id_to_name: dict[int, str] = {}
        self._load(Path(annotations_path))

        self._train_tf = T.Compose([
            T.ToImage(),
            T.Resize((input_size, input_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=blur_p),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        self._eval_tf = T.Compose([
            T.ToImage(),
            T.Resize((input_size, input_size)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def _load(self, ann_path: Path) -> None:
        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        cat_name_to_id = {cat["name"]: cat["id"] for cat in coco.get("categories", [])}
        if self._bicycle_name not in cat_name_to_id or self._helmet_name not in cat_name_to_id:
            raise ValueError(
                f"Could not find both {self._bicycle_name!r} and {self._helmet_name!r} in {ann_path}"
            )
        bicycle_id = cat_name_to_id[self._bicycle_name]
        helmet_id = cat_name_to_id[self._helmet_name]
        self._coco_id_to_name = {bicycle_id: self._bicycle_name, helmet_id: self._helmet_name}

        images_meta = {img["id"]: img for img in coco.get("images", [])}
        anns_by_image: dict[int, dict[str, list[_Ann]]] = {}
        for ann in coco.get("annotations", []):
            if ann["category_id"] not in (bicycle_id, helmet_id):
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            key = self._coco_id_to_name[ann["category_id"]]
            anns_by_image.setdefault(ann["image_id"], {"Bicycle": [], "Helmet": []})[key].append(
                _Ann(
                    image_id=ann["image_id"],
                    category_id=ann["category_id"],
                    bbox_xywh=(x, y, w, h),
                    area=w * h,
                )
            )

        for image_id, grouped in anns_by_image.items():
            meta = images_meta.get(image_id)
            if meta is None:
                continue
            img_path = self.images_dir / meta["file_name"]
            if not img_path.exists():
                continue
            if self._included_filenames is not None and img_path.name not in self._included_filenames:
                continue

            bicycle_anns = grouped["Bicycle"]
            helmet_anns = grouped["Helmet"]
            counts = [len(bicycle_anns), len(helmet_anns)]
            if self.require_all_classes and any(count == 0 for count in counts):
                continue
            if self.max_instances_per_class is not None and any(
                count > self.max_instances_per_class for count in counts
            ):
                continue
            if not bicycle_anns or not helmet_anns:
                continue

            self._samples.append({
                "path": img_path,
                "meta": meta,
                "bicycle_best": max(bicycle_anns, key=lambda ann: ann.area),
                "helmet_best": max(helmet_anns, key=lambda ann: ann.area),
                "bicycles": bicycle_anns,
                "helmets": helmet_anns,
            })

        if not self._samples:
            raise RuntimeError(f"No valid stage-2 samples found in {ann_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def _pick_ann_pair(self, sample: dict) -> tuple[_Ann, _Ann]:
        if not self.train:
            return sample["bicycle_best"], sample["helmet_best"]
        bicycle = random.choice(sample["bicycles"])
        helmet = random.choice(sample["helmets"])
        return bicycle, helmet

    def __getitem__(self, idx: int):
        sample = self._samples[idx]
        bicycle_ann, helmet_ann = self._pick_ann_pair(sample)
        image = Image.open(sample["path"]).convert("RGB")
        img_w, img_h = image.size

        crop_x1, crop_y1, crop_x2, crop_y2 = make_stage2_crop_xyxy(
            bicycle_ann.bbox_xywh, float(img_w), float(img_h)
        )
        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        hx, hy, hw, hh = helmet_ann.bbox_xywh
        cx = (hx + hw / 2.0 - crop_x1) / max(crop_x2 - crop_x1, 1)
        cy = (hy + hh / 2.0 - crop_y1) / max(crop_y2 - crop_y1, 1)
        nw = hw / max(crop_x2 - crop_x1, 1)
        nh = hh / max(crop_y2 - crop_y1, 1)
        target_box = torch.tensor([cx, cy, nw, nh], dtype=torch.float32).clamp(0.0, 1.0)

        tf = self._train_tf if self.train else self._eval_tf
        crop = tf(crop)
        return crop, target_box


def default_coco_paths(data_dir: Path, split: str):
    split_dir = Path(data_dir) / split
    return split_dir / "_annotations.coco.json", split_dir
