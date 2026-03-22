"""
Part 3 dataset: COCO with fixed-slot multi-object detection.

KEY CHANGE from previous version (grid assignment → fixed slot):
- Each class gets a fixed slot index based on its position in category_names.
  e.g. category_names=["black-pawn","black-knight"] → slot 0=black-pawn, slot 1=black-knight
- At most one instance per class per image: keeps the LARGEST by area (most visible object).
- Missing class in an image → that slot gets class_id=0 (background), zero box.
- Slot order is deterministic and constant across all images and epochs.

Why this works better than grid assignment:
  Grid assignment forces the model to solve a spatial routing problem (which cell
  contains each object?) on top of learning to detect. Fixed slots remove that
  entirely: slot 0 *always* means black-pawn. The model just learns one thing per slot.

Returns:
    image:     (3, H, W)  float, ImageNet-normalised
    boxes:     (num_slots, 4) float, normalized (cx, cy, w, h) in [0,1]; bg slots = zeros
    class_ids: (num_slots,) long, 1..num_classes for fg, 0 for background
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

BACKGROUND_CLASS_ID = 0   # 0 = background; 1..num_classes = foreground


@dataclass(frozen=True)
class _Ann:
    image_id: int
    category_id: int   # COCO numeric id
    bbox_xywh: Tuple[float, float, float, float]
    area: float


class COCOMultiObjectDataset(Dataset):
    """
    Fixed-slot COCO multi-object dataset.

    Slot i corresponds to category_names[i]. Each image gets exactly num_slots
    target entries; missing classes are filled with background.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        annotations_path: Path,
        images_dir: Path,
        category_names: List[str],
        input_size: int = 320,
        num_slots: Optional[int] = None,   # ignored; kept for backwards compat
        train: bool = True,
        require_all_classes: bool = False,
        max_instances_per_class: Optional[int] = None,
        blur_p: float = 0.3,
        include_filenames_from_dir: Optional[Path] = None,
    ) -> None:
        self.images_dir     = Path(images_dir)
        self.category_names = category_names
        self.input_size     = input_size
        self.train          = train
        self.require_all_classes = require_all_classes
        self.max_instances_per_class = max_instances_per_class
        self.blur_p         = blur_p
        self.include_filenames_from_dir = (
            Path(include_filenames_from_dir) if include_filenames_from_dir is not None else None
        )
        self._included_filenames: Optional[set[str]] = None
        if self.include_filenames_from_dir is not None:
            self._included_filenames = {
                p.name for p in self.include_filenames_from_dir.iterdir() if p.is_file()
            }
        # slot i → foreground class label i+1 (label 0 = background)
        self.num_slots = len(category_names)

        # COCO id → slot index (0-based) → label (1-based)
        self._coco_id_to_slot: dict[int, int] = {}
        self._samples: List[dict] = []   # list of {path, slots: List[Optional[Ann]]}

        self._load(Path(annotations_path))

        self._train_tf = T.Compose([
            T.ToImage(),
            T.Resize((input_size, input_size)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=blur_p),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        self._eval_tf = T.Compose([
            T.ToImage(),
            T.Resize((input_size, input_size)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def _group_anns_by_slot(self, anns: List[_Ann]) -> List[List[_Ann]]:
        by_slot: List[List[_Ann]] = [[] for _ in range(self.num_slots)]
        for ann in anns:
            slot = self._coco_id_to_slot.get(ann.category_id)
            if slot is not None:
                by_slot[slot].append(ann)
        return by_slot

    def _load(self, ann_path: Path) -> None:
        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        # Build COCO-id → slot mapping
        name_to_slot = {name: i for i, name in enumerate(self.category_names)}
        for cat in coco.get("categories", []):
            if cat["name"] in name_to_slot:
                self._coco_id_to_slot[cat["id"]] = name_to_slot[cat["name"]]

        if len(self._coco_id_to_slot) < 2:
            raise ValueError(
                f"Found only {len(self._coco_id_to_slot)} matching categories "
                f"for {self.category_names}. Check category names."
            )

        # Group annotations by image
        images_meta = {img["id"]: img for img in coco["images"]}
        anns_by_image: dict[int, List[_Ann]] = {}
        for a in coco.get("annotations", []):
            if a["category_id"] not in self._coco_id_to_slot:
                continue
            x, y, w, h = [float(v) for v in a["bbox"]]
            if w <= 0 or h <= 0:
                continue
            ann = _Ann(
                image_id=a["image_id"],
                category_id=a["category_id"],
                bbox_xywh=(x, y, w, h),
                area=w * h,
            )
            anns_by_image.setdefault(a["image_id"], []).append(ann)

        # Build samples
        for img_id, anns in anns_by_image.items():
            meta = images_meta.get(img_id)
            if meta is None:
                continue
            img_path = self.images_dir / meta["file_name"]
            if not img_path.exists():
                continue
            if self._included_filenames is not None and img_path.name not in self._included_filenames:
                continue

            by_slot = self._group_anns_by_slot(anns)
            counts = [len(candidates) for candidates in by_slot]

            if self.require_all_classes and any(count == 0 for count in counts):
                continue
            if self.max_instances_per_class is not None and any(
                count > self.max_instances_per_class for count in counts
            ):
                continue

            # For each slot, find the largest-area annotation of that class
            best: List[Optional[_Ann]] = [None] * self.num_slots
            for slot, candidates in enumerate(by_slot):
                if candidates:
                    best[slot] = max(candidates, key=lambda ann: ann.area)

            # Only keep images that have at least one foreground object
            if all(b is None for b in best):
                continue

            self._samples.append({
                "path":   img_path,
                "meta":   meta,
                "best":   best,   # List[Optional[_Ann]], length = num_slots
                "all":    anns,   # all annotations for random-sampling during training
                "by_slot": by_slot,
            })

        if not self._samples:
            raise RuntimeError(
                f"No valid samples found for categories {self.category_names}."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def _pick_anns(self, sample: dict) -> List[Optional[_Ann]]:
        """
        Eval: largest instance per class.
        Train: random instance per class (so the same image can show different
               instances of the same class across epochs, increasing effective diversity).
        """
        if not self.train:
            return sample["best"]

        chosen: List[Optional[_Ann]] = [None] * self.num_slots
        for slot, candidates in enumerate(sample["by_slot"]):
            if candidates:
                chosen[slot] = random.choice(candidates)
        return chosen

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self._samples[idx]
        meta   = sample["meta"]
        img_w, img_h = float(meta["width"]), float(meta["height"])

        image = Image.open(sample["path"]).convert("RGB")
        chosen = self._pick_anns(sample)

        # Build target tensors
        boxes     = torch.zeros(self.num_slots, 4, dtype=torch.float32)
        class_ids = torch.zeros(self.num_slots, dtype=torch.long)  # 0 = background

        for slot, ann in enumerate(chosen):
            if ann is None:
                continue
            x, y, w, h = ann.bbox_xywh
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h
            boxes[slot]     = torch.tensor([cx, cy, nw, nh], dtype=torch.float32)
            class_ids[slot] = slot + 1   # 1-based foreground label

        # Augmentation
        do_flip = self.train and torch.rand(()).item() < 0.5

        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # cx → 1 - cx; cy, w, h unchanged
            flipped = boxes.clone()
            fg = class_ids > 0
            flipped[fg, 0] = 1.0 - boxes[fg, 0]
            boxes = flipped

        tf    = self._train_tf if self.train else self._eval_tf
        image = tf(image)
        return image, boxes, class_ids


def default_coco_paths(data_dir: Path, split: str):
    split_dir = Path(data_dir) / split
    return split_dir / "_annotations.coco.json", split_dir