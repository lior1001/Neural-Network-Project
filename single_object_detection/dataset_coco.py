import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import clamp_box, xyxy_to_cxcywh


@dataclass
class Sample:
    """Holds a single image path and its COCO bbox in xywh format."""
    image_path: Path
    bbox_xywh: Tuple[float, float, float, float]


class COCOSingleObjectDataset(Dataset):
    """COCO dataset wrapper that keeps exactly one object per image."""
    def __init__(
        self,
        annotations_path: Path,
        images_dir: Path,
        category_name: str | list[str] = "bicycle",
        input_size: int = 320,
        train: bool = True,
    ) -> None:
        self.annotations_path = annotations_path
        self.images_dir = images_dir
        if isinstance(category_name, str):
            self.category_names = [category_name]
        else:
            self.category_names = category_name
        self.input_size = input_size
        self.train = train

        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(
                f"No samples found for category '{category_name}' with exactly one "
                "annotation per image."
            )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_samples(self) -> List[Sample]:
        """Parse COCO JSON and return single-object samples for target classes."""
        with self.annotations_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        category_ids = []
        for cat in coco.get("categories", []):
            if cat.get("name") in self.category_names:
                category_ids.append(cat.get("id"))
        if not category_ids:
            raise ValueError(
                f"Categories {self.category_names} not found in COCO JSON."
            )

        images_by_id = {img["id"]: img for img in coco.get("images", [])}
        anns_by_image = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        samples: List[Sample] = []
        for image_id, ann_list in anns_by_image.items():
            if len(ann_list) != 1:
                continue
            ann = ann_list[0]
            if ann.get("category_id") not in category_ids:
                continue
            img_meta = images_by_id.get(image_id)
            if img_meta is None:
                continue
            image_path = self.images_dir / img_meta["file_name"]
            if not image_path.exists():
                continue
            bbox_xywh = tuple(float(v) for v in ann["bbox"])
            samples.append(Sample(image_path=image_path, bbox_xywh=bbox_xywh))

        return samples

    def __len__(self) -> int:
        """Return number of usable samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Load image, apply transforms, and return normalized bbox target."""
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        orig_w, orig_h = image.size

        x, y, w, h = sample.bbox_xywh

        if self.train:
            # Lecture 2: simple data augmentation (horizontal flip).
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                x = orig_w - x - w

        # Lecture 1: normalize inputs using ImageNet statistics.
        image = self.image_transform(image)

        # Scale bbox from original size to resized input size.
        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h
        x = x * scale_x
        y = y * scale_y
        w = w * scale_x
        h = h * scale_y

        # Convert to xyxy then to center-based representation.
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        box_xyxy = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        box_cxcywh = xyxy_to_cxcywh(box_xyxy)

        # Normalize to [0, 1] so the head can use a sigmoid output.
        norm = torch.tensor(
            [self.input_size, self.input_size, self.input_size, self.input_size],
            dtype=torch.float32,
        )
        box_cxcywh = box_cxcywh / norm
        box_cxcywh = clamp_box(box_cxcywh, 0.0, 1.0)

        return image, box_cxcywh


def default_coco_paths(data_dir: Path, split: str) -> Tuple[Path, Path]:
    """Return default COCO annotation path and image directory for a split."""
    split_dir = data_dir / split
    annotations_path = split_dir / "_annotations.coco.json"
    images_dir = split_dir
    return annotations_path, images_dir
