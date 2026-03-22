import fiftyone as fo
import fiftyone.zoo as foz

# Load images containing both Bicycle and Helmet
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Bicycle", "Helmet"],
    max_samples=500 
)

# Export to COCO JSON format
dataset.export(
    export_dir="./coco_bicycle_helmet",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)
