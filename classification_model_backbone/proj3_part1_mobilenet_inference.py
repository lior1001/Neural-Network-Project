import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_model(device: torch.device):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories


def list_images(image_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]


@torch.inference_mode()
def run_inference(model, preprocess, categories, image_paths, device, topk=5):
    results = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=topk)

        top_labels = [
            (categories[idx], float(prob))
            for idx, prob in zip(top_idxs.tolist(), top_probs.tolist())
        ]
        results.append((image_path, top_labels))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run MobileNetV3 inference on random images."
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=Path("images"),
        help="Directory to sample images from.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of random images to sample.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k predictions to show.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    random.seed(args.seed)
    sample_count = min(args.num_images, len(image_paths))
    sampled = random.sample(image_paths, sample_count)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, categories = load_model(device)

    results = run_inference(
        model, preprocess, categories, sampled, device=device, topk=args.topk
    )

    print(f"Device: {device}")
    for image_path, top_labels in results:
        print(f"\nImage: {image_path}")
        for label, prob in top_labels:
            print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
