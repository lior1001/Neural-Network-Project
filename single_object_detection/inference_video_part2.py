import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from .model import MobileNetV3BBox
from .utils import cxcywh_to_xyxy, denormalize_box


def parse_args() -> argparse.Namespace:
    """Parse CLI args and prompt for paths if missing."""
    parser = argparse.ArgumentParser(description="Run Part 2 inference on a video.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--video_in", type=Path, default=None)
    parser.add_argument("--video_out", type=Path, default=None)
    parser.add_argument("--input_size", type=int, default=320)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = Path(input("Enter checkpoint path: ").strip())
    if args.video_in is None:
        args.video_in = Path(input("Enter input video path: ").strip())
    if args.video_out is None:
        args.video_out = Path(input("Enter output video path: ").strip())
    return args


def main() -> None:
    """Load a trained model, run bbox inference, and save an annotated video."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileNetV3BBox(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Use the same normalization as training.
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    cap = cv2.VideoCapture(str(args.video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.video_out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.video_out), fourcc, fps, (width, height))

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

            # Predict normalized bbox, then scale back to pixel coordinates.
            pred = model(input_tensor)[0]
            pred_px = denormalize_box(pred, (width, height))
            pred_xyxy = cxcywh_to_xyxy(pred_px)
            x1, y1, x2, y2 = pred_xyxy.int().tolist()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "bicycle",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
