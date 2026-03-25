import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from .dataset_coco import COCOSingleObjectDataset, default_coco_paths
from .model import MobileNetV3BBox
from .utils import cxcywh_to_xyxy, iou_xyxy


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and prompt for dataset path if missing."""
    parser = argparse.ArgumentParser(description="Train Part 2 detector.")
    # Lecture 1: define hyperparameters explicitly (LR, batch size, epochs).
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--input_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_dir", type=Path, default=Path("outputs/part2_logs"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--category_name", type=str, default="bicycle")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    args = parser.parse_args()
    if args.data_dir is None:
        user_input = input("Enter dataset root path (COCO export): ").strip()
        if not user_input:
            raise ValueError("Dataset path is required.")
        args.data_dir = Path(user_input)
    return args


def make_loader(
    data_dir: Path,
    split: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    category_name: str,
    max_samples: int | None,
) -> DataLoader:
    """Create a DataLoader for a COCO split with optional sample limiting."""
    # Lecture 1: train/val/test split handling is crucial for generalization.
    ann_path, img_dir = default_coco_paths(data_dir, split)
    # Accept multiple class names (e.g., "bicycle,Bicycle") from the COCO file.
    category_names = [name.strip() for name in category_name.split(",") if name.strip()]
    dataset = COCOSingleObjectDataset(
        annotations_path=ann_path,
        images_dir=img_dir,
        category_name=category_names,
        input_size=input_size,
        train=train,
    )
    if max_samples is not None:
        # Lecture 2: small overfit test to verify the pipeline is correct.
        max_samples = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Compute mean SmoothL1 loss and IoU over a dataloader."""
    # Lecture 1: validation is done without gradient updates (eval mode).
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0
    loss_fn = nn.SmoothL1Loss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            # Forward pass: produce bbox predictions from the network.
            preds = model(images)
            # Lecture 1: regression loss for continuous targets (box coords).
            loss = loss_fn(preds, targets)

            # Lecture 2: evaluation with IoU for detection quality.
            preds_xyxy = cxcywh_to_xyxy(preds)
            targets_xyxy = cxcywh_to_xyxy(targets)
            iou = iou_xyxy(preds_xyxy, targets_xyxy).mean().item()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_iou += iou * batch_size
            count += batch_size

    return total_loss / count, total_iou / count


def main() -> None:
    """Train the MobileNetV3 bbox regressor and log metrics to TensorBoard."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Lecture 2: use GPU if available for faster convolutional training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lecture 1: build train/val loaders from separate splits.
    train_loader = make_loader(
        args.data_dir,
        "train",
        args.input_size,
        args.batch_size,
        args.num_workers,
        train=True,
        category_name=args.category_name,
        max_samples=args.max_train_samples,
    )
    val_loader = make_loader(
        args.data_dir,
        "valid",
        args.input_size,
        args.batch_size,
        args.num_workers,
        train=False,
        category_name=args.category_name,
        max_samples=args.max_val_samples,
    )

    # Lecture 2: transfer learning - reuse pretrained MobileNetV3 features.
    model = MobileNetV3BBox(pretrained=True).to(device)
    # Lecture 1: AdamW is a gradient descent variant with weight decay.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Lecture 1: SmoothL1 is robust for bounding-box regression.
    loss_fn = nn.SmoothL1Loss()

    # Lecture requirement: track metrics in TensorBoard.
    writer = SummaryWriter(log_dir=str(args.log_dir))

    best_val_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        # Lecture 1/2: training mode enables dropout, batch norm updates.
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        count = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            # Forward pass (Lecture 2: conv features -> bbox head).
            preds = model(images)
            loss = loss_fn(preds, targets)

            # Lecture 1: backpropagation + gradient descent update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metric computation does not affect gradients.
            preds_xyxy = cxcywh_to_xyxy(preds)
            targets_xyxy = cxcywh_to_xyxy(targets)
            iou = iou_xyxy(preds_xyxy, targets_xyxy).mean().item()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_iou += iou * batch_size
            count += batch_size

        train_loss = running_loss / count
        train_iou = running_iou / count
        # Lecture 1: validate on held-out data each epoch.
        val_loss, val_iou = evaluate(model, val_loader, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("iou/train", train_iou, epoch)
        writer.add_scalar("iou/val", val_iou, epoch)

        print(
            f"Epoch {epoch:02d} | Train loss {train_loss:.4f} IoU {train_iou:.4f} "
            f"| Val loss {val_loss:.4f} IoU {val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            # Save the best validation model for inference.
            best_val_iou = val_iou
            torch.save(model.state_dict(), args.out_dir / "part2_best.pt")

    torch.save(model.state_dict(), args.out_dir / "part2_last.pt")
    writer.close()


if __name__ == "__main__":
    main()
