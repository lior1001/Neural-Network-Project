"""
Train stage 2 helmet localizer on rider crops.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_helmet_stage2 import HelmetCropDataset, default_coco_paths
from model_helmet_stage2 import HelmetCropRegressor
from utils import box_cxcywh_to_xyxy_normalized


def parse_args():
    p = argparse.ArgumentParser(description="Train stage-2 helmet localizer.")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs_max", type=int, default=40)
    p.add_argument("--head_only_epochs", type=int, default=5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_last_block", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--blur_p", type=float, default=0.0)
    p.add_argument("--plateau_patience", type=int, default=5)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--early_stop_patience", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_dir", type=Path, default=Path("outputs/stage2_helmet_logs"))
    p.add_argument("--out_dir", type=Path, default=Path("outputs/stage2_helmet"))
    p.add_argument("--train_image_subset_dir", type=Path, default=None)
    p.add_argument("--require_all_classes", action="store_true")
    p.add_argument("--max_instances_per_class", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _giou(pred_cxcywh: torch.Tensor, gt_cxcywh: torch.Tensor) -> torch.Tensor:
    p = box_cxcywh_to_xyxy_normalized(pred_cxcywh, size=1.0).clamp(0.0, 1.0)
    g = box_cxcywh_to_xyxy_normalized(gt_cxcywh, size=1.0).clamp(0.0, 1.0)

    ix1 = torch.max(p[..., 0], g[..., 0])
    iy1 = torch.max(p[..., 1], g[..., 1])
    ix2 = torch.min(p[..., 2], g[..., 2])
    iy2 = torch.min(p[..., 3], g[..., 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    ap = (p[..., 2] - p[..., 0]).clamp(0) * (p[..., 3] - p[..., 1]).clamp(0)
    ag = (g[..., 2] - g[..., 0]).clamp(0) * (g[..., 3] - g[..., 1]).clamp(0)
    union = ap + ag - inter
    iou = inter / union.clamp(min=1e-6)

    ex1 = torch.min(p[..., 0], g[..., 0])
    ey1 = torch.min(p[..., 1], g[..., 1])
    ex2 = torch.max(p[..., 2], g[..., 2])
    ey2 = torch.max(p[..., 3], g[..., 3])
    enc = ((ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0)).clamp(min=1e-6)
    return iou - (enc - union) / enc


def compute_stage2_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
    loss_bbox = F.smooth_l1_loss(pred_boxes, gt_boxes, reduction="mean")
    loss_giou = (1.0 - _giou(pred_boxes, gt_boxes)).mean()
    total = loss_bbox + loss_giou
    return total, float(loss_bbox.detach())


def mean_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    p = box_cxcywh_to_xyxy_normalized(pred_boxes, size=1.0).clamp(0.0, 1.0)
    g = box_cxcywh_to_xyxy_normalized(gt_boxes, size=1.0).clamp(0.0, 1.0)
    ix1 = torch.max(p[:, 0], g[:, 0])
    iy1 = torch.max(p[:, 1], g[:, 1])
    ix2 = torch.min(p[:, 2], g[:, 2])
    iy2 = torch.min(p[:, 3], g[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    ap = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
    ag = (g[:, 2] - g[:, 0]).clamp(0) * (g[:, 3] - g[:, 1]).clamp(0)
    union = ap + ag - inter
    return (inter / union.clamp(min=1e-6)).mean()


def make_loaders(args):
    ann, img = default_coco_paths(args.data_dir, "train")
    train_ds = HelmetCropDataset(
        annotations_path=ann,
        images_dir=img,
        input_size=args.input_size,
        train=True,
        require_all_classes=args.require_all_classes,
        max_instances_per_class=args.max_instances_per_class,
        blur_p=args.blur_p,
        include_filenames_from_dir=args.train_image_subset_dir,
    )
    ann_v, img_v = default_coco_paths(args.data_dir, "valid")
    val_ds = HelmetCropDataset(
        annotations_path=ann_v,
        images_dir=img_v,
        input_size=args.input_size,
        train=False,
        require_all_classes=args.require_all_classes,
        max_instances_per_class=args.max_instances_per_class,
        blur_p=0.0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    return train_loader, val_loader


def _run_epoch(model, loader, device, optimizer, *, is_train: bool):
    model.train(is_train)
    total_loss = total_bbox = total_iou = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.inference_mode()
    with ctx:
        for images, gt_boxes in loader:
            images = images.to(device)
            gt_boxes = gt_boxes.to(device)

            pred_boxes = model(images)
            loss, bbox_s = compute_stage2_loss(pred_boxes, gt_boxes)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            miou = mean_iou(pred_boxes, gt_boxes)
            bs = images.size(0)
            total_loss += float(loss.detach()) * bs
            total_bbox += bbox_s * bs
            total_iou += float(miou.detach()) * bs
            n += bs

    n = max(1, n)
    return total_loss / n, total_bbox / n, total_iou / n


def _build_phase_b_optimizer(model: HelmetCropRegressor, args):
    return torch.optim.AdamW(
        [
            {"params": model.features[-1].parameters(), "lr": args.lr_last_block},
            {"params": list(model.proj.parameters()) + list(model.attn.parameters()) + list(model.fc_box.parameters()),
             "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(args)
    print(f"Stage-2 train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)}")
    if args.train_image_subset_dir is not None:
        print(f"Training subset from: {args.train_image_subset_dir}")

    model = HelmetCropRegressor(pretrained=True, dropout=args.dropout).to(device)
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.plateau_patience,
        factor=args.plateau_factor, min_lr=1e-6,
    )

    writer = SummaryWriter(log_dir=str(args.log_dir))
    best_val_miou = -1.0
    no_improve = 0
    start_epoch = 1

    if args.resume:
        best_path = args.out_dir / "best.pt"
        last_path = args.out_dir / "last.pt"
        for p in (last_path, best_path):
            if p.exists():
                ckpt = torch.load(p, map_location=device)
                sd = ckpt.get("model_state", ckpt)
                model.load_state_dict(sd, strict=True)
                start_epoch = ckpt.get("epoch", 1) + 1
                best_val_miou = ckpt.get("best_val_miou", -1.0)
                no_improve = ckpt.get("no_improve", 0)

                # If resuming after Phase B started, rebuild optimizer with unfrozen block.
                if start_epoch > args.head_only_epochs + 1:
                    model.unfreeze_last_block()
                    optimizer = _build_phase_b_optimizer(model, args)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", patience=args.plateau_patience,
                        factor=args.plateau_factor, min_lr=1e-6,
                    )

                if "optimizer_state" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                if "scheduler_state" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                print(f"Resumed from {p.name}, epoch {start_epoch-1}, best mIoU {best_val_miou:.4f}")
                break

    for epoch in range(start_epoch, args.epochs_max + 1):
        if epoch == args.head_only_epochs + 1:
            model.unfreeze_last_block()
            optimizer = _build_phase_b_optimizer(model, args)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=args.plateau_patience,
                factor=args.plateau_factor, min_lr=1e-6,
            )
            print(
                f"Epoch {epoch}: Phase B — unfroze last backbone block "
                f"(LR last_block={args.lr_last_block:.1e}, head={args.lr_head:.1e})"
            )

        tl, tb, ti = _run_epoch(model, train_loader, device, optimizer, is_train=True)
        vl, vb, vi = _run_epoch(model, val_loader, device, optimizer, is_train=False)
        scheduler.step(vi)

        writer.add_scalar("train/loss", tl, epoch)
        writer.add_scalar("train/bbox_loss", tb, epoch)
        writer.add_scalar("train/mIoU", ti, epoch)
        writer.add_scalar("val/loss", vl, epoch)
        writer.add_scalar("val/bbox_loss", vb, epoch)
        writer.add_scalar("val/mIoU", vi, epoch)

        is_best = vi > best_val_miou
        print(
            f"Epoch {epoch:03d} | "
            f"Train loss={tl:.4f} bbox={tb:.4f} mIoU={ti:.4f} | "
            f"Val   loss={vl:.4f} bbox={vb:.4f} mIoU={vi:.4f}"
            + (" ← best" if is_best else "")
        )

        if is_best:
            best_val_miou = vi
            no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_miou": best_val_miou,
                "no_improve": no_improve,
            }
            torch.save(ckpt, args.out_dir / "last.pt")
            torch.save({**ckpt, "best_val_miou": best_val_miou}, args.out_dir / "best.pt")
            print(f"  → Saved best.pt  (val mIoU={best_val_miou:.4f})")
        else:
            no_improve += 1
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_miou": best_val_miou,
                "no_improve": no_improve,
            }
            torch.save(ckpt, args.out_dir / "last.pt")

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping: val mIoU no improvement for {args.early_stop_patience} epochs.")
            break

    writer.close()
    print(f"\nDone. Best val mIoU: {best_val_miou:.4f}")
    print("Use stage-2 best.pt for helmet crop inference.")


if __name__ == "__main__":
    main()
