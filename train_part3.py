"""
Train Part 3: fixed-slot multi-object detector.

Training strategy directly mirrors the reference implementation:

Phase A — head only (head_only_epochs):
  Backbone is fully frozen. Only det_head, fc_boxes, fc_logits train.
  LR = lr_head, weight_decay = weight_decay.
  Purpose: let the randomly-initialised head stabilise before touching
  the pretrained backbone.

Phase B — unfreeze last backbone block:
  Unfreeze only features[-1] (the last InvertedResidual block of
  MobileNetV3-Large, ~analogous to layer4 of ResNet18).
  Two param groups: last block at lr_layer4 (lower), head at lr_head.
  Purpose: fine-tune the most task-relevant backbone features slightly
  while keeping the rest of the pretrained weights intact.

Scheduler: ReduceLROnPlateau on val mIoU (mode=max).
Saves: best.pt (highest val mIoU) and last.pt.
Early stopping on val mIoU if use_early_stopping=True.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset_coco import COCOMultiObjectDataset, default_coco_paths
from model import MobileNetV3MultiBBox
from losses import compute_loss, mean_iou


def parse_args():
    p = argparse.ArgumentParser(description="Train Part 3 fixed-slot multi-object detector.")
    p.add_argument("--data_dir",          type=Path,  required=True)
    p.add_argument("--category_name",     type=str,   default="Bicycle,Helmet,No_helmet",
        help="Comma-separated class names IN SLOT ORDER. Slot 0=first, slot 1=second, ...")
    p.add_argument("--input_size",        type=int,   default=320)
    p.add_argument("--batch_size",        type=int,   default=16)
    p.add_argument("--epochs_max",        type=int,   default=60)
    p.add_argument("--head_only_epochs",  type=int,   default=10,
        help="Phase A: train only the detection head for this many epochs.")
    p.add_argument("--lr_head",           type=float, default=1e-3,
        help="LR for detection head (both phases).")
    p.add_argument("--lr_last_block",     type=float, default=1e-4,
        help="LR for backbone last block in Phase B (should be ~lr_head/10).")
    p.add_argument("--weight_decay",      type=float, default=3e-4)
    p.add_argument("--dropout",           type=float, default=0.2)
    p.add_argument("--blur_p",            type=float, default=0.3,
        help="Probability of Gaussian blur augmentation during training.")
    p.add_argument("--bbox_loss_weight",  type=float, default=1.0)
    p.add_argument("--giou_loss_weight",  type=float, default=1.0)
    p.add_argument("--slot_loss_weights", type=str, default=None,
        help="Comma-separated localization weights in slot order, e.g. '1.0,2.0'.")
    p.add_argument("--plateau_patience",  type=int,   default=5)
    p.add_argument("--plateau_factor",    type=float, default=0.5)
    p.add_argument("--early_stop_patience", type=int, default=12,
        help="Stop if val mIoU does not improve for this many epochs. 0=disabled.")
    p.add_argument("--num_workers",       type=int,   default=2)
    p.add_argument("--log_dir",  type=Path, default=Path("outputs/part3_logs"))
    p.add_argument("--out_dir",  type=Path, default=Path("outputs/part3"))
    p.add_argument("--max_train_samples", type=int,   default=None)
    p.add_argument("--max_val_samples",   type=int,   default=None)
    p.add_argument("--require_all_classes", action="store_true",
        help="Keep only images that contain all requested categories.")
    p.add_argument("--max_instances_per_class", type=int, default=None,
        help="Keep only images where each requested class appears at most this many times.")
    p.add_argument("--train_image_subset_dir", type=Path, default=None,
        help="Optional directory of images whose filenames define a training subset.")
    p.add_argument("--init_checkpoint",   type=Path, default=None,
        help="Initialise model from a checkpoint but start a fresh training run.")
    p.add_argument("--no_augment",        action="store_true")
    p.add_argument("--resume",            action="store_true")
    return p.parse_args()


def _parse_slot_loss_weights(raw: str | None, num_slots: int, device: torch.device):
    if raw is None:
        return None
    weights = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(weights) != num_slots:
        raise ValueError(f"--slot_loss_weights must have {num_slots} values, got {len(weights)}.")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loaders(args):
    cats = [s.strip() for s in args.category_name.split(",") if s.strip()]
    if len(cats) < 2:
        raise ValueError("Need at least 2 categories.")

    ann, img = default_coco_paths(args.data_dir, "train")
    train_ds = COCOMultiObjectDataset(
        annotations_path=ann, images_dir=img,
        category_names=cats, input_size=args.input_size,
        train=not args.no_augment,
        require_all_classes=args.require_all_classes,
        max_instances_per_class=args.max_instances_per_class,
        blur_p=args.blur_p,
        include_filenames_from_dir=args.train_image_subset_dir,
    )
    if args.max_train_samples:
        train_ds = Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))

    ann_v, img_v = default_coco_paths(args.data_dir, "valid")
    val_ds = COCOMultiObjectDataset(
        annotations_path=ann_v, images_dir=img_v,
        category_names=cats, input_size=args.input_size,
        train=False,
        require_all_classes=args.require_all_classes,
        max_instances_per_class=args.max_instances_per_class,
        blur_p=0.0,
    )
    if args.max_val_samples:
        val_ds = Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    return train_loader, val_loader, len(cats)


def _run_epoch(model, loader, device, optimizer, args, slot_loss_weights, *, is_train: bool):
    model.train(is_train)
    total_loss = total_bbox = total_class = total_iou = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.inference_mode()
    with ctx:
        for images, gt_boxes, gt_class_ids in loader:
            images       = images.to(device)
            gt_boxes     = gt_boxes.to(device)
            gt_class_ids = gt_class_ids.to(device)

            pred_boxes, pred_logits = model(images)
            loss, bbox_s, cls_s = compute_loss(
                pred_boxes, pred_logits, gt_boxes, gt_class_ids,
                bbox_loss_weight=args.bbox_loss_weight,
                giou_loss_weight=args.giou_loss_weight,
                slot_loss_weights=slot_loss_weights,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            miou = mean_iou(pred_boxes, gt_boxes, gt_class_ids)

            bs          = images.size(0)
            total_loss  += float(loss.detach()) * bs
            total_bbox  += bbox_s              * bs
            total_class += cls_s               * bs
            total_iou   += float(miou.detach()) * bs
            n           += bs

    n = max(1, n)
    return total_loss/n, total_bbox/n, total_class/n, total_iou/n


def _build_phase_b_optimizer(model: MobileNetV3MultiBBox, args):
    """Phase B: two groups — last backbone block + FPN detection head."""
    head_params = (
        list(model.lat_high.parameters())
        + list(model.lat_low.parameters())
        + list(model.smooth.parameters())
        + list(model.slot_attention.parameters())
        + list(model.slot_fc_boxes.parameters())
        + list(model.slot_fc_logits.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": model.features[-1].parameters(), "lr": args.lr_last_block},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, num_classes = make_loaders(args)
    print(f"Classes: {num_classes} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    slot_loss_weights = _parse_slot_loss_weights(args.slot_loss_weights, num_classes, device)
    if slot_loss_weights is not None:
        print(f"Slot localization weights: {slot_loss_weights.tolist()}")
    if args.train_image_subset_dir is not None:
        print(f"Training subset from: {args.train_image_subset_dir}")
    print(f"Training blur probability: {args.blur_p:.2f}")

    model = MobileNetV3MultiBBox(
        num_classes=num_classes,
        num_slots=num_classes,   # one slot per class
        pretrained=True,
        dropout=args.dropout,
    ).to(device)

    # ── Phase A: freeze backbone, train head only ─────────────────────────────
    model.freeze_backbone()
    print(f"Phase A: head-only training for {args.head_only_epochs} epochs.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.plateau_patience,
        factor=args.plateau_factor, min_lr=1e-6,
    )

    writer          = SummaryWriter(log_dir=str(args.log_dir))
    best_val_miou   = -1.0
    no_improve      = 0
    start_epoch     = 1

    if args.resume and args.init_checkpoint is not None:
        raise ValueError("Use either --resume or --init_checkpoint, not both.")

    if args.resume:
        best_path = args.out_dir / "best.pt"
        last_path = args.out_dir / "last.pt"
        for p in (last_path, best_path):
            if p.exists():
                ckpt = torch.load(p, map_location=device)
                sd   = ckpt.get("model_state", ckpt)
                model.load_state_dict(sd, strict=True)
                start_epoch   = ckpt.get("epoch", 1) + 1
                best_val_miou = ckpt.get("best_val_miou", -1.0)
                print(f"Resumed from {p.name}, epoch {start_epoch-1}, best mIoU {best_val_miou:.4f}")
                break
    elif args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        sd   = ckpt.get("model_state", ckpt)
        model.load_state_dict(sd, strict=True)
        print(f"Initialised model from {args.init_checkpoint}")

    for epoch in range(start_epoch, args.epochs_max + 1):

        # ── Switch to Phase B after head_only_epochs ──────────────────────────
        if epoch == args.head_only_epochs + 1:
            model.unfreeze_last_block()
            optimizer = _build_phase_b_optimizer(model, args)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=args.plateau_patience,
                factor=args.plateau_factor, min_lr=1e-6,
            )
            print(f"Epoch {epoch}: Phase B — unfroze last backbone block "
                  f"(LR last_block={args.lr_last_block:.1e}, head={args.lr_head:.1e})")

        # ── Train + validate ──────────────────────────────────────────────────
        tl, tb, tc, ti = _run_epoch(model, train_loader, device, optimizer, args, slot_loss_weights, is_train=True)
        vl, vb, vc, vi = _run_epoch(model, val_loader,   device, optimizer, args, slot_loss_weights, is_train=False)

        scheduler.step(vi)

        lrs = [g["lr"] for g in optimizer.param_groups]
        writer.add_scalar("train/loss",       tl, epoch)
        writer.add_scalar("train/bbox_loss",  tb, epoch)
        writer.add_scalar("train/class_loss", tc, epoch)
        writer.add_scalar("train/mIoU",       ti, epoch)
        writer.add_scalar("val/loss",         vl, epoch)
        writer.add_scalar("val/bbox_loss",    vb, epoch)
        writer.add_scalar("val/class_loss",   vc, epoch)
        writer.add_scalar("val/mIoU",         vi, epoch)
        writer.add_scalar("lr/group0",        lrs[0], epoch)
        if len(lrs) > 1:
            writer.add_scalar("lr/group1", lrs[1], epoch)

        is_best = vi > best_val_miou
        print(
            f"Epoch {epoch:03d} | "
            f"Train loss={tl:.4f} bbox={tb:.4f} cls={tc:.4f} mIoU={ti:.4f} | "
            f"Val   loss={vl:.4f} bbox={vb:.4f} cls={vc:.4f} mIoU={vi:.4f}"
            + (" ← best" if is_best else "")
        )

        # Save checkpoints
        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "best_val_miou": best_val_miou}
        torch.save(ckpt, args.out_dir / "last.pt")
        if is_best:
            best_val_miou = vi
            no_improve    = 0
            torch.save({**ckpt, "best_val_miou": best_val_miou},
                       args.out_dir / "best.pt")
            print(f"  → Saved best.pt  (val mIoU={best_val_miou:.4f})")
        else:
            no_improve += 1

        # Early stopping
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping: val mIoU no improvement for {args.early_stop_patience} epochs.")
            break

    writer.close()
    print(f"\nDone. Best val mIoU: {best_val_miou:.4f}")
    print("Use  best.pt  for inference.")


if __name__ == "__main__":
    main()