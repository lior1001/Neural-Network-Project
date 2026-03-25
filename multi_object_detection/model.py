"""
Part 3 model: MobileNetV3-Large backbone + FPN + per-slot spatial attention.

WHY FPN HELPS:
  The previous model used only the last backbone layer (10×10 at 320 input).
  Each grid cell covers 32×32 pixels — larger than a helmet.
  A helmet in the dataset is typically 50-80px tall, so it only spans 1-2 cells.
  The attention mechanism has too little resolution to localize it precisely.

  FPN (Feature Pyramid Network) fuses:
    - Layer features[-1]: 10×10, 960ch  — high semantics, low resolution
    - Layer features[-4]: 20×20, 112ch  — medium semantics, medium resolution
  After fusion: 20×20 feature map with 256ch.
  Each cell now covers 16×16 pixels — much better for helmets.

PER-SLOT SPATIAL ATTENTION (kept from previous version):
  Each slot gets its own attention map over the 20×20 = 400 spatial locations.
  The attention-weighted feature descriptor encodes WHERE the slot is looking,
  enabling the box prediction to know the object's location in the frame.

FIXED-SLOT SEMANTICS (unchanged):
  Slot 0 = Bicycle, Slot 1 = Helmet. Always. No matching needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

BACKGROUND_CLASS_ID = 0

# MobileNetV3-Large intermediate layer indices for FPN
# features[-1]  : 10×10, 960 channels  (last block)
# features[-4]  : 20×20, 112 channels  (earlier block)
_IDX_HIGH = -1   # 10×10
_IDX_LOW  = -4   # 20×20


class MobileNetV3MultiBBox(nn.Module):
    """
    MobileNetV3-Large + FPN + per-slot spatial attention detection head.

    Returns
    -------
    pred_boxes  : (B, num_slots, 4)  normalised (cx, cy, w, h) in [0,1]
    pred_logits : (B, num_slots, num_classes+1)  raw class logits
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_slots: int | None = None,
        pretrained: bool = True,
        dropout: float = 0.2,
        fpn_channels: int = 256,
        hidden_ch: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes      = num_classes
        self.num_slots        = num_slots if num_slots is not None else num_classes
        self.num_class_logits = num_classes + 1

        # ── Backbone ──────────────────────────────────────────────────────────
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        bb = models.mobilenet_v3_large(weights=weights)
        self.features = bb.features
        # We will tap into features[-4] and features[-1] for FPN

        # Get actual channel counts by running a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 320, 320)
            feats = list(self.features)
            x = dummy
            for i, layer in enumerate(feats):
                x = layer(x)
                if i == len(feats) + _IDX_LOW:
                    ch_low = x.shape[1]   # channels at 20×20
                if i == len(feats) + _IDX_HIGH:
                    ch_high = x.shape[1]  # channels at 10×10

        # ── FPN lateral convs ─────────────────────────────────────────────────
        # Bring both levels to fpn_channels
        self.lat_high = nn.Sequential(
            nn.Conv2d(ch_high, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.lat_low = nn.Sequential(
            nn.Conv2d(ch_low, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        # Smooth conv after fusion at 20×20
        self.smooth = nn.Sequential(
            nn.Conv2d(fpn_channels, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        # Output: (B, hidden_ch, 20, 20)

        # ── Per-slot attention maps ───────────────────────────────────────────
        # Each slot has its own 1×1 conv producing a 20×20 attention map.
        self.slot_attention = nn.ModuleList([
            nn.Conv2d(hidden_ch, 1, kernel_size=1)
            for _ in range(self.num_slots)
        ])

        # ── Per-slot prediction heads ─────────────────────────────────────────
        self.slot_fc_boxes  = nn.ModuleList([
            nn.Linear(hidden_ch, 4) for _ in range(self.num_slots)
        ])
        self.slot_fc_logits = nn.ModuleList([
            nn.Linear(hidden_ch, self.num_class_logits)
            for _ in range(self.num_slots)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for mod in [self.lat_high, self.lat_low, self.smooth]:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        for i in range(self.num_slots):
            nn.init.normal_(self.slot_attention[i].weight, std=0.01)
            nn.init.zeros_(self.slot_attention[i].bias)
            nn.init.normal_(self.slot_fc_boxes[i].weight,  std=0.01)
            nn.init.zeros_(self.slot_fc_boxes[i].bias)
            nn.init.normal_(self.slot_fc_logits[i].weight, std=0.01)
            nn.init.zeros_(self.slot_fc_logits[i].bias)

    # ── Backbone feature extraction at two scales ─────────────────────────────

    def _extract_features(self, x: torch.Tensor):
        """Return (feat_low, feat_high) at 20×20 and 10×10 respectively."""
        feats = list(self.features)
        n = len(feats)
        low_idx  = n + _IDX_LOW
        high_idx = n + _IDX_HIGH
        feat_low = feat_high = None
        for i, layer in enumerate(feats):
            x = layer(x)
            if i == low_idx:
                feat_low = x
            if i == high_idx:
                feat_high = x
        return feat_low, feat_high

    # ── Freezing helpers ──────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False
        for mod in [self.lat_high, self.lat_low, self.smooth,
                    self.slot_attention, self.slot_fc_boxes, self.slot_fc_logits]:
            for p in mod.parameters():
                p.requires_grad = True

    def unfreeze_last_block(self) -> None:
        self.freeze_backbone()
        for p in self.features[_IDX_HIGH].parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        pred_boxes  : (B, num_slots, 4)   Sigmoid → [0,1]  (cx, cy, w, h)
        pred_logits : (B, num_slots, C+1) raw logits
        """
        B = x.size(0)

        # Extract features at two scales
        feat_low, feat_high = self._extract_features(x)
        # feat_low:  (B, ch_low,  20, 20)
        # feat_high: (B, ch_high, 10, 10)

        # FPN: upsample high-level features and add to low-level
        p_high = self.lat_high(feat_high)                          # (B, 256, 10, 10)
        p_low  = self.lat_low(feat_low)                            # (B, 256, 20, 20)
        p_high_up = F.interpolate(p_high, size=p_low.shape[-2:],
                                  mode="nearest")                  # (B, 256, 20, 20)
        fused = self.smooth(p_low + p_high_up)                     # (B, hidden, 20, 20)

        _, C, Hf, Wf = fused.shape
        N = Hf * Wf  # 400 spatial locations (vs 100 before)

        all_boxes, all_logits = [], []

        for s in range(self.num_slots):
            # Per-slot attention: (B, 1, Hf, Wf) → softmax → (B, N)
            attn = F.softmax(
                self.slot_attention[s](fused).view(B, N), dim=-1
            )  # (B, N)

            # Attention-weighted descriptor: (B, C)
            shared_flat = fused.view(B, C, N)
            descriptor  = torch.einsum("bn,bcn->bc", attn, shared_flat)

            all_boxes.append(torch.sigmoid(self.slot_fc_boxes[s](descriptor)))
            all_logits.append(self.slot_fc_logits[s](descriptor))

        pred_boxes  = torch.stack(all_boxes,  dim=1)  # (B, num_slots, 4)
        pred_logits = torch.stack(all_logits, dim=1)  # (B, num_slots, C+1)
        return pred_boxes, pred_logits