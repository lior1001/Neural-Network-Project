"""
Stage 2 model: helmet box regressor on cropped rider regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HelmetCropRegressor(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.1,
        hidden_ch: int = 128,
    ) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        bb = models.mobilenet_v3_small(weights=weights)
        self.features = bb.features

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.features(dummy)
            feat_ch = feat.shape[1]

        self.proj = nn.Sequential(
            nn.Conv2d(feat_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )
        self.attn = nn.Conv2d(hidden_ch, 1, kernel_size=1)
        self.fc_box = nn.Linear(hidden_ch, 4)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.attn.weight, std=0.01)
        nn.init.zeros_(self.attn.bias)
        nn.init.normal_(self.fc_box.weight, std=0.01)
        nn.init.zeros_(self.fc_box.bias)

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = True
        for p in self.attn.parameters():
            p.requires_grad = True
        for p in self.fc_box.parameters():
            p.requires_grad = True

    def unfreeze_last_block(self) -> None:
        self.freeze_backbone()
        for p in self.features[-1].parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(self.features(x))
        b, c, h, w = feat.shape
        n = h * w
        attn = F.softmax(self.attn(feat).view(b, n), dim=-1)
        desc = torch.einsum("bn,bcn->bc", attn, feat.view(b, c, n))
        return torch.sigmoid(self.fc_box(desc))
