import torch
from torch import nn
from torchvision import models


class MobileNetV3BBox(nn.Module):
    """MobileNet V3 backbone with a bbox regression head."""
    def __init__(self, pretrained: bool = True, dropout: float = 0.2) -> None:
        """Initialize backbone and a 4D normalized bbox head."""
        super().__init__()
        # Lecture 2: transfer learning with pretrained ImageNet weights.
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        # Use only convolutional features from the classifier backbone.
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Small regression head outputs 4 normalized bbox values.
        in_features = backbone.classifier[0].in_features
        self.bbox_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run images through backbone and output normalized cx,cy,w,h."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.bbox_head(x)
