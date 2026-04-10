from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

# ResNet-18 adapted for CIFAR-100
class CIFARResNet18(nn.Module):
    def __init__(self, num_classes: int = 100, dropout_p: float = 0.0):
        super().__init__()
        self.model = resnet18(weights=None, num_classes=num_classes)

        # Adjust for CIFAR-100
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        # Replace final layer
        in_features = self.model.fc.in_features
        if dropout_p > 0:
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_model(num_classes: int = 100, dropout_p: float = 0.0):
    return CIFARResNet18(num_classes=num_classes, dropout_p=dropout_p)