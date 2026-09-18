"""
models.py
=========
Architectures for the victim f_V and the substitute f_S.

Two families are provided so the project can demonstrate the central Knockoff
Nets claim: the substitute need not match the victim's architecture.

  * resnet18_cifar / resnet34_cifar : a CIFAR-adapted ResNet (3x3 stem, no
    initial max-pool) -- the natural "strong" victim.
  * SmallCNN : a compact 4-layer conv net -- a natural "weak / different"
    substitute, matching the "four-layer CNN" mentioned in the design report.

All models output raw logits over NUM_CLASSES; softmax is applied where needed
(the server, the soft-label loss). Never bake softmax into the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES


# --------------------------------------------------------------------------- #
# CIFAR ResNet                                                                 #
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    """ResNet with a CIFAR stem (3x3 conv, stride 1, no max-pool)."""

    def __init__(self, block, num_blocks, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def resnet18_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes)


# --------------------------------------------------------------------------- #
# SmallCNN (deliberately different / weaker family)                           #
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """A compact 4-conv-layer network for 32x32 inputs."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # 32 -> 16
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
_BUILDERS = {
    "resnet18": resnet18_cifar,
    "resnet34": resnet34_cifar,
    "smallcnn": SmallCNN,
}

ARCH_CHOICES = tuple(_BUILDERS.keys())


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    name = name.lower()
    if name not in _BUILDERS:
        raise ValueError(f"unknown arch '{name}', choose from {ARCH_CHOICES}")
    return _BUILDERS[name](num_classes=num_classes)
