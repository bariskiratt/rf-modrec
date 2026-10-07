"""Three model families to compare. Input: (B, 2, L) I/Q, output: (B, num_classes) logits.

- cnn:         VGG-style 1D CNN, after O'Shea, Roy & Clancy (2018).
- resnet:      residual stacks from the same paper.
- transformer: patch embedding + Transformer encoder.
"""

from __future__ import annotations

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_classes: int, width: int = 64, blocks: int = 7):
        super().__init__()
        layers, c_in = [], 2
        for _ in range(blocks):
            layers += [
                nn.Conv1d(c_in, width, 3, padding=1),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            c_in = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.SELU(),
            nn.AlphaDropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class _ResUnit(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c),
        )

    def forward(self, x):
        return torch.relu(x + self.body(x))


class _ResStack(nn.Module):
    def __init__(self, c_in: int, c: int):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c, 1)
        self.units = nn.Sequential(_ResUnit(c), _ResUnit(c))
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.units(self.proj(x)))


class ResNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 32, stacks: int = 6):
        super().__init__()
        self.stacks = nn.Sequential(
            *[_ResStack(2 if i == 0 else width, width) for i in range(stacks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.SELU(),
            nn.AlphaDropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.stacks(x))


class Transformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        length: int = 1024,
        patch: int = 16,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.embed = nn.Conv1d(2, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, length // patch + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        t = self.embed(x).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1) + self.pos
        t = self.encoder(t)
        return self.head(self.norm(t[:, 0]))


MODELS = {"cnn": CNN, "resnet": ResNet, "transformer": Transformer}


def build(name: str, num_classes: int, length: int = 1024) -> nn.Module:
    if name == "transformer":
        return Transformer(num_classes, length=length)
    return MODELS[name](num_classes)
