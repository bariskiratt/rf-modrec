"""Three model families to compare. Input: (B, 2, L) I/Q, output: (B, num_classes) logits.

- cnn:         VGG-style 1D CNN, after O'Shea, Roy & Clancy (2018), Table III.
- resnet:      residual stacks from the same paper, Table IV.
- transformer: patch embedding + Transformer encoder.

All three are left for me to implement. `tests/test_smoke.py::test_model_shapes`
passes once each returns (B, num_classes) logits for a (B, 2, 1024) input.
"""

from __future__ import annotations

import torch
from torch import nn


class CNN(nn.Module):
    """Stack of Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d blocks, then a small MLP head.

    The paper uses 7 conv blocks (64 filters, kernel 3) and halves the length after each,
    so 1024 samples become 8 before the head.
    """

    def __init__(self, num_classes: int, width: int = 64, blocks: int = 7):
        super().__init__()
        raise NotImplementedError("week 3: CNN baseline")

    def forward(self, x):
        raise NotImplementedError


class ResNet(nn.Module):
    """Residual stacks: 1x1 conv to `width` channels, two residual units, MaxPool1d(2).

    A residual unit is conv -> BN -> ReLU -> conv -> BN, added back to its input
    before the final ReLU. The paper uses 6 stacks of 32 filters.
    """

    def __init__(self, num_classes: int, width: int = 32, stacks: int = 6):
        super().__init__()
        raise NotImplementedError("week 5: ResNet")

    def forward(self, x):
        raise NotImplementedError


class Transformer(nn.Module):
    """Cut the (2, L) frame into L // patch patches, embed each to `dim`, prepend a
    learned [CLS] token, add learned position embeddings, run nn.TransformerEncoder,
    classify from the [CLS] output.

    Hint: a Conv1d with kernel_size = stride = patch is a patch embedding.
    """

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
        raise NotImplementedError("week 6: Transformer")

    def forward(self, x):
        raise NotImplementedError


MODELS = {"cnn": CNN, "resnet": ResNet, "transformer": Transformer}


def build(name: str, num_classes: int, length: int = 1024) -> nn.Module:
    if name == "transformer":
        return Transformer(num_classes, length=length)
    return MODELS[name](num_classes)
