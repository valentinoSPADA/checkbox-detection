"""The Stage 2 classifier architecture.

Deliberately tiny (~60k parameters). The reasoning is not modesty for its own sake: the
input is a 40x40 grayscale patch of an essentially geometric object, the training signal is
synthetic, and a large network would overfit the generator's quirks rather than the concept
of "a small square with a mark in it". A small model also keeps CPU inference in the tens of
microseconds per crop, which matters because a single page produces thousands of proposals
and the JD asks for low latency at scale.
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_SIZE = 40
NUM_CLASSES = 3
CLASS_NAMES = ("not_a_checkbox", "unchecked", "checked")


def _block(cin: int, cout: int) -> nn.Sequential:
    """Conv-BN-ReLU pair. BatchNorm is what lets the model tolerate the wide brightness
    range the generator produces (white paper through to heavily shaded rows) without
    needing per-image contrast normalisation at inference."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CheckboxNet(nn.Module):
    """Classifies a proposal crop as not_a_checkbox / unchecked / checked.

    Folding "is this even a checkbox" into the same head as "is it filled" is intentional:
    the two questions share every low-level feature (border straightness, interior ink), and
    a single softmax gives the Go policy layer one comparable confidence to threshold and to
    rank during suppression, instead of two scores that could disagree.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 16),
            nn.MaxPool2d(2),      # 40 -> 20
            _block(16, 32),
            nn.MaxPool2d(2),      # 20 -> 10
            _block(32, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N,1,40,40) in [0,1] to (N,3) logits. Softmax is applied by the caller so the
        exported ONNX graph stays loss-agnostic."""
        return self.head(self.features(x))
