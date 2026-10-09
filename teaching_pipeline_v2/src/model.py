"""CNN of Fig. 3: input 1 x C 'image' -> conv -> batch-norm -> ReLU -> FC (N nodes) -> softmax.

The network classifies which of the N calibration (training) positions the event came
from. The predicted position is the probability-weighted centre of mass of the
calibration positions (Eq. 1 of the paper).
"""
import torch
import torch.nn as nn


class PositionCNN(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=200, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(1, n_filters, kernel_size=(1, kernel_size), padding=(0, pad))
        self.bn = nn.BatchNorm2d(n_filters)
        self.fc = nn.Linear(n_filters * n_channels, n_classes)

    def forward(self, x):                     # x: (B, 1, 1, C)
        h = torch.relu(self.bn(self.conv(x)))
        return self.fc(h.flatten(1))          # logits; softmax is inside the loss / predict


def centre_of_mass(logits, calib_xy):
    """Eq. 1: sum_i p_i * x_i, with p = softmax(logits). calib_xy: (N, 2) tensor."""
    p = torch.softmax(logits, dim=1)
    return p @ calib_xy
