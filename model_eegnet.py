# model_eegnet.py
# EEGNet-style model with dynamic computation of final linear layer input size.
# This avoids hard-coded flatten sizes and resolves runtime mismatches.

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    Lightweight EEGNet-like architecture.
    Args:
        num_channels: number of EEG channels (height dimension)
        samples: number of time samples (width dimension)
        num_classes: number of output classes
        F1, D, F2, kern_length, dropout_rate: tunable hyperparams (kept defaults)
    """

    def __init__(
        self,
        num_channels: int,
        samples: int,
        num_classes: int = 2,
        F1: int = 16,
        D: int = 2,
        F2: int = None,
        kern_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        if F2 is None:
            F2 = F1 * D

        self.num_channels = int(num_channels)
        self.samples = int(samples)
        self.num_classes = int(num_classes)
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kern_length = kern_length
        self.dropout_rate = dropout_rate

        # First temporal conv
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Depthwise conv (spatial) - groups = F1 to do depthwise across channels per filter
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate),
        )

        # Separable conv (pointwise conv)
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 16 // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout_rate),
        )

        # We'll create the final classifier with a Linear whose in_features is computed
        # dynamically by forwarding a dummy tensor through the conv blocks.
        flat_feat = self._compute_flattened_size()
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_feat, self.num_classes),
        )

    def _compute_flattened_size(self) -> int:
        """
        Create a dummy tensor (1,1,num_channels,samples), forward through conv blocks,
        and return the flattened feature dimension. Runs on CPU so it is safe in __init__.
        """
        device = torch.device("cpu")
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.num_channels, self.samples, device=device)
            x = self.firstconv(dummy)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            x = x.view(1, -1)
            flat = x.shape[1]
        if flat <= 0:
            raise ValueError(f"Computed flattened feature size is {flat} (bad). Check num_channels/samples.")
        return int(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x expected shape: (batch, 1, channels, samples)
        returns logits shape: (batch, num_classes)
        """
        # Defensive checks
        if x.ndim != 4:
            raise ValueError(f"Expected input with 4 dims (B,1,C,T), got shape {tuple(x.shape)}")

        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)
        return x
