"""Anisotropic Spectral Gating (ASG)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class AnisotropicSpectralGating(nn.Module):
    """Polar-parameterized spectral gate used by ASG-SAM.

    The learnable table is indexed jointly by radial and angular frequency.
    It is bilinearly mapped to the native, unshifted ``rfft2`` grid at run
    time. Frequencies outside the per-axis Nyquist radius use the outermost
    radial bin, matching the implementation described in the paper.

    This module returns ``x + delta``. :class:`ASGDelta` exposes only the
    learned residual so that the enclosing EfficientSAM residual path keeps
    checkpoint-compatible parameter names and behavior.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        radial_bins: int = 64,
        angular_bins: int = 128,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)

        self.polar_weight = nn.Parameter(
            torch.empty(1, self.channels, self.radial_bins, self.angular_bins)
        )
        nn.init.normal_(self.polar_weight, mean=1.0, std=0.02)
        self.register_buffer("grid", self._build_grid(self.height, self.width))

    @staticmethod
    def _build_grid(height: int, width: int) -> torch.Tensor:
        vertical = torch.fft.fftfreq(height)
        horizontal = torch.fft.rfftfreq(width)
        freq_y, freq_x = torch.meshgrid(vertical, horizontal, indexing="ij")

        radius = torch.sqrt(freq_y.square() + freq_x.square())
        radius_normalized = radius / 0.5 * 2.0 - 1.0

        angle = torch.atan2(freq_y, freq_x)
        angle_normalized = angle / (math.pi / 2.0)

        # grid_sample expects (x, y), corresponding to (angle, radius).
        return torch.stack((angle_normalized, radius_normalized), dim=-1).unsqueeze(0)

    def get_cartesian_gate(
        self,
        batch_size: int = 1,
        spatial_size: tuple[int, int] | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Map the learned polar table to an rFFT grid for inspection/use."""

        height, width = spatial_size or (self.height, self.width)
        device = device or self.polar_weight.device
        weight = self.polar_weight.to(device=device, dtype=dtype or self.polar_weight.dtype)
        grid = (
            self.grid
            if (height, width) == (self.height, self.width)
            else self._build_grid(height, width).to(device)
        )
        grid = grid.to(device=device, dtype=weight.dtype).expand(batch_size, -1, -1, -1)
        weight = weight.expand(batch_size, -1, -1, -1)
        mapped = F.grid_sample(
            weight,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return torch.sigmoid(mapped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")

        original_dtype = x.dtype
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spectrum = torch.fft.rfft2(work, norm="ortho")
        gate = self.get_cartesian_gate(
            batch,
            (height, width),
            device=x.device,
            dtype=work.dtype,
        )
        residual = torch.fft.irfft2(spectrum * gate, s=(height, width), norm="ortho")
        return x + residual.to(original_dtype)


class ASGDelta(nn.Module):
    """Checkpoint-compatible wrapper that returns the ASG residual only."""

    def __init__(self, asg: AnisotropicSpectralGating) -> None:
        super().__init__()
        self.asg = asg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.asg(x) - x
