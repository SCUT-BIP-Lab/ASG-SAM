# Adapted from EfficientSAM, Copyright (c) Meta Platforms, Inc. and affiliates.
# ASG-SAM modifications: intermediate-feature output and encoder spectral gating.
# SPDX-License-Identifier: Apache-2.0
"""ViT-Tiny image encoder used by the ASG-SAM paper configuration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: type[nn.Module]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: list[int]) -> torch.Tensor:
    height, width = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    size = int(math.sqrt(abs_pos.shape[1]))
    if size * size != abs_pos.shape[1]:
        raise ValueError("Absolute position embedding is not a square grid.")
    if size != height or size != width:
        abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
    else:
        abs_pos = abs_pos.reshape(1, height, width, -1)
    return abs_pos


class ImageEncoderViT(nn.Module):
    """EfficientSAM-Ti encoder returning both deep and first-block features."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        neck_dim: int = 256,
        detail_layer: int = 0,
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.image_embedding_size = self.img_size // patch_size
        self.detail_layer = int(detail_layer)
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)

        pretrained_grid = 224 // patch_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, pretrained_grid * pretrained_grid + 1, embed_dim)
        )
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, neck_dim, kernel_size=1, bias=False),
            LayerNorm2d(neck_dim),
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(neck_dim),
        )

        # Populated by the builder to retain published-checkpoint key names.
        self.radial_gate: nn.Module | None = None
        self.rgate_strength = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected {self.img_size}x{self.img_size} input, got {tuple(x.shape[-2:])}."
            )
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        height, width = x.shape[1:3]
        x = x + get_abs_pos(self.pos_embed, True, [height, width])
        tokens = x.reshape(x.shape[0], height * width, x.shape[-1])

        detail = None
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index == self.detail_layer:
                detail = tokens.reshape(x.shape[0], height, width, x.shape[-1])

        grid = tokens.reshape(x.shape[0], height, width, x.shape[-1])
        if detail is None:
            detail = grid
        embedding = self.neck(grid.permute(0, 3, 1, 2))
        if self.radial_gate is not None:
            embedding = embedding + self.rgate_strength * self.radial_gate(embedding)
        return embedding, detail
