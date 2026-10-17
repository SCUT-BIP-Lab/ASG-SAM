# Adapted from Segment Anything and EfficientSAM.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# ASG-SAM modifications: standalone point-prompt interface.
# SPDX-License-Identifier: Apache-2.0
"""SAM point-prompt encoder used by ASG-SAM."""

from __future__ import annotations

import math

import torch
from torch import nn

from .encoder import LayerNorm2d


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = (2.0 * coords - 1.0) @ self.positional_encoding_gaussian_matrix
        coords = 2.0 * math.pi * coords
        return torch.cat((torch.sin(coords), torch.cos(coords)), dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        height, width = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((height, width), device=device, dtype=torch.float32)
        y = (grid.cumsum(dim=0) - 0.5) / height
        x = (grid.cumsum(dim=1) - 0.5) / width
        return self._pe_encoding(torch.stack((x, y), dim=-1)).permute(2, 0, 1)

    def forward_with_coords(
        self, coords: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        coords = coords.clone()
        coords[:, :, 0] /= image_size[1]
        coords[:, :, 1] /= image_size[0]
        return self._pe_encoding(coords.float())


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_channels: int = 16,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for _ in range(4)])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels // 4),
            activation(),
            nn.Conv2d(mask_in_channels // 4, mask_in_channels, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels),
            activation(),
            nn.Conv2d(mask_in_channels, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        points = points + 0.5
        if pad:
            points = torch.cat(
                (points, torch.zeros((points.shape[0], 1, 2), device=points.device)),
                dim=1,
            )
            labels = torch.cat(
                (labels, -torch.ones((labels.shape[0], 1), device=labels.device)),
                dim=1,
            )
        embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        embedding[labels == -1] = 0.0
        embedding[labels == -1] += self.not_a_point_embed.weight
        embedding[labels == 0] += self.point_embeddings[0].weight
        embedding[labels == 1] += self.point_embeddings[1].weight
        return embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        coords = (boxes + 0.5).reshape(-1, 2, 2)
        embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        embedding[:, 0, :] += self.point_embeddings[2].weight
        embedding[:, 1, :] += self.point_embeddings[3].weight
        return embedding

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points is not None:
            batch_size = points[0].shape[0]
        elif boxes is not None:
            batch_size = boxes.shape[0]
        elif masks is not None:
            batch_size = masks.shape[0]
        else:
            batch_size = 1

        device = self.point_embeddings[0].weight.device
        sparse = torch.empty((batch_size, 0, self.embed_dim), device=device)
        if points is not None:
            sparse = torch.cat((sparse, self._embed_points(*points, pad=boxes is None)), dim=1)
        if boxes is not None:
            sparse = torch.cat((sparse, self._embed_boxes(boxes)), dim=1)

        if masks is None:
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )
        else:
            dense = self.mask_downscaling(masks)
        return sparse, dense
