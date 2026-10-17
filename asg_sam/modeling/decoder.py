# Adapted from EfficientSAM and the SAM-HQ detail-token decoder design.
# Copyright (c) Meta Platforms, Inc. and affiliates (EfficientSAM portions).
# ASG-SAM modifications: DoG-AMGD, global detail routing, and decoder spectral gating.
# SPDX-License-Identifier: Apache-2.0
"""Mask decoder with Difference-of-Gaussians Adaptive Multi-Granularity Decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .mlp import MLPBlock


class MaskDecoder(nn.Module):
    """EfficientSAM mask decoder augmented with the paper's DoG-AMGD path."""

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 1,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 192,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4,
                transformer_dim // 8,
                kernel_size=2,
                stride=2,
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(
                    transformer_dim,
                    transformer_dim,
                    transformer_dim // 8,
                    3,
                    act=activation,
                )
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLPBlock(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            act=activation,
        )

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLPBlock(
            transformer_dim,
            transformer_dim,
            transformer_dim // 8,
            3,
            act=activation,
        )
        self.num_mask_tokens += 1

        # Retained for compatibility with the EfficientSAM-HQ initialization.
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.amgd_fine = self._make_granularity_branch(vit_dim, transformer_dim, 1, activation)
        self.amgd_mid = self._make_granularity_branch(vit_dim, transformer_dim, 3, activation)
        self.amgd_coarse = self._make_granularity_branch(vit_dim, transformer_dim, 5, activation)
        self.amgd_router = MLPBlock(
            transformer_dim,
            transformer_dim // 2,
            3,
            2,
            act=activation,
        )

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4,
                transformer_dim // 8,
                kernel_size=2,
                stride=2,
            ),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )

        # Populated by the builder to retain published-checkpoint key names.
        self.radial_gate: nn.Module | None = None
        self.rgate_strength_dec = 1.0

    @staticmethod
    def _make_granularity_branch(
        vit_dim: int,
        transformer_dim: int,
        kernel_size: int,
        activation: type[nn.Module],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
            nn.Conv2d(
                transformer_dim // 8,
                transformer_dim // 8,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )

    def _dog_amgd(
        self, vit_features: torch.Tensor, global_detail_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build and adaptively fuse adjacent-granularity differences."""

        routing = F.softmax(self.amgd_router(global_detail_token), dim=-1)
        fine = self.amgd_fine(vit_features)
        middle = self.amgd_mid(vit_features)
        coarse = self.amgd_coarse(vit_features)

        weight_fine = routing[:, 0, None, None, None]
        weight_dog_1 = routing[:, 1, None, None, None]
        weight_dog_2 = routing[:, 2, None, None, None]
        fused = (
            weight_fine * fine + weight_dog_1 * (fine - middle) + weight_dog_2 * (middle - coarse)
        )
        return fused, routing

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
        return_routing: bool = False,
    ):
        output_tokens = torch.cat(
            (self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight), dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings + dense_prompt_embeddings
        batch, channels, height, width = src.shape
        token_output, src = self.transformer(src, image_pe, tokens)
        iou_token = token_output[:, 0, :]
        mask_tokens = token_output[:, 1 : 1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).reshape(batch, channels, height, width)

        upscaled_sam = self.output_upscaling(src)
        vit_features = interm_embeddings.permute(0, 3, 1, 2)
        global_detail_token = mask_tokens[:, -1, :]
        detail_features, routing = self._dog_amgd(vit_features, global_detail_token)

        hq_features = self.embedding_encoder(image_embeddings) + detail_features
        if self.radial_gate is not None:
            hq_features = hq_features + self.rgate_strength_dec * self.radial_gate(hq_features)
        upscaled_hq = self.embedding_maskfeature(upscaled_sam) + hq_features

        hypernetworks: list[torch.Tensor] = []
        for index in range(self.num_mask_tokens):
            if index < self.num_mask_tokens - 1:
                hypernetworks.append(
                    self.output_hypernetworks_mlps[index](mask_tokens[:, index, :])
                )
            else:
                hypernetworks.append(self.hf_mlp(mask_tokens[:, index, :]))
        hyper = torch.stack(hypernetworks, dim=1)

        _, feature_channels, mask_h, mask_w = upscaled_sam.shape
        sam_masks = (
            hyper[:, : self.num_mask_tokens - 1]
            @ upscaled_sam.reshape(batch, feature_channels, mask_h * mask_w)
        ).reshape(batch, -1, mask_h, mask_w)
        hq_mask = (
            hyper[:, self.num_mask_tokens - 1 : self.num_mask_tokens]
            @ upscaled_hq.reshape(batch, feature_channels, mask_h * mask_w)
        ).reshape(batch, 1, mask_h, mask_w)

        iou_prediction = self.iou_prediction_head(iou_token)
        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            candidate_iou = iou_prediction[:, mask_slice]
            best_iou, best_index = torch.max(candidate_iou, dim=1)
            candidates = sam_masks[:, mask_slice]
            batch_index = torch.arange(batch, device=candidates.device)
            base_mask = candidates[batch_index, best_index].unsqueeze(1)
            iou_prediction = best_iou.unsqueeze(1)
        else:
            base_mask = sam_masks[:, :1]
            iou_prediction = iou_prediction[:, :1]

        final_mask = hq_mask if hq_token_only else base_mask + hq_mask
        if return_routing:
            return final_mask, iou_prediction, routing
        return final_mask, iou_prediction
