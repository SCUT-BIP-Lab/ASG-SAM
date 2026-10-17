"""ASG-SAM model assembly and checkpoint utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .asg import AnisotropicSpectralGating, ASGDelta
from .decoder import MaskDecoder
from .encoder import ImageEncoderViT
from .prompt_encoder import PromptEncoder
from .two_way_transformer import TwoWayTransformer


class ASGSAM(nn.Module):
    """Paper configuration of ASG-SAM built on EfficientSAM-Ti."""

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, 3, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, 3, 1, 1), False)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (self.image_encoder.img_size, self.image_encoder.img_size):
            images = F.interpolate(
                images,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return (images - self.pixel_mean) / self.pixel_std

    def get_image_embeddings(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image_encoder(self.preprocess(images))

    def _rescale_points(
        self,
        points: torch.Tensor,
        input_size: tuple[int, int],
    ) -> torch.Tensor:
        input_h, input_w = input_size
        return torch.stack(
            (
                torch.where(
                    points[..., 0] >= 0,
                    points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    points[..., 1] >= 0,
                    points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ),
            dim=-1,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
        points: torch.Tensor,
        point_labels: torch.Tensor,
        *,
        input_size: tuple[int, int],
        output_size: tuple[int, int] | None = None,
        multimask_output: bool = False,
        hq_token_only: bool = False,
        return_routing: bool = False,
    ):
        batch_size, queries, num_points, _ = points.shape
        scaled_points = self._rescale_points(points, input_size)
        sparse, dense = self.prompt_encoder(
            points=(
                scaled_points.reshape(batch_size * queries, num_points, 2),
                point_labels.reshape(batch_size * queries, num_points),
            ),
            boxes=None,
            masks=None,
        )
        image_embeddings = image_embeddings.repeat_interleave(queries, dim=0)
        interm_embeddings = interm_embeddings.repeat_interleave(queries, dim=0)
        decoder_output = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse,
            dense,
            multimask_output,
            hq_token_only,
            interm_embeddings,
            return_routing=return_routing,
        )
        low_res_masks, iou_prediction = decoder_output[:2]
        target_size = output_size or low_res_masks.shape[-2:]
        masks = F.interpolate(
            low_res_masks,
            target_size,
            mode="bicubic",
            align_corners=False,
        )
        masks = masks.reshape(batch_size, queries, 1, *target_size)
        iou_prediction = iou_prediction.reshape(batch_size, queries, 1)
        if return_routing:
            routing = decoder_output[2].reshape(batch_size, queries, 3)
            return masks, iou_prediction, routing
        return masks, iou_prediction

    def forward(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        point_labels: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
        hq_token_only: bool = False,
        return_routing: bool = False,
    ):
        input_size = tuple(images.shape[-2:])
        embeddings, detail = self.get_image_embeddings(images)
        return self.predict_masks(
            embeddings,
            detail,
            points,
            point_labels,
            input_size=input_size,
            output_size=output_size or input_size,
            hq_token_only=hq_token_only,
            return_routing=return_routing,
        )


def _state_dict_from_checkpoint(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("model"), Mapping):
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a model state dictionary.")
    return {
        (key.removeprefix("module.")): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Load an EfficientSAM or ASG-SAM checkpoint.

    ``strict=False`` performs shape-matched loading so the public
    EfficientSAM-Ti checkpoint can initialize the added HQ/ASG/DoG-AMGD layers.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = _state_dict_from_checkpoint(checkpoint)
    if strict:
        incompatible = model.load_state_dict(source, strict=True)
        loaded = len(source)
    else:
        destination = model.state_dict()
        matched = {
            key: value
            for key, value in source.items()
            if key in destination and destination[key].shape == value.shape
        }
        incompatible = model.load_state_dict(matched, strict=False)
        loaded = len(matched)
    return {
        "loaded_tensors": loaded,
        "source_tensors": len(source),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "metadata": checkpoint if isinstance(checkpoint, Mapping) else {},
    }


def build_asg_sam(
    pretrained: str | Path | None = None,
    *,
    radial_bins: int = 64,
    angular_bins: int = 128,
    encoder_asg_strength: float = 1.0,
    decoder_asg_strength: float = 1.0,
) -> ASGSAM:
    """Build the exact EfficientSAM-Ti configuration used in the paper."""

    image_size = 1024
    embedding_size = 64
    transformer_dim = 256
    encoder_dim = 192

    image_encoder = ImageEncoderViT(
        img_size=image_size,
        patch_size=16,
        in_channels=3,
        embed_dim=encoder_dim,
        depth=12,
        num_heads=3,
        detail_layer=0,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=transformer_dim,
        image_embedding_size=(embedding_size, embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_channels=16,
    )
    mask_decoder = MaskDecoder(
        transformer_dim=transformer_dim,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=2048,
            num_heads=8,
            activation=nn.GELU,
            normalize_before_activation=False,
        ),
        num_multimask_outputs=1,
        vit_dim=encoder_dim,
    )
    model = ASGSAM(image_encoder, prompt_encoder, mask_decoder)

    image_encoder.radial_gate = ASGDelta(
        AnisotropicSpectralGating(
            transformer_dim,
            embedding_size,
            embedding_size,
            radial_bins,
            angular_bins,
        )
    )
    image_encoder.rgate_strength = float(encoder_asg_strength)
    mask_decoder.radial_gate = ASGDelta(
        AnisotropicSpectralGating(
            transformer_dim // 8,
            embedding_size * 4,
            embedding_size * 4,
            radial_bins,
            angular_bins,
        )
    )
    mask_decoder.rgate_strength_dec = float(decoder_asg_strength)

    if pretrained is not None:
        load_checkpoint(model, pretrained, strict=False)
    return model
