# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict

from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..pixel_decoder.fpn import build_pixel_decoder
from ..transformer_decoder.latentformer_transformer_decoder import build_latentformer_transformer_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class LatentFormerHead(nn.Module):
    """Sem-seg head scaffold for the LatentFormer architecture."""

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        latent_fpn: nn.Module,
        transformer_predictor: nn.Module,
    ):
        super().__init__()
        self.in_features = [k for k, _ in sorted(input_shape.items(), key=lambda x: x[1].stride)]
        self.latent_fpn = latent_fpn
        self.predictor = transformer_predictor

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "latent_fpn": build_pixel_decoder(cfg, input_shape),
            "transformer_predictor": build_latentformer_transformer_decoder(
                cfg,
                cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        multi_scale_features = self.latent_fpn.forward_features(features)
        predictions = self.predictor(multi_scale_features)
        predictions["mask_features"] = multi_scale_features
        return predictions
