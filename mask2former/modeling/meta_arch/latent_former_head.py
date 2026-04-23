# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict

from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..pixel_decoder.fpn import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class LatentFormerHead(nn.Module):
    """Sem-seg head scaffold for the LatentFormer architecture."""

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        latent_fpn: nn.Module,
        ignore_value: int = -1,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.in_features = [k for k, _ in sorted(input_shape.items(), key=lambda x: x[1].stride)]
        self.ignore_value = ignore_value
        self.loss_weight = loss_weight
        self.latent_fpn = latent_fpn
        self.num_classes = num_classes

        # TODO: replace these placeholders with LatentFormer prediction heads.
        self.predictor = nn.Identity()

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "latent_fpn": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        multi_scale_features = self.latent_fpn.forward_features(features)
        raise NotImplementedError(
            "LatentFormerHead prediction layers are not implemented yet. "
            "Use multi_scale_features as the FPN inputs."
        )
