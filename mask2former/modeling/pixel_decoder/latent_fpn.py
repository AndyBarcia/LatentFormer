# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, Dict, Optional, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


@SEM_SEG_HEADS_REGISTRY.register()
class LatentFPN(nn.Module):
    """Simple Feature Pyramid Network scaffold for LatentFormer."""

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        num_feature_levels: int = 3,
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, _ in input_shape]
        feature_channels = [v.channels for _, v in input_shape]
        self.num_feature_levels = num_feature_levels

        use_bias = norm == ""
        lateral_convs = []
        output_convs = []
        for idx, in_channels in enumerate(feature_channels):
            lateral_conv = Conv2d(
                in_channels,
                conv_dim,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, conv_dim),
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "conv_dim": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "num_feature_levels": cfg.MODEL.SEM_SEG_HEAD.get("LATENT_FPN_NUM_LEVELS", 3),
        }

    def forward_features(self, features, mask=None):
        multi_scale_features = []
        y = None
        for idx, feature_name in enumerate(self.in_features[::-1]):
            lateral = self.lateral_convs[idx](features[feature_name])
            if y is None:
                y = lateral
            else:
                y = lateral + F.interpolate(y, size=lateral.shape[-2:], mode="nearest")
            y = self.output_convs[idx](y)
            if len(multi_scale_features) < self.num_feature_levels:
                multi_scale_features.append(y)

        return multi_scale_features

    def forward(self, features, targets=None):
        return self.forward_features(features)
