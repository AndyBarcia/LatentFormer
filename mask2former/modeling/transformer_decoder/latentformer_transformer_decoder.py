# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn

from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY


def build_latentformer_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build the LatentFormer transformer decoder from ``cfg.MODEL.LATENT_FORMER``.
    """
    name = cfg.MODEL.LATENT_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


@TRANSFORMER_DECODER_REGISTRY.register()
class LatentTransformerDecoder(nn.Module):
    """Dummy LatentFormer decoder scaffold."""

    def __init__(self, cfg, in_channels, mask_classification=True):
        super().__init__()
        self.in_channels = in_channels
        self.mask_classification = mask_classification

    def forward(self, multi_scale_features):
        return {"multi_scale_features": multi_scale_features}
