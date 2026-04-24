# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, Iterable, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F


class GroundTruthEncoder(nn.Module):
    """Encode class, box, and mask-context cues for LatentFormer ground-truth regions."""

    def __init__(
        self,
        *,
        num_classes: int,
        hidden_dim: int,
        sig_dim: int,
        feature_levels: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.feature_levels = tuple(feature_levels or ())
        self.num_classes = num_classes
        self.background_label = num_classes
        self.gt_cls_proj = nn.Embedding(num_classes + 1, sig_dim)
        self.gt_bbox_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sig_dim),
        )
        self.gt_mask_context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, sig_dim),
        )

    def _select_feature_maps(
        self, features: Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
    ) -> Iterable[torch.Tensor]:
        if isinstance(features, dict):
            if self.feature_levels:
                feature_maps = [features[name] for name in self.feature_levels if name in features]
            else:
                feature_maps = list(features.values())
            if len(feature_maps) == 0:
                raise ValueError("GroundTruthEncoder received no matching feature maps.")
            return feature_maps
        if isinstance(features, (list, tuple)):
            if len(features) == 0:
                raise ValueError("GroundTruthEncoder received an empty feature list.")
            return features
        return (features,)

    def _encode_gt_mask_context(
        self,
        feature_map: torch.Tensor,
        masks: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = feature_map.shape
        valid = pad_mask[:, :, None, None].to(dtype=feature_map.dtype)
        masks_small = F.interpolate(
            masks.float(), size=(height, width), mode="bilinear", align_corners=False
        )
        masks_small = masks_small * valid

        inside_denom = masks_small.flatten(2).sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled_inside = torch.einsum("bmhw,bchw->bmc", masks_small, feature_map) / inside_denom

        outside_masks = (1.0 - masks_small) * valid
        outside_denom = outside_masks.flatten(2).sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled_outside = torch.einsum("bmhw,bchw->bmc", outside_masks, feature_map) / outside_denom

        pooled_context = torch.cat([pooled_inside, pooled_outside], dim=-1)
        return self.gt_mask_context_proj(pooled_context)

    def forward(
        self,
        features: Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor],
        masks: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        feature_maps = list(self._select_feature_maps(features))
        boxes = boxes.to(dtype=feature_maps[0].dtype)

        labels = labels.clamp(min=0, max=self.gt_cls_proj.num_embeddings - 1)
        gt_signature = self.gt_cls_proj(labels) + self.gt_bbox_proj(boxes)
        for feature_map in feature_maps:
            gt_signature = gt_signature + self._encode_gt_mask_context(feature_map, masks, pad_mask)
        return gt_signature * pad_mask[:, :, None].to(dtype=gt_signature.dtype)
