# Copyright (c) Facebook, Inc. and its affiliates.
"""
LatentFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size

from ..utils.misc import is_dist_avail_and_initialized


def calculate_entropy_uncertainty(logits):
    """
    Estimate mask uncertainty from the entropy of the softmax distribution over latent signatures.

    Args:
        logits (Tensor): A tensor of shape (N, S, ...), where S is the number of valid
            ground-truth signatures, including the background signature.
    Returns:
        Tensor: Uncertainty scores shaped (N, 1, ...); larger values are more uncertain.
    """
    probs = logits.softmax(dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1, keepdim=True)


def soft_dice_loss(inputs, targets, valid_mask, eps=1.0):
    """
    Softmax DICE loss for mutually-exclusive latent masks.

    Args:
        inputs: logits shaped [B, S, P].
        targets: soft target distributions shaped [B, S, P].
        valid_mask: bool tensor shaped [B, S] selecting real signatures.
    """
    inputs = inputs.softmax(dim=1)
    numerator = 2 * (inputs * targets).sum(dim=-1)
    denominator = inputs.sum(dim=-1) + targets.sum(dim=-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss * valid_mask.to(dtype=loss.dtype)
    return loss.sum() / valid_mask.sum().clamp_min(1).to(dtype=loss.dtype)


class LatentCriterion(nn.Module):
    """Criterion for LatentFormer prototype predictions.

    LatentFormer produces one class logit vector and one mask logit map per ground-truth
    signature, plus a background signature. Since the signatures are already aligned with
    targets, no Hungarian matching is needed here.
    """

    def __init__(
        self,
        num_classes,
        weight_dict,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.background_label = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def _get_num_signatures(self, targets):
        num_signatures = targets["pad_mask"].sum()
        num_signatures = num_signatures.to(dtype=torch.float)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_signatures)
        return torch.clamp(num_signatures / get_world_size(), min=1).item()

    @staticmethod
    def _mask_invalid_slots(logits, pad_mask):
        mask_shape = (pad_mask.shape[0], pad_mask.shape[1]) + (1,) * (logits.dim() - 2)
        return logits.masked_fill(~pad_mask.view(mask_shape), -1e4)

    @staticmethod
    def _normalize_target_masks(target_masks, pad_mask):
        target_masks = target_masks * pad_mask[:, :, None, None].to(dtype=target_masks.dtype)
        denom = target_masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return target_masks / denom

    def _resize_target_masks(self, masks, size):
        return F.interpolate(masks.float(), size=size, mode="area")

    def loss_labels(self, outputs, targets, num_signatures):
        del num_signatures
        assert "proto_cls" in outputs
        src_logits = outputs["proto_cls"].float()
        target_classes = targets["labels"].to(device=src_logits.device)
        pad_mask = targets["pad_mask"].to(device=src_logits.device)

        loss_ce = F.cross_entropy(src_logits[pad_mask], target_classes[pad_mask])
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, num_signatures):
        del num_signatures
        assert "proto_masks" in outputs
        src_masks = outputs["proto_masks"]
        if not isinstance(src_masks, (list, tuple)):
            src_masks = [src_masks]

        target_masks = targets["masks"].to(device=src_masks[0].device)
        pad_mask = targets["pad_mask"].to(device=src_masks[0].device)

        losses = {}
        mask_ce_losses = []
        for src_masks_level in src_masks:
            src_masks_level = self._mask_invalid_slots(src_masks_level.float(), pad_mask)
            target_masks_level = self._resize_target_masks(
                target_masks,
                src_masks_level.shape[-2:],
            ).to(dtype=src_masks_level.dtype)
            target_masks_level = self._normalize_target_masks(target_masks_level, pad_mask)

            log_probs = F.log_softmax(src_masks_level, dim=1)
            mask_ce_losses.append(-(target_masks_level * log_probs).sum(dim=1).mean())

        losses["loss_mask"] = torch.stack(mask_ce_losses).mean()

        highest_res_masks = max(src_masks, key=lambda mask: mask.shape[-2] * mask.shape[-1])
        highest_res_masks = self._mask_invalid_slots(highest_res_masks.float(), pad_mask)
        target_masks = self._normalize_target_masks(target_masks.float(), pad_mask)

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                highest_res_masks,
                lambda logits: calculate_entropy_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            )
            point_labels = point_labels * pad_mask[:, :, None].to(dtype=point_labels.dtype)
            point_labels = point_labels / point_labels.sum(dim=1, keepdim=True).clamp_min(1e-6)

        point_logits = point_sample(
            highest_res_masks,
            point_coords,
            align_corners=False,
        )
        losses["loss_dice"] = soft_dice_loss(point_logits, point_labels, pad_mask)

        return losses

    def get_loss(self, loss, outputs, targets, num_signatures):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_signatures)

    def forward(self, outputs, targets):
        num_signatures = self._get_num_signatures(targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_signatures))
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
