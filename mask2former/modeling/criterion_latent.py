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

from .similarity import pairwise_similarity
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


def soft_dice_loss(inputs, targets, valid_mask, num_masks, eps=1.0):
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
    return loss.sum() / num_masks


class LatentCriterion(nn.Module):
    """Criterion for LatentFormer prototype predictions.

    LatentFormer produces one class logit vector and one mask logit map per ground-truth
    signature, plus a background signature. Prototype losses are already aligned with
    targets; seed losses use Hungarian matching over query and ground-truth signatures.
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        similarity_metric,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.background_label = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.similarity_metric = similarity_metric

    @staticmethod
    def _get_global_normalizer(counts):
        counts = counts.to(dtype=torch.float)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(counts)
        return torch.clamp(counts / get_world_size(), min=1).item()

    @staticmethod
    def _mask_invalid_slots(logits, pad_mask):
        mask_shape = (pad_mask.shape[0], pad_mask.shape[1]) + (1,) * (logits.dim() - 2)
        return logits.masked_fill(~pad_mask.view(mask_shape), -1e4)

    @staticmethod
    def _normalize_target_masks(target_masks, pad_mask):
        target_masks = target_masks * pad_mask[:, :, None, None].to(dtype=target_masks.dtype)
        denom = target_masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return target_masks / denom

    @staticmethod
    def _flatten_query_features(values, name):
        if values.dim() == 3:
            return values
        if values.dim() == 4:
            layers, batch, queries = values.shape[:3]
            return values.permute(1, 0, 2, 3).reshape(batch, layers * queries, values.shape[-1])
        raise ValueError(f"{name} must be shaped [B,Q,C] or [L,B,Q,C], got {values.shape}.")

    @staticmethod
    def _flatten_query_logits(values, name):
        if values.dim() == 2:
            return values
        if values.dim() == 3:
            layers, batch, queries = values.shape
            return values.permute(1, 0, 2).reshape(batch, layers * queries)
        raise ValueError(f"{name} must be shaped [B,Q] or [L,B,Q], got {values.shape}.")

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
            mask_ce_loss = -(target_masks_level * log_probs).flatten(2).mean(dim=-1)
            mask_ce_loss = mask_ce_loss * pad_mask.to(dtype=mask_ce_loss.dtype)
            mask_ce_losses.append(mask_ce_loss.sum() / num_signatures)

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
        losses["loss_dice"] = soft_dice_loss(point_logits, point_labels, pad_mask, num_signatures)

        return losses

    def loss_gt_sep(self, outputs, targets, num_signatures):
        del num_signatures
        assert "gt_signatures" in outputs
        gt_signatures = outputs["gt_signatures"]
        pad_mask = targets["pad_mask"].to(device=gt_signatures.device)
        loss_sep = gt_signatures.sum() * 0.0

        num_gt = pad_mask.shape[1]
        if num_gt <= 1:
            return {"loss_gt_sep": loss_sep}

        gt_sim = pairwise_similarity(
            gt_signatures,
            gt_signatures,
            metric=self.similarity_metric,
        )
        eye = torch.eye(num_gt, dtype=torch.bool, device=gt_signatures.device).unsqueeze(0)
        valid_pair_mask = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1)
        off_diag_mask = valid_pair_mask & ~eye

        if off_diag_mask.any():
            valid_pair_count = self._get_global_normalizer(off_diag_mask.sum())
            loss_sep = F.relu(gt_sim[off_diag_mask]).pow(2).sum() / valid_pair_count

        return {"loss_gt_sep": loss_sep}

    def loss_seed(self, outputs, targets, num_signatures):
        assert "pred_signatures" in outputs
        assert "pred_seed_logits" in outputs
        assert "gt_signatures" in outputs

        q_sig_flat = self._flatten_query_features(
            outputs["pred_signatures"],
            "pred_signatures",
        )
        q_seed_logits_flat = self._flatten_query_logits(
            outputs["pred_seed_logits"],
            "pred_seed_logits",
        ).float()
        gt_signatures = outputs["gt_signatures"].to(device=q_sig_flat.device)
        pad_mask = targets["pad_mask"].to(device=q_sig_flat.device)

        matched_query_mask, matched_gt_indices = self.matcher(
            q_sig_flat,
            gt_signatures,
            pad_mask,
        )

        seed_targets = matched_query_mask.to(dtype=q_seed_logits_flat.dtype)
        loss_seed = F.binary_cross_entropy_with_logits(q_seed_logits_flat, seed_targets)

        loss_seed_sig = q_sig_flat.sum() * 0.0
        matched_pos = matched_query_mask.nonzero(as_tuple=False)
        if matched_pos.numel() > 0:
            matched_gt = matched_gt_indices[matched_query_mask]
            matched_q_sig = q_sig_flat[matched_query_mask]
            matched_gt_sig = gt_signatures[matched_pos[:, 0], matched_gt].detach()
            matched_similarity = pairwise_similarity(
                matched_q_sig.unsqueeze(1),
                matched_gt_sig.unsqueeze(1),
                metric=self.similarity_metric,
            ).squeeze(-1).squeeze(-1)
            loss_seed_sig = (1.0 - matched_similarity).sum() / num_signatures

        return {
            "loss_seed": loss_seed,
            "loss_seed_sig": loss_seed_sig,
        }

    def get_loss(self, loss, outputs, targets, num_signatures):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "gt_sep": self.loss_gt_sep,
            "seed": self.loss_seed,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_signatures)

    def forward(self, outputs, targets):
        num_signatures = self._get_global_normalizer(targets["pad_mask"].sum())

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_signatures))
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
            "similarity_metric: {}".format(self.similarity_metric),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
