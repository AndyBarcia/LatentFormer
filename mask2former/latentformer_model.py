# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList

from .modeling.criterion_latent import LatentCriterion
from .modeling.gt_encoder import GroundTruthEncoder
from .modeling.matcher_latent import LatentMatcher
from .modeling.similarity import pairwise_similarity


def assignment_weights_from_similarity(
    *,
    similarity: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    weights = similarity.clamp_min(0.0)
    if valid_mask is not None:
        weights = weights * valid_mask.unsqueeze(-2).to(dtype=weights.dtype)
    return weights


def normalize_assignment_weights(
    weights: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    denom = weights.sum(dim=-2, keepdim=True).clamp_min(eps)
    return weights / denom


def aggregate_with_weights(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bqs,bqc->bsc", weights, values)


class LatentAggregator(nn.Module):
    """Aggregate per-layer query predictions into seed-conditioned prototypes."""

    def __init__(self, aggregation_similarity_metric: str = "dot"):
        super().__init__()
        self.aggregation_similarity_metric = aggregation_similarity_metric

    @staticmethod
    def _flatten_layer_queries(values: torch.Tensor, name: str) -> torch.Tensor:
        if values.dim() < 3:
            raise ValueError(f"{name} must have at least 3 dimensions, got {values.shape}.")
        if values.dim() == 3:
            return values
        if values.dim() == 4:
            layers, batch, queries = values.shape[:3]
            return values.permute(1, 0, 2, 3).reshape(batch, layers * queries, values.shape[-1])
        raise ValueError(f"{name} must be shaped [B,Q,C] or [L,B,Q,C], got {values.shape}.")

    def forward(
        self,
        query_class_logits: torch.Tensor,
        query_mask_embeddings: torch.Tensor,
        query_signatures: torch.Tensor,
        seed_signatures: torch.Tensor,
        *,
        mask_features,
        target_pad_mask: torch.Tensor = None,
    ):
        q_cls_flat = self._flatten_layer_queries(query_class_logits, "query_class_logits")
        q_mask_emb_flat = self._flatten_layer_queries(query_mask_embeddings, "query_mask_embeddings")
        q_sig_flat = self._flatten_layer_queries(query_signatures, "query_signatures")

        sim = pairwise_similarity(
            q_sig_flat,
            seed_signatures,
            metric=self.aggregation_similarity_metric,
        )
        weights_flat = assignment_weights_from_similarity(
            similarity=sim,
            valid_mask=target_pad_mask,
        )
        norm_w = normalize_assignment_weights(weights_flat)

        proto_mask_emb = aggregate_with_weights(norm_w, q_mask_emb_flat)
        proto_cls = aggregate_with_weights(norm_w, q_cls_flat)
        proto_masks = [torch.einsum("bsc,bchw->bshw", proto_mask_emb, feature) for feature in mask_features]

        if target_pad_mask is not None:
            valid = target_pad_mask.unsqueeze(-1).to(dtype=proto_cls.dtype)
            proto_cls = proto_cls * valid
            proto_mask_emb = proto_mask_emb * valid
            proto_masks = [
                masks * target_pad_mask[:, :, None, None].to(dtype=masks.dtype)
                for masks in proto_masks
            ]

        return proto_cls, proto_masks, proto_mask_emb


@META_ARCH_REGISTRY.register()
class LatentFormer(nn.Module):
    """Top-level LatentFormer scaffold."""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        matcher: nn.Module,
        criterion: nn.Module,
        gt_encoder: nn.Module,
        aggregator: nn.Module,
        num_queries: int,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        score_threshold: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.matcher = matcher
        self.criterion = criterion
        self.gt_encoder = gt_encoder
        self.aggregator = aggregator
        self.num_queries = num_queries
        self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.score_threshold = score_threshold

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        matcher = LatentMatcher(
            similarity_metric=cfg.MODEL.LATENT_FORMER.MATCHING_SIMILARITY_METRIC,
            seed_cost_weight=cfg.MODEL.LATENT_FORMER.SEED_COST_WEIGHT,
        )

        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        gt_sep_weight = cfg.MODEL.LATENT_FORMER.GT_SEP_WEIGHT
        seed_weight = cfg.MODEL.LATENT_FORMER.SEED_WEIGHT
        seed_sig_weight = cfg.MODEL.LATENT_FORMER.SEED_SIG_WEIGHT
        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_gt_sep": gt_sep_weight,
            "loss_seed": seed_weight,
            "loss_seed_sig": seed_sig_weight,
        }
        criterion = LatentCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=["labels", "masks", "gt_sep", "seed"],
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            similarity_metric=cfg.MODEL.LATENT_FORMER.MATCHING_SIMILARITY_METRIC,
        )
        gt_encoder = GroundTruthEncoder(
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            hidden_dim=cfg.MODEL.LATENT_FORMER.GT_ENCODER.HIDDEN_DIM,
            sig_dim=cfg.MODEL.LATENT_FORMER.GT_ENCODER.SIG_DIM,
            feature_levels=cfg.MODEL.LATENT_FORMER.GT_ENCODER.FEATURE_LEVELS,
        )
        aggregator = LatentAggregator(
            aggregation_similarity_metric=cfg.MODEL.LATENT_FORMER.AGGREGATION_SIMILARITY_METRIC,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "matcher": matcher,
            "criterion": criterion,
            "gt_encoder": gt_encoder,
            "aggregator": aggregator,
            "num_queries": cfg.MODEL.LATENT_FORMER.NUM_OBJECT_QUERIES,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.LATENT_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.LATENT_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.LATENT_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.LATENT_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": cfg.MODEL.LATENT_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.LATENT_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.LATENT_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "score_threshold": cfg.MODEL.LATENT_FORMER.TEST.SCORE_THRESHOLD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_gt_encoder_inputs(gt_instances, images)
            outputs["gt_signatures"] = self.gt_encoder(
                features,
                targets["masks"],
                targets["labels"],
                targets["boxes"],
                targets["pad_mask"],
            )
            proto_cls, proto_masks, proto_mask_emb = self.aggregator(
                outputs["pred_logits"],
                outputs["pred_masks"],
                outputs["pred_signatures"],
                outputs["gt_signatures"],
                mask_features=outputs["mask_features"],
                target_pad_mask=targets["pad_mask"],
            )

            outputs["proto_cls"] = proto_cls
            outputs["proto_masks"] = proto_masks
            outputs["proto_mask_emb"] = proto_mask_emb

            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:
            raise NotImplementedError("LatentFormer inference aggregation is not implemented yet.")

        return self.inference(outputs, batched_inputs, images.image_sizes)

    def inference(self, outputs, batched_inputs, image_sizes):
        raise NotImplementedError("LatentFormer inference path is not implemented yet.")

    def prepare_gt_encoder_inputs(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        background_label = self.gt_encoder.background_label
        max_instances = max(
            (len(targets_per_image.gt_classes) for targets_per_image in targets), default=0
        ) + 1
        batch_size = len(targets)

        masks = torch.zeros(
            (batch_size, max_instances, h_pad, w_pad),
            dtype=images.tensor.dtype,
            device=self.device,
        )
        labels = torch.full(
            (batch_size, max_instances),
            background_label,
            dtype=torch.long,
            device=self.device,
        )
        boxes = torch.zeros(
            (batch_size, max_instances, 4),
            dtype=images.tensor.dtype,
            device=self.device,
        )
        pad_mask = torch.zeros((batch_size, max_instances), dtype=torch.bool, device=self.device)

        for idx, targets_per_image in enumerate(targets):
            image_height, image_width = images.image_sizes[idx]
            num_instances = len(targets_per_image.gt_classes)
            background_idx = num_instances
            masks[idx, background_idx, :image_height, :image_width] = 1.0
            boxes[idx, background_idx] = boxes.new_tensor(
                [
                    image_width * 0.5 / float(w_pad),
                    image_height * 0.5 / float(h_pad),
                    image_width / float(w_pad),
                    image_height / float(h_pad),
                ]
            )
            pad_mask[idx, background_idx] = True

            if num_instances > 0 and not hasattr(targets_per_image, "gt_boxes"):
                raise ValueError("LatentFormer GroundTruthEncoder requires annotation gt_boxes.")
            if num_instances == 0:
                continue

            gt_masks = targets_per_image.gt_masks
            if hasattr(gt_masks, "tensor"):
                gt_masks = gt_masks.tensor
            gt_masks = gt_masks.to(device=self.device)
            masks[idx, :num_instances, : gt_masks.shape[-2], : gt_masks.shape[-1]] = gt_masks.to(
                dtype=masks.dtype
            )
            labels[idx, :num_instances] = targets_per_image.gt_classes
            bg_height = min(gt_masks.shape[-2], image_height)
            bg_width = min(gt_masks.shape[-1], image_width)
            masks[idx, background_idx, :bg_height, :bg_width] = 1.0 - gt_masks[
                :, :bg_height, :bg_width
            ].to(dtype=masks.dtype).clamp(0, 1).amax(dim=0)
            gt_boxes = targets_per_image.gt_boxes.tensor.to(device=self.device, dtype=boxes.dtype)
            x0, y0, x1, y1 = gt_boxes.unbind(dim=-1)
            boxes[idx, :num_instances] = torch.stack(
                [
                    (x0 + x1) * 0.5 / float(w_pad),
                    (y0 + y1) * 0.5 / float(h_pad),
                    (x1 - x0).clamp_min(0.0) / float(w_pad),
                    (y1 - y0).clamp_min(0.0) / float(h_pad),
                ],
                dim=-1,
            )
            pad_mask[idx, :num_instances] = True

        return {"masks": masks, "labels": labels, "boxes": boxes, "pad_mask": pad_mask}
