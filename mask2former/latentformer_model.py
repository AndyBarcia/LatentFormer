# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList

from .modeling.gt_encoder import GroundTruthEncoder
from .modeling.matcher_latent import LatentMatcher


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
        gt_encoder: nn.Module,
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
        self.gt_encoder = gt_encoder
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
            similarity_metric=cfg.MODEL.LATENT_FORMER.SIMILARITY_METRIC,
            seed_cost_weight=cfg.MODEL.LATENT_FORMER.SEED_COST_WEIGHT,
        )
        gt_encoder = GroundTruthEncoder(
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            hidden_dim=cfg.MODEL.LATENT_FORMER.GT_ENCODER.HIDDEN_DIM,
            sig_dim=cfg.MODEL.LATENT_FORMER.GT_ENCODER.SIG_DIM,
            feature_levels=cfg.MODEL.LATENT_FORMER.GT_ENCODER.FEATURE_LEVELS,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "matcher": matcher,
            "gt_encoder": gt_encoder,
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
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_gt_encoder_inputs(gt_instances, images)
                outputs["gt_signature"] = self.gt_encoder(
                    features,
                    targets["masks"],
                    targets["labels"],
                    targets["boxes"],
                    targets["pad_mask"],
                )
                outputs["gt_pad_mask"] = targets["pad_mask"]
            raise NotImplementedError("LatentFormer training losses are not implemented yet.")

        return self.inference(outputs, batched_inputs, images.image_sizes)

    def inference(self, outputs, batched_inputs, image_sizes):
        raise NotImplementedError("LatentFormer inference path is not implemented yet.")

    def prepare_gt_encoder_inputs(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        max_instances = max(
            (len(targets_per_image.gt_classes) for targets_per_image in targets), default=0
        )
        batch_size = len(targets)

        masks = torch.zeros(
            (batch_size, max_instances, h_pad, w_pad),
            dtype=images.tensor.dtype,
            device=self.device,
        )
        labels = torch.zeros((batch_size, max_instances), dtype=torch.long, device=self.device)
        boxes = torch.zeros(
            (batch_size, max_instances, 4),
            dtype=images.tensor.dtype,
            device=self.device,
        )
        pad_mask = torch.zeros((batch_size, max_instances), dtype=torch.bool, device=self.device)

        for idx, targets_per_image in enumerate(targets):
            num_instances = len(targets_per_image.gt_classes)
            if num_instances == 0:
                continue
            if not hasattr(targets_per_image, "gt_boxes"):
                raise ValueError("LatentFormer GroundTruthEncoder requires annotation gt_boxes.")

            gt_masks = targets_per_image.gt_masks
            if hasattr(gt_masks, "tensor"):
                gt_masks = gt_masks.tensor
            gt_masks = gt_masks.to(device=self.device)
            masks[idx, :num_instances, : gt_masks.shape[-2], : gt_masks.shape[-1]] = gt_masks.to(
                dtype=masks.dtype
            )
            labels[idx, :num_instances] = targets_per_image.gt_classes
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
