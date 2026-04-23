# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList

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

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "matcher": matcher,
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
            raise NotImplementedError("LatentFormer training losses are not implemented yet.")

        return self.inference(outputs, batched_inputs, images.image_sizes)

    def inference(self, outputs, batched_inputs, image_sizes):
        raise NotImplementedError("LatentFormer inference path is not implemented yet.")
