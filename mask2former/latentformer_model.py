# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion_latent import LatentCriterion
from .modeling.gt_encoder import GroundTruthEncoder
from .modeling.matcher_latent import LatentMatcher
from .modeling.seed_selection import build_seed_selection_modules
from .modeling.similarity import pairwise_similarity
from .utils.padding import image_padding_mask


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

    @staticmethod
    def _flatten_layer_logits(values: torch.Tensor, name: str) -> torch.Tensor:
        if values.dim() == 2:
            return values
        if values.dim() == 3:
            layers, batch, queries = values.shape
            return values.permute(1, 0, 2).reshape(batch, layers * queries)
        raise ValueError(f"{name} must be shaped [B,Q] or [L,B,Q], got {values.shape}.")

    def forward(
        self,
        query_mask_embeddings: torch.Tensor,
        query_mask_signatures: torch.Tensor,
        query_class_signatures: torch.Tensor,
        query_mask_seed_logits: torch.Tensor,
        query_class_seed_logits: torch.Tensor,
        seed_mask_signatures: torch.Tensor,
        *,
        class_signatures: torch.Tensor,
        mask_features,
        target_pad_mask: torch.Tensor = None,
        class_pad_mask: torch.Tensor = None,
    ):
        q_mask_emb_flat = self._flatten_layer_queries(query_mask_embeddings, "query_mask_embeddings")
        q_mask_sig_flat = self._flatten_layer_queries(query_mask_signatures, "query_mask_signatures")
        q_class_sig_flat = self._flatten_layer_queries(query_class_signatures, "query_class_signatures")
        q_mask_seed_logits_flat = self._flatten_layer_logits(
            query_mask_seed_logits,
            "query_mask_seed_logits",
        )
        q_class_seed_logits_flat = self._flatten_layer_logits(
            query_class_seed_logits,
            "query_class_seed_logits",
        )
        non_mask_seed_gate = 1.0 - q_mask_seed_logits_flat.float().sigmoid()
        class_seed_gate = q_class_seed_logits_flat.float().sigmoid()

        object_sim = pairwise_similarity(
            q_mask_sig_flat,
            seed_mask_signatures,
            metric=self.aggregation_similarity_metric,
        )
        object_weights_flat = assignment_weights_from_similarity(
            similarity=object_sim,
            valid_mask=target_pad_mask,
        )
        object_weights_flat = object_weights_flat * non_mask_seed_gate.to(
            dtype=object_weights_flat.dtype
        ).unsqueeze(-1)
        object_norm_w = normalize_assignment_weights(object_weights_flat)

        object_mask_emb = aggregate_with_weights(object_norm_w, q_mask_emb_flat)
        object_masks = [
            torch.einsum("bsc,bchw->bshw", object_mask_emb, feature) for feature in mask_features
        ]

        class_signatures = class_signatures.to(device=q_class_sig_flat.device, dtype=q_class_sig_flat.dtype)
        if class_signatures.dim() == 2:
            class_signatures = class_signatures.unsqueeze(0).expand(q_class_sig_flat.shape[0], -1, -1)
        elif class_signatures.dim() != 3:
            raise ValueError(
                f"class_signatures must be shaped [C,D] or [B,C,D], got {class_signatures.shape}."
            )
        if class_pad_mask is None:
            class_pad_mask = torch.ones(
                class_signatures.shape[:2],
                dtype=torch.bool,
                device=q_class_sig_flat.device,
            )
        else:
            class_pad_mask = class_pad_mask.to(device=q_class_sig_flat.device)
        semantic_sim = pairwise_similarity(
            q_class_sig_flat,
            class_signatures,
            metric=self.aggregation_similarity_metric,
        )
        semantic_weights = assignment_weights_from_similarity(
            similarity=semantic_sim,
            valid_mask=class_pad_mask,
        )
        semantic_weights = semantic_weights * class_seed_gate.to(
            dtype=semantic_weights.dtype
        ).unsqueeze(-1)
        semantic_weights = normalize_assignment_weights(
            semantic_weights
        )
        semantic_mask_emb = aggregate_with_weights(semantic_weights, q_mask_emb_flat)
        semantic_masks = [
            torch.einsum("bsc,bchw->bshw", semantic_mask_emb, feature) for feature in mask_features
        ]
        semantic_mask_emb = semantic_mask_emb * class_pad_mask[:, :, None].to(
            dtype=semantic_mask_emb.dtype
        )
        semantic_masks = [
            masks * class_pad_mask[:, :, None, None].to(dtype=masks.dtype)
            for masks in semantic_masks
        ]

        if target_pad_mask is not None:
            valid = target_pad_mask.unsqueeze(-1).to(dtype=object_mask_emb.dtype)
            object_mask_emb = object_mask_emb * valid
            object_masks = [
                masks * target_pad_mask[:, :, None, None].to(dtype=masks.dtype)
                for masks in object_masks
            ]

        return object_masks, semantic_masks, object_mask_emb, semantic_mask_emb


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
        seed_selection_modules: nn.Module,
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
        overlap_threshold: float,
        signature_on: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.matcher = matcher
        self.criterion = criterion
        self.gt_encoder = gt_encoder
        self.aggregator = aggregator
        self.seed_selection_modules = seed_selection_modules
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
        self.overlap_threshold = overlap_threshold
        self.signature_on = signature_on

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        matcher = LatentMatcher(similarity_metric=cfg.MODEL.LATENT_FORMER.MATCHING_SIMILARITY_METRIC,)

        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        gt_sep_weight = cfg.MODEL.LATENT_FORMER.GT_SEP_WEIGHT
        seed_weight = cfg.MODEL.LATENT_FORMER.SEED_WEIGHT
        seed_sig_weight = cfg.MODEL.LATENT_FORMER.SEED_SIG_WEIGHT
        seed_weight_pattern_weight = cfg.MODEL.LATENT_FORMER.SEED_WEIGHT_PATTERN_WEIGHT
        seed_cluster_pr_weight = cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR_WEIGHT
        weight_dict = {
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_sem_mask": mask_weight,
            "loss_sem_dice": dice_weight,
            "loss_gt_sep": gt_sep_weight,
            "loss_class_gt_sep": gt_sep_weight,
            "loss_seed": seed_weight,
            "loss_seed_sig": seed_sig_weight,
            "loss_seed_weight": seed_weight_pattern_weight,
            "loss_seed_cluster_pr": seed_cluster_pr_weight,
            "loss_class_seed": seed_weight,
            "loss_class_seed_sig": seed_sig_weight,
            "loss_class_seed_weight": seed_weight_pattern_weight,
            "loss_class_seed_cluster_pr": seed_cluster_pr_weight,
        }
        eval_modes = tuple(cfg.MODEL.LATENT_FORMER.TEST.EVAL_MODES)
        seed_selection_modules = build_seed_selection_modules(
            eval_modes=eval_modes,
            similarity_metric=cfg.MODEL.LATENT_FORMER.AGGREGATION_SIMILARITY_METRIC,
            seed_cluster_pr_num_seed_thresholds=cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR.NUM_SEED_THRESHOLDS,
            seed_cluster_pr_num_duplicate_thresholds=(
                cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR.NUM_DUPLICATE_THRESHOLDS
            ),
            seed_cluster_pr_seed_threshold_range=(
                cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR.SEED_THRESHOLD_RANGE
            ),
            seed_cluster_pr_duplicate_threshold_range=(
                cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR.DUPLICATE_THRESHOLD_RANGE
            ),
            seed_cluster_pr_hidden_dim=cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR.HIDDEN_DIM,
            seed_cluster_pr_inference_num_points=(
                cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR.INFERENCE_NUM_POINTS
            ),
        )
        criterion = LatentCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=["masks", "semantic_masks", "gt_sep", "seed"],
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            similarity_metric=cfg.MODEL.LATENT_FORMER.MATCHING_SIMILARITY_METRIC,
            aggregation_similarity_metric=cfg.MODEL.LATENT_FORMER.AGGREGATION_SIMILARITY_METRIC,
        )
        gt_encoder = GroundTruthEncoder(
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            hidden_dim=cfg.MODEL.LATENT_FORMER.GT_ENCODER.HIDDEN_DIM,
            sig_dim=cfg.MODEL.LATENT_FORMER.GT_ENCODER.SIG_DIM,
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
            "seed_selection_modules": seed_selection_modules,
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
            "overlap_threshold": cfg.MODEL.LATENT_FORMER.TEST.OVERLAP_THRESHOLD,
            "signature_on": cfg.MODEL.LATENT_FORMER.TEST.SIGNATURE_ON,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(
            features,
            mask=image_padding_mask(images),
        )
        outputs["class_signatures"] = self.gt_encoder.all_class_signatures()

        if self._needs_gt_signatures():
            targets = self._prepare_gt_signatures(batched_inputs, images, outputs)
        else:
            targets = None

        if self.training:
            object_masks, semantic_masks, object_mask_emb, semantic_mask_emb = self.aggregator(
                outputs["pred_masks"],
                outputs["pred_mask_signatures"],
                outputs["pred_class_signatures"],
                outputs["pred_mask_seed_logits"],
                outputs["pred_class_seed_logits"],
                outputs["gt_mask_signatures"],
                class_signatures=outputs["gt_semantic_class_signatures"],
                mask_features=outputs["mask_features"],
                target_pad_mask=targets["pad_mask"],
                class_pad_mask=targets["semantic_pad_mask"],
            )

            outputs["object_masks"] = object_masks
            outputs["semantic_masks"] = semantic_masks
            outputs["object_mask_emb"] = object_mask_emb
            outputs["semantic_mask_emb"] = semantic_mask_emb
            outputs["clustering_seed_selection"] = self.seed_selection_modules[
                "ClusteringSeedSelection"
            ]

            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:
            return self.inference(
                outputs,
                batched_inputs,
                images.image_sizes,
                padded_image_size=images.tensor.shape[-2:],
                targets=targets,
            )

    def _needs_gt_signatures(self):
        return self.training or self.signature_on or any(
            seed_selection.requires_gt_signatures
            for seed_selection in self.seed_selection_modules.values()
        )

    def _prepare_gt_signatures(self, batched_inputs, images, outputs):
        if any("instances" not in x for x in batched_inputs):
            raise ValueError(
                "LatentFormer needs GT instances to encode GT signatures, but the current batch "
                "does not include them. For GT-backed eval modes, enable "
                "MODEL.LATENT_FORMER.TEST.LOAD_GT_FOR_EVAL."
            )

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_gt_encoder_inputs(gt_instances, images, batched_inputs)
        outputs["gt_mask_signatures"] = self.gt_encoder(
            outputs["mask_features"],
            targets["masks"],
            targets["boxes"],
            targets["pad_mask"],
        )
        outputs["gt_class_signatures"] = self.gt_encoder.class_signatures(targets["labels"])
        if "semantic_labels" in targets:
            semantic_class_signatures = self.gt_encoder.class_signatures(targets["semantic_labels"])
            outputs["gt_semantic_class_signatures"] = (
                semantic_class_signatures
                * targets["semantic_pad_mask"][:, :, None].to(dtype=semantic_class_signatures.dtype)
            )
        outputs["class_signatures"] = self.gt_encoder.all_class_signatures()
        return targets

    def inference(self, outputs, batched_inputs, image_sizes, padded_image_size, targets=None):
        mode_results = {}
        for mode, seed_selection in self.seed_selection_modules.items():
            gt_signatures = outputs.get("gt_mask_signatures")
            gt_pad_mask = targets["pad_mask"] if targets is not None else None
            seed_signatures, seed_pad_mask, seed_scores = seed_selection(
                outputs["pred_mask_signatures"],
                outputs["pred_mask_seed_logits"],
                gt_signatures=gt_signatures,
                gt_pad_mask=gt_pad_mask,
                policy="mask",
            )
            gt_class_signatures = outputs.get("gt_semantic_class_signatures")
            gt_class_pad_mask = targets.get("semantic_pad_mask") if targets is not None else None
            if mode == "ClusteringSeedSelection":
                class_seed_signatures, class_seed_pad_mask, class_seed_scores = (
                    self.clustering_class_seed_selection(
                        seed_selection,
                        outputs["pred_class_signatures"],
                        outputs["pred_class_seed_logits"],
                        outputs["class_signatures"],
                    )
                )
            else:
                class_seed_signatures, class_seed_pad_mask, class_seed_scores = seed_selection(
                    outputs["pred_class_signatures"],
                    outputs["pred_class_seed_logits"],
                    gt_signatures=gt_class_signatures,
                    gt_pad_mask=gt_class_pad_mask,
                    policy="class",
                )
            class_seed_labels = self.class_labels_from_signatures(
                class_seed_signatures,
                class_seed_pad_mask,
                outputs["class_signatures"],
            )
            mode_results[mode] = self._inference_with_seeds(
                outputs,
                batched_inputs,
                image_sizes,
                padded_image_size,
                seed_signatures,
                seed_pad_mask,
                seed_scores,
                class_seed_signatures,
                class_seed_pad_mask,
                class_seed_labels,
                class_seed_scores,
                targets=targets,
            )

        if len(mode_results) == 1:
            return next(iter(mode_results.values()))
        return mode_results

    def clustering_class_seed_selection(
        self,
        seed_selection,
        query_class_signatures,
        query_class_seed_logits,
        all_class_signatures,
    ):
        q_sig_flat = LatentAggregator._flatten_layer_queries(
            query_class_signatures,
            "pred_class_signatures",
        )
        q_logits_flat = self._flatten_layer_logits(
            query_class_seed_logits,
            "pred_class_seed_logits",
        ).float()
        q_scores = q_logits_flat.sigmoid()

        seed_threshold, _ = seed_selection.best_thresholds(policy="class")
        seed_threshold = max(float(seed_threshold), 0.03)
        selected_mask = q_scores >= seed_threshold
        empty_selection = ~selected_mask.any(dim=1)
        if empty_selection.any():
            fallback_indices = q_scores.argmax(dim=1)
            selected_mask = selected_mask.clone()
            selected_mask[empty_selection, fallback_indices[empty_selection]] = True

        all_class_signatures = all_class_signatures.to(
            device=q_sig_flat.device,
            dtype=q_sig_flat.dtype,
        )
        all_class_signatures = all_class_signatures.unsqueeze(0).expand(
            q_sig_flat.shape[0],
            -1,
            -1,
        )
        predicted_labels = pairwise_similarity(
            q_sig_flat,
            all_class_signatures,
            metric=self.aggregator.aggregation_similarity_metric,
        ).argmax(dim=-1)

        batch_seed_signatures = []
        batch_seed_scores = []
        max_num_seeds = 0
        for signatures_per_image, scores_per_image, labels_per_image, selected in zip(
            q_sig_flat,
            q_scores,
            predicted_labels,
            selected_mask,
        ):
            selected_indices = selected.nonzero(as_tuple=False).flatten()
            image_seed_indices = []
            for label in labels_per_image[selected_indices].unique(sorted=True):
                class_indices = selected_indices[labels_per_image[selected_indices] == label]
                best_idx = class_indices[scores_per_image[class_indices].argmax()]
                image_seed_indices.append(best_idx)

            image_seed_indices = torch.stack(image_seed_indices)
            order = scores_per_image[image_seed_indices].argsort(descending=True)
            image_seed_indices = image_seed_indices[order]
            image_seed_indices = image_seed_indices[:64]
            image_seed_signatures = signatures_per_image[image_seed_indices]
            image_seed_scores = scores_per_image[image_seed_indices]
            batch_seed_signatures.append(image_seed_signatures)
            batch_seed_scores.append(image_seed_scores)
            max_num_seeds = max(max_num_seeds, image_seed_signatures.shape[0])

        batch_size = q_sig_flat.shape[0]
        sig_dim = q_sig_flat.shape[-1]
        seed_signatures = q_sig_flat.new_zeros((batch_size, max_num_seeds, sig_dim))
        seed_scores = q_scores.new_zeros((batch_size, max_num_seeds))
        pad_mask = torch.zeros(
            (batch_size, max_num_seeds),
            dtype=torch.bool,
            device=q_sig_flat.device,
        )
        for batch_idx, (image_seed_signatures, image_seed_scores) in enumerate(
            zip(batch_seed_signatures, batch_seed_scores)
        ):
            num_seeds = image_seed_signatures.shape[0]
            seed_signatures[batch_idx, :num_seeds] = image_seed_signatures
            seed_scores[batch_idx, :num_seeds] = image_seed_scores
            pad_mask[batch_idx, :num_seeds] = True

        return seed_signatures, pad_mask, seed_scores

    def _inference_with_seeds(
        self,
        outputs,
        batched_inputs,
        image_sizes,
        padded_image_size,
        seed_signatures,
        seed_pad_mask,
        seed_scores,
        class_seed_signatures,
        class_seed_pad_mask,
        class_seed_labels,
        class_seed_scores,
        targets=None,
    ):
        object_masks, semantic_masks, _, _ = self.aggregator(
            outputs["pred_masks"],
            outputs["pred_mask_signatures"],
            outputs["pred_class_signatures"],
            outputs["pred_mask_seed_logits"],
            outputs["pred_class_seed_logits"],
            seed_signatures,
            class_signatures=class_seed_signatures,
            mask_features=outputs["mask_features"],
            target_pad_mask=seed_pad_mask,
            class_pad_mask=class_seed_pad_mask,
        )

        mask_pred_results = max(object_masks, key=lambda masks: masks.shape[-2] * masks.shape[-1])
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=padded_image_size,
            mode="bilinear",
            align_corners=False,
        )
        semantic_pred_results = max(
            semantic_masks,
            key=lambda masks: masks.shape[-2] * masks.shape[-1],
        )
        semantic_pred_results = F.interpolate(
            semantic_pred_results,
            size=padded_image_size,
            mode="bilinear",
            align_corners=False,
        )

        mask_pred_results, output_sizes = self._prepare_batched_mask_predictions(
            mask_pred_results,
            batched_inputs,
            image_sizes,
        )
        semantic_pred_results, _ = self._prepare_batched_mask_predictions(
            semantic_pred_results,
            batched_inputs,
            image_sizes,
        )
        semantic_pred_results = semantic_pred_results.to(mask_pred_results)
        spatial_valid_mask = self._spatial_valid_mask(output_sizes, mask_pred_results)
        semantic_prob_results = retry_if_cuda_oom(self.batched_semantic_inference)(
            semantic_pred_results,
            class_seed_pad_mask,
            class_seed_labels,
            class_seed_scores,
        )
        object_class_scores = self.batched_object_class_scores(
            semantic_prob_results,
            mask_pred_results,
            seed_pad_mask,
            spatial_valid_mask,
        )

        processed_results = [{} for _ in batched_inputs]
        sem_seg_results = None
        panoptic_results = None
        instance_results = None
        if self.semantic_on:
            sem_seg_results = semantic_prob_results
            sem_seg_results = sem_seg_results * spatial_valid_mask[:, None].to(dtype=sem_seg_results.dtype)
        if self.panoptic_on:
            panoptic_results = retry_if_cuda_oom(self.batched_panoptic_inference)(
                object_class_scores,
                mask_pred_results,
                seed_pad_mask,
                spatial_valid_mask,
            )
        if self.instance_on:
            instance_results = retry_if_cuda_oom(self.batched_instance_inference)(
                object_class_scores,
                mask_pred_results,
                seed_pad_mask,
                spatial_valid_mask,
            )

        if self.signature_on and targets is not None and "gt_mask_signatures" in outputs:
            query_signatures = LatentAggregator._flatten_layer_queries(
                outputs["pred_mask_signatures"],
                "pred_mask_signatures",
            )
            query_seed_scores = self._flatten_layer_logits(
                outputs["pred_mask_seed_logits"],
                "pred_mask_seed_logits",
            ).sigmoid()
        else:
            query_signatures = None
            query_seed_scores = None

        for idx, (input_per_image, image_size, output_size) in enumerate(
            zip(batched_inputs, image_sizes, output_sizes)
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            out_h, out_w = output_size

            if sem_seg_results is not None:
                sem_seg = sem_seg_results[idx, :, :out_h, :out_w]
                if not self.sem_seg_postprocess_before_inference:
                    sem_seg = retry_if_cuda_oom(sem_seg_postprocess)(sem_seg, image_size, height, width)
                processed_results[idx]["sem_seg"] = sem_seg

            if panoptic_results is not None:
                panoptic_seg, segments_info = panoptic_results[idx]
                processed_results[idx]["panoptic_seg"] = (
                    panoptic_seg[:out_h, :out_w],
                    segments_info,
                )

            if instance_results is not None:
                instances = instance_results[idx]
                cropped_instances = Instances((out_h, out_w))
                cropped_instances.pred_masks = instances.pred_masks[:, :out_h, :out_w]
                cropped_instances.pred_boxes = instances.pred_boxes
                cropped_instances.scores = instances.scores
                cropped_instances.pred_classes = instances.pred_classes
                processed_results[idx]["instances"] = cropped_instances

            if self.signature_on and targets is not None and "gt_mask_signatures" in outputs:
                gt_pad_mask = targets["pad_mask"][idx]
                processed_results[idx]["latentformer_signature_eval"] = {
                    "gt_signatures": outputs["gt_mask_signatures"][idx, gt_pad_mask].detach().cpu(),
                    "det_signatures": query_signatures[idx].detach().cpu(),
                    "det_seed_scores": query_seed_scores[idx].detach().cpu(),
                    "selected_seed_signatures": seed_signatures[idx, seed_pad_mask[idx]].detach().cpu(),
                    "selected_seed_scores": seed_scores[idx, seed_pad_mask[idx]].detach().cpu(),
                }

        return processed_results

    @staticmethod
    def _flatten_layer_logits(values: torch.Tensor, name: str) -> torch.Tensor:
        if values.dim() == 2:
            return values
        if values.dim() == 3:
            layers, batch, queries = values.shape
            return values.permute(1, 0, 2).reshape(batch, layers * queries)
        raise ValueError(f"{name} must be shaped [B,Q] or [L,B,Q], got {values.shape}.")

    def _prepare_batched_mask_predictions(self, mask_pred_results, batched_inputs, image_sizes):
        if not self.sem_seg_postprocess_before_inference:
            return mask_pred_results, list(image_sizes)

        output_sizes = [
            (
                input_per_image.get("height", image_size[0]),
                input_per_image.get("width", image_size[1]),
            )
            for input_per_image, image_size in zip(batched_inputs, image_sizes)
        ]
        max_h = max(height for height, _ in output_sizes)
        max_w = max(width for _, width in output_sizes)
        batched_masks = mask_pred_results.new_zeros(
            (mask_pred_results.shape[0], mask_pred_results.shape[1], max_h, max_w)
        )

        for idx, (mask_pred_result, image_size, output_size) in enumerate(
            zip(mask_pred_results, image_sizes, output_sizes)
        ):
            resized = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result,
                image_size,
                output_size[0],
                output_size[1],
            )
            batched_masks[idx, :, : output_size[0], : output_size[1]] = resized

        return batched_masks, output_sizes

    def _spatial_valid_mask(self, output_sizes, mask_pred_results):
        mask = torch.zeros(
            (mask_pred_results.shape[0], mask_pred_results.shape[-2], mask_pred_results.shape[-1]),
            dtype=torch.bool,
            device=mask_pred_results.device,
        )
        for idx, (height, width) in enumerate(output_sizes):
            mask[idx, :height, :width] = True
        return mask

    def _masked_seed_softmax(self, mask_pred, seed_pad_mask):
        mask_logits = mask_pred.masked_fill(~seed_pad_mask[:, :, None, None], torch.finfo(mask_pred.dtype).min)
        return F.softmax(mask_logits, dim=1)

    def class_labels_from_signatures(self, class_seed_signatures, class_seed_pad_mask, all_class_signatures):
        all_class_signatures = all_class_signatures.to(
            device=class_seed_signatures.device,
            dtype=class_seed_signatures.dtype,
        )
        all_class_signatures = all_class_signatures.unsqueeze(0).expand(
            class_seed_signatures.shape[0],
            -1,
            -1,
        )
        similarity = pairwise_similarity(
            class_seed_signatures,
            all_class_signatures,
            metric=self.aggregator.aggregation_similarity_metric,
        )
        labels = similarity.argmax(dim=-1)
        return labels.masked_fill(~class_seed_pad_mask, 0)

    def batched_semantic_inference(
        self,
        semantic_pred,
        class_seed_pad_mask,
        class_seed_labels,
        class_seed_scores,
    ):
        batch_size, _, height, width = semantic_pred.shape
        full_semantic = semantic_pred.new_zeros(
            (batch_size, self.sem_seg_head.num_classes, height, width)
        )
        if semantic_pred.shape[1] == 0:
            return full_semantic

        score_prior = class_seed_scores.to(dtype=semantic_pred.dtype).clamp_min(1e-6).log()
        semantic_logits = semantic_pred + score_prior[:, :, None, None]
        seed_probs = F.softmax(
            semantic_logits.masked_fill(
                ~class_seed_pad_mask[:, :, None, None],
                torch.finfo(semantic_pred.dtype).min,
            ),
            dim=1,
        )
        seed_probs = seed_probs * class_seed_pad_mask[:, :, None, None].to(dtype=seed_probs.dtype)
        scatter_labels = class_seed_labels[:, :, None, None].expand(-1, -1, height, width)
        full_semantic.scatter_add_(1, scatter_labels, seed_probs)
        return full_semantic

    def batched_object_class_scores(
        self,
        semantic_pred,
        mask_pred,
        seed_pad_mask,
        spatial_valid_mask,
    ):
        batch_size, num_seeds = mask_pred.shape[:2]
        if num_seeds == 0:
            return mask_pred.new_zeros((batch_size, 0, self.sem_seg_head.num_classes))
        class_probs = semantic_pred
        mask_probs = self._masked_seed_softmax(mask_pred, seed_pad_mask)
        mask_probs = mask_probs * spatial_valid_mask[:, None].to(dtype=mask_probs.dtype)
        denom = mask_probs.flatten(2).sum(dim=-1).clamp_min(1e-6)
        scores = torch.einsum("bchw,bqhw->bqc", class_probs, mask_probs)
        scores = scores / denom[:, :, None]
        return scores * seed_pad_mask[:, :, None].to(dtype=scores.dtype)

    def batched_panoptic_inference(self, class_scores, mask_pred, seed_pad_mask, spatial_valid_mask):
        batch_size, _, height, width = mask_pred.shape
        if class_scores.shape[1] == 0:
            empty = mask_pred.new_zeros((height, width), dtype=torch.int32)
            return [(empty.clone(), []) for _ in range(batch_size)]

        scores, labels = class_scores.max(-1)
        mask_probs = self._masked_seed_softmax(mask_pred, seed_pad_mask)
        keep = (
            seed_pad_mask
            & (scores > self.score_threshold)
        )
        prob_masks = scores[:, :, None, None] * mask_probs
        prob_masks = prob_masks.masked_fill(~seed_pad_mask[:, :, None, None], -1.0)
        prob_masks = prob_masks.masked_fill(~spatial_valid_mask[:, None], -1.0)
        mask_ids = prob_masks.argmax(dim=1)
        seed_ids = torch.arange(class_scores.shape[1], device=mask_pred.device)
        assigned_masks = (mask_ids[:, None] == seed_ids[None, :, None, None]) & spatial_valid_mask[:, None]
        final_masks = assigned_masks & (mask_probs >= 0.5)
        mask_areas = assigned_masks.flatten(2).sum(2)
        original_areas = ((mask_probs >= 0.5) & spatial_valid_mask[:, None]).flatten(2).sum(2)
        final_areas = final_masks.flatten(2).sum(2)

        results = []
        thing_ids = set(self.metadata.thing_dataset_id_to_contiguous_id.values())
        for batch_idx in range(batch_size):
            panoptic_seg = torch.zeros((height, width), dtype=torch.int32, device=mask_pred.device)
            segments_info = []
            current_segment_id = 0
            stuff_memory_list = {}

            kept_seed_indices = keep[batch_idx].nonzero(as_tuple=False).flatten()
            for seed_idx in kept_seed_indices:
                pred_class = labels[batch_idx, seed_idx].item()
                isthing = pred_class in thing_ids
                mask_area = mask_areas[batch_idx, seed_idx].item()
                original_area = original_areas[batch_idx, seed_idx].item()
                final_area = final_areas[batch_idx, seed_idx].item()
                mask = final_masks[batch_idx, seed_idx]

                if mask_area > 0 and original_area > 0 and final_area > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    if not isthing:
                        if int(pred_class) in stuff_memory_list:
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            results.append((panoptic_seg, segments_info))

        return results

    def batched_instance_inference(self, class_scores, mask_pred, seed_pad_mask, spatial_valid_mask):
        batch_size, num_seeds, height, width = mask_pred.shape
        image_size = (height, width)
        if num_seeds == 0:
            return [self._empty_instances(image_size, mask_pred) for _ in range(batch_size)]

        scores = class_scores
        scores = scores.masked_fill(~seed_pad_mask[:, :, None], -1.0)
        k = min(self.test_topk_per_image, scores.shape[1] * scores.shape[2])
        scores_per_image, topk_indices = scores.flatten(1, 2).topk(k, dim=1, sorted=False)
        labels_per_image = topk_indices % self.sem_seg_head.num_classes
        seed_indices = topk_indices // self.sem_seg_head.num_classes
        mask_probs = self._masked_seed_softmax(mask_pred, seed_pad_mask)
        selected_masks = mask_probs.gather(
            1,
            seed_indices[:, :, None, None].expand(-1, -1, height, width),
        )
        selected_masks = selected_masks * spatial_valid_mask[:, None].to(dtype=selected_masks.dtype)
        pred_masks = ((selected_masks >= 0.5) & spatial_valid_mask[:, None]).float()
        mask_scores_per_image = (
            (selected_masks * pred_masks).flatten(2).sum(2)
            / (pred_masks.flatten(2).sum(2) + 1e-6)
        )
        scores_per_image = scores_per_image * mask_scores_per_image

        thing_class_mask = torch.zeros(self.sem_seg_head.num_classes, dtype=torch.bool, device=mask_pred.device)
        thing_class_mask[
            torch.tensor(
                list(self.metadata.thing_dataset_id_to_contiguous_id.values()),
                dtype=torch.long,
                device=mask_pred.device,
            )
        ] = True
        results = []
        for batch_idx in range(batch_size):
            keep = scores_per_image[batch_idx] > self.score_threshold
            if self.panoptic_on:
                keep = keep & thing_class_mask[labels_per_image[batch_idx]]

            result = Instances(image_size)
            result.pred_masks = pred_masks[batch_idx, keep]
            result.pred_boxes = Boxes(
                torch.zeros((result.pred_masks.shape[0], 4), dtype=mask_pred.dtype, device=mask_pred.device)
            )
            result.scores = scores_per_image[batch_idx, keep]
            result.pred_classes = labels_per_image[batch_idx, keep]
            results.append(result)

        return results

    def _empty_instances(self, image_size, mask_pred):
        result = Instances(image_size)
        result.pred_masks = torch.zeros((0, *image_size), dtype=mask_pred.dtype, device=mask_pred.device)
        result.pred_boxes = Boxes(torch.zeros((0, 4), dtype=mask_pred.dtype, device=mask_pred.device))
        result.scores = torch.zeros((0,), dtype=mask_pred.dtype, device=mask_pred.device)
        result.pred_classes = torch.zeros((0,), dtype=torch.long, device=mask_pred.device)
        return result

    def prepare_gt_encoder_inputs(self, targets, images, batched_inputs=None):
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

        result = {"masks": masks, "labels": labels, "boxes": boxes, "pad_mask": pad_mask}
        semantic_targets = self.prepare_semantic_targets(batched_inputs, targets, images)
        if semantic_targets is not None:
            result.update(semantic_targets)
        return result

    def prepare_semantic_targets(self, batched_inputs, targets, images):
        if batched_inputs is None:
            return None

        h_pad, w_pad = images.tensor.shape[-2:]
        per_image_masks = []
        per_image_labels = []
        has_semantic_gt = False

        for idx, input_per_image in enumerate(batched_inputs):
            class_to_mask = {}
            if "sem_seg" in input_per_image:
                sem_seg = input_per_image["sem_seg"].to(device=self.device)
                sem_h = min(sem_seg.shape[-2], h_pad)
                sem_w = min(sem_seg.shape[-1], w_pad)
                sem_seg = sem_seg[:sem_h, :sem_w]
                valid = (sem_seg >= 0) & (sem_seg < self.sem_seg_head.num_classes)
                if valid.any():
                    for class_id in sem_seg[valid].unique().long().tolist():
                        mask = torch.zeros(
                            (h_pad, w_pad),
                            dtype=images.tensor.dtype,
                            device=self.device,
                        )
                        mask[:sem_h, :sem_w] = (sem_seg == class_id).to(dtype=mask.dtype)
                        class_to_mask[int(class_id)] = mask
                    has_semantic_gt = True
            else:
                targets_per_image = targets[idx]
                if len(targets_per_image.gt_classes) > 0:
                    gt_masks = targets_per_image.gt_masks
                    if hasattr(gt_masks, "tensor"):
                        gt_masks = gt_masks.tensor
                    gt_masks = gt_masks.to(device=self.device, dtype=images.tensor.dtype)
                    for class_id, mask in zip(targets_per_image.gt_classes.tolist(), gt_masks):
                        if 0 <= class_id < self.sem_seg_head.num_classes:
                            class_mask = class_to_mask.setdefault(
                                int(class_id),
                                torch.zeros(
                                    (h_pad, w_pad),
                                    dtype=images.tensor.dtype,
                                    device=self.device,
                                ),
                            )
                            class_mask[: mask.shape[-2], : mask.shape[-1]] = torch.maximum(
                                class_mask[: mask.shape[-2], : mask.shape[-1]],
                                mask,
                            )
                            has_semantic_gt = True

            labels = sorted(class_to_mask)
            if labels:
                per_image_labels.append(torch.tensor(labels, dtype=torch.long, device=self.device))
                per_image_masks.append(torch.stack([class_to_mask[class_id] for class_id in labels]))
            else:
                per_image_labels.append(torch.zeros((0,), dtype=torch.long, device=self.device))
                per_image_masks.append(torch.zeros((0, h_pad, w_pad), dtype=images.tensor.dtype, device=self.device))

        if not has_semantic_gt:
            return None

        max_classes = max((labels.numel() for labels in per_image_labels), default=0)
        semantic_masks = torch.zeros(
            (len(batched_inputs), max_classes, h_pad, w_pad),
            dtype=images.tensor.dtype,
            device=self.device,
        )
        semantic_labels = torch.zeros(
            (len(batched_inputs), max_classes),
            dtype=torch.long,
            device=self.device,
        )
        semantic_pad_mask = torch.zeros(
            (len(batched_inputs), max_classes),
            dtype=torch.bool,
            device=self.device,
        )
        for idx, (labels, masks) in enumerate(zip(per_image_labels, per_image_masks)):
            num_classes = labels.numel()
            if num_classes == 0:
                continue
            semantic_labels[idx, :num_classes] = labels
            semantic_masks[idx, :num_classes] = masks
            semantic_pad_mask[idx, :num_classes] = True

        return {
            "semantic_masks": semantic_masks,
            "semantic_labels": semantic_labels,
            "semantic_pad_mask": semantic_pad_mask,
        }
