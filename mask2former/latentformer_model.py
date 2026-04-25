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
        )
        aggregator = LatentAggregator(
            aggregation_similarity_metric=cfg.MODEL.LATENT_FORMER.AGGREGATION_SIMILARITY_METRIC,
        )
        eval_modes = tuple(cfg.MODEL.LATENT_FORMER.TEST.EVAL_MODES)
        seed_selection_modules = build_seed_selection_modules(
            eval_modes=eval_modes,
            seed_threshold=cfg.MODEL.LATENT_FORMER.TEST.SEED_THRESHOLD,
            duplicate_threshold=cfg.MODEL.LATENT_FORMER.TEST.DUPLICATE_THRESHOLD,
            similarity_metric=cfg.MODEL.LATENT_FORMER.AGGREGATION_SIMILARITY_METRIC,
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

        if self._needs_gt_signatures():
            targets = self._prepare_gt_signatures(batched_inputs, images, outputs)
        else:
            targets = None

        if self.training:
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
        targets = self.prepare_gt_encoder_inputs(gt_instances, images)
        outputs["gt_signatures"] = self.gt_encoder(
            outputs["mask_features"],
            targets["masks"],
            targets["labels"],
            targets["boxes"],
            targets["pad_mask"],
        )
        return targets

    def inference(self, outputs, batched_inputs, image_sizes, padded_image_size, targets=None):
        mode_results = {}
        for mode, seed_selection in self.seed_selection_modules.items():
            gt_signatures = outputs.get("gt_signatures")
            gt_pad_mask = targets["pad_mask"] if targets is not None else None
            seed_signatures, seed_pad_mask, seed_scores = seed_selection(
                outputs["pred_signatures"],
                outputs["pred_seed_logits"],
                gt_signatures=gt_signatures,
                gt_pad_mask=gt_pad_mask,
            )
            mode_results[mode] = self._inference_with_seeds(
                outputs,
                batched_inputs,
                image_sizes,
                padded_image_size,
                seed_signatures,
                seed_pad_mask,
                seed_scores,
                targets=targets,
            )

        if len(mode_results) == 1:
            return next(iter(mode_results.values()))
        return mode_results

    def _inference_with_seeds(
        self,
        outputs,
        batched_inputs,
        image_sizes,
        padded_image_size,
        seed_signatures,
        seed_pad_mask,
        seed_scores,
        targets=None,
    ):
        proto_cls, proto_masks, _ = self.aggregator(
            outputs["pred_logits"],
            outputs["pred_masks"],
            outputs["pred_signatures"],
            seed_signatures,
            mask_features=outputs["mask_features"],
            target_pad_mask=seed_pad_mask,
        )

        mask_pred_results = max(proto_masks, key=lambda masks: masks.shape[-2] * masks.shape[-1])
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=padded_image_size,
            mode="bilinear",
            align_corners=False,
        )

        mask_pred_results, output_sizes = self._prepare_batched_mask_predictions(
            mask_pred_results,
            batched_inputs,
            image_sizes,
        )
        proto_cls = proto_cls.to(mask_pred_results)
        spatial_valid_mask = self._spatial_valid_mask(output_sizes, mask_pred_results)

        processed_results = [{} for _ in batched_inputs]
        sem_seg_results = None
        panoptic_results = None
        instance_results = None
        if self.semantic_on:
            sem_seg_results = retry_if_cuda_oom(self.batched_semantic_inference)(
                proto_cls,
                mask_pred_results,
                seed_pad_mask,
            )
            sem_seg_results = sem_seg_results * spatial_valid_mask[:, None].to(dtype=sem_seg_results.dtype)
        if self.panoptic_on:
            panoptic_results = retry_if_cuda_oom(self.batched_panoptic_inference)(
                proto_cls,
                mask_pred_results,
                seed_pad_mask,
                spatial_valid_mask,
            )
        if self.instance_on:
            instance_results = retry_if_cuda_oom(self.batched_instance_inference)(
                proto_cls,
                mask_pred_results,
                seed_pad_mask,
                spatial_valid_mask,
            )

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

            if self.signature_on and targets is not None and "gt_signatures" in outputs:
                gt_pad_mask = targets["pad_mask"][idx]
                gt_labels = targets["labels"][idx]
                object_gt_mask = gt_pad_mask & gt_labels.ne(self.gt_encoder.background_label)
                processed_results[idx]["latentformer_signature_eval"] = {
                    "gt_signatures": outputs["gt_signatures"][idx, object_gt_mask].detach().cpu(),
                    "det_signatures": seed_signatures[idx, seed_pad_mask[idx]].detach().cpu(),
                    "det_seed_scores": seed_scores[idx, seed_pad_mask[idx]].detach().cpu(),
                }

        return processed_results

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

    def batched_semantic_inference(self, mask_cls, mask_pred, seed_pad_mask):
        if mask_cls.shape[1] == 0:
            return mask_pred.new_zeros((mask_pred.shape[0], self.sem_seg_head.num_classes, *mask_pred.shape[-2:]))
        class_probs = F.softmax(mask_cls, dim=-1)[..., :-1]
        class_probs = class_probs * seed_pad_mask[:, :, None].to(dtype=class_probs.dtype)
        mask_probs = self._masked_seed_softmax(mask_pred, seed_pad_mask)
        return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

    def batched_panoptic_inference(self, mask_cls, mask_pred, seed_pad_mask, spatial_valid_mask):
        batch_size, _, height, width = mask_pred.shape
        if mask_cls.shape[1] == 0:
            empty = mask_pred.new_zeros((height, width), dtype=torch.int32)
            return [(empty.clone(), []) for _ in range(batch_size)]

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_probs = self._masked_seed_softmax(mask_pred, seed_pad_mask)
        keep = (
            seed_pad_mask
            & labels.ne(self.sem_seg_head.num_classes)
            & (scores > self.score_threshold)
        )
        prob_masks = scores[:, :, None, None] * mask_probs
        prob_masks = prob_masks.masked_fill(~keep[:, :, None, None], -1.0)
        prob_masks = prob_masks.masked_fill(~spatial_valid_mask[:, None], -1.0)
        mask_ids = prob_masks.argmax(dim=1)
        seed_ids = torch.arange(mask_cls.shape[1], device=mask_pred.device)
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

    def batched_instance_inference(self, mask_cls, mask_pred, seed_pad_mask, spatial_valid_mask):
        batch_size, num_seeds, height, width = mask_pred.shape
        image_size = (height, width)
        if num_seeds == 0:
            return [self._empty_instances(image_size, mask_pred) for _ in range(batch_size)]

        scores = F.softmax(mask_cls, dim=-1)[:, :, :-1]
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
