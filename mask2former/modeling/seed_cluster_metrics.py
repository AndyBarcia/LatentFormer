# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .seed_selection_ops.functions import seed_cluster_precision_recall_native
from .similarity import pairwise_similarity


_NATIVE_SUPPORTED_METRICS = {
    "centered-cosine",
    "cosine",
    "dot",
    "dot-sigmoid",
    "softmax",
}


def _as_1d_threshold_tensor(
    value: float | Sequence[float] | torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        threshold = value.to(device=device, dtype=torch.float32)
    else:
        threshold = torch.as_tensor(value, device=device, dtype=torch.float32)
    if threshold.dim() == 0:
        threshold = threshold.unsqueeze(0)
    if threshold.dim() != 1:
        raise ValueError(f"{name} must be a scalar or 1D sequence, got shape {tuple(threshold.shape)}.")
    return threshold


def _connected_component_stats(
    adjacency: torch.Tensor,
    selected: torch.Tensor,
    matched: torch.Tensor,
) -> tuple[int, int]:
    visited = torch.zeros_like(selected)
    tp = 0
    fp = 0

    for start in selected.nonzero(as_tuple=False).flatten().tolist():
        if visited[start]:
            continue

        has_match = False
        stack = [start]
        visited[start] = True

        while stack:
            current = stack.pop()
            has_match = has_match or bool(matched[current])
            neighbors = adjacency[current].nonzero(as_tuple=False).flatten().tolist()
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        if has_match:
            tp += 1
        else:
            fp += 1

    return tp, fp


@torch.no_grad()
def compute_seed_cluster_precision_recall(
    query_signatures: torch.Tensor,
    query_seed_logits: torch.Tensor,
    matched_query_mask: torch.Tensor,
    matched_gt_indices: torch.Tensor,
    seed_threshold: float | Sequence[float] | torch.Tensor,
    duplicate_threshold: float | Sequence[float] | torch.Tensor,
    metric: str,
) -> dict[str, torch.Tensor]:
    """Approximate detection precision/recall from the seed clustering graph.

    Queries above ``seed_threshold`` are graph vertices. Edges connect selected
    queries whose signature similarity is at least ``duplicate_threshold``.
    A connected component contributes one TP if it contains any matched query,
    otherwise one FP. Matched queries not represented by that one TP are FNs,
    so merged matched queries in the same component count as duplicates.
    """
    if query_signatures.dim() != 3:
        raise ValueError(f"query_signatures must be shaped [B,Q,C], got {tuple(query_signatures.shape)}.")
    if query_seed_logits.dim() != 2:
        raise ValueError(f"query_seed_logits must be shaped [B,Q], got {tuple(query_seed_logits.shape)}.")
    if matched_query_mask.shape != query_seed_logits.shape:
        raise ValueError(
            "matched_query_mask must match query_seed_logits shape, "
            f"got {tuple(matched_query_mask.shape)} and {tuple(query_seed_logits.shape)}."
        )
    if matched_gt_indices.shape != query_seed_logits.shape:
        raise ValueError(
            "matched_gt_indices must match query_seed_logits shape, "
            f"got {tuple(matched_gt_indices.shape)} and {tuple(query_seed_logits.shape)}."
        )
    if query_signatures.shape[:2] != query_seed_logits.shape:
        raise ValueError(
            "query_signatures and query_seed_logits must agree on [B,Q], "
            f"got {tuple(query_signatures.shape[:2])} and {tuple(query_seed_logits.shape)}."
        )

    device = query_signatures.device
    seed_thresholds = _as_1d_threshold_tensor(seed_threshold, name="seed_threshold", device=device)
    duplicate_thresholds = _as_1d_threshold_tensor(
        duplicate_threshold,
        name="duplicate_threshold",
        device=device,
    )

    matched_query_mask = matched_query_mask.to(device=device, dtype=torch.bool)
    matched_gt_indices = matched_gt_indices.to(device=device)
    if (matched_query_mask & (matched_gt_indices < 0)).any():
        raise ValueError("matched_gt_indices must be non-negative where matched_query_mask is true.")

    metric_name = metric.lower()
    if metric_name in _NATIVE_SUPPORTED_METRICS:
        native_result = seed_cluster_precision_recall_native(
            query_signatures,
            query_seed_logits,
            matched_query_mask,
            matched_gt_indices,
            seed_thresholds=seed_thresholds,
            duplicate_thresholds=duplicate_thresholds,
            similarity_metric=metric_name,
        )
        if native_result is not None:
            native_result["seed_thresholds"] = seed_thresholds
            native_result["duplicate_thresholds"] = duplicate_thresholds
            return native_result

    batch_size = query_signatures.shape[0]
    num_seed_thresholds = int(seed_thresholds.numel())
    num_duplicate_thresholds = int(duplicate_thresholds.numel())
    out_shape = (batch_size, num_seed_thresholds, num_duplicate_thresholds)

    tp = torch.zeros(out_shape, device=device, dtype=torch.float32)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)

    seed_scores = query_seed_logits.float().sigmoid()
    similarity = pairwise_similarity(
        query_signatures,
        query_signatures,
        metric=metric,
    )
    query_indices = torch.arange(query_signatures.shape[1], device=device)
    total_matched = matched_query_mask.sum(dim=1).to(dtype=torch.float32)

    for seed_idx, seed_thresh in enumerate(seed_thresholds):
        selected_mask = seed_scores >= seed_thresh

        for duplicate_idx, duplicate_thresh in enumerate(duplicate_thresholds):
            adjacency = similarity >= duplicate_thresh
            adjacency = adjacency & selected_mask[:, :, None] & selected_mask[:, None, :]
            adjacency[:, query_indices, query_indices] = selected_mask

            for batch_idx in range(batch_size):
                image_tp, image_fp = _connected_component_stats(
                    adjacency[batch_idx],
                    selected_mask[batch_idx],
                    matched_query_mask[batch_idx],
                )
                tp[batch_idx, seed_idx, duplicate_idx] = image_tp
                fp[batch_idx, seed_idx, duplicate_idx] = image_fp
                fn[batch_idx, seed_idx, duplicate_idx] = total_matched[batch_idx] - image_tp

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    total_tp = tp.sum(dim=0)
    total_fp = fp.sum(dim=0)
    total_fn = fn.sum(dim=0)
    micro_precision = total_tp / (total_tp + total_fp).clamp_min(1.0)
    micro_recall = total_tp / (total_tp + total_fn).clamp_min(1.0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "seed_thresholds": seed_thresholds,
        "duplicate_thresholds": duplicate_thresholds,
    }


class SeedClusterPrecisionRecall(nn.Module):
    """Module wrapper for seed-cluster precision/recall sweeps."""

    def __init__(
        self,
        *,
        seed_threshold: float | Sequence[float] | torch.Tensor,
        duplicate_threshold: float | Sequence[float] | torch.Tensor,
        metric: str = "cosine",
    ):
        super().__init__()
        self.seed_threshold = seed_threshold
        self.duplicate_threshold = duplicate_threshold
        self.metric = metric

    def forward(
        self,
        query_signatures: torch.Tensor,
        query_seed_logits: torch.Tensor,
        matched_query_mask: torch.Tensor,
        matched_gt_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return compute_seed_cluster_precision_recall(
            query_signatures=query_signatures,
            query_seed_logits=query_seed_logits,
            matched_query_mask=matched_query_mask,
            matched_gt_indices=matched_gt_indices,
            seed_threshold=self.seed_threshold,
            duplicate_threshold=self.duplicate_threshold,
            metric=self.metric,
        )

