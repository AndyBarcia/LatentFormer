# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .similarity import pairwise_similarity


class LatentMatcher(nn.Module):
    """Hungarian matcher for LatentFormer seed/signature assignments."""

    def __init__(self, similarity_metric: str = "dot", seed_cost_weight: float = 1.0):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.seed_cost_weight = seed_cost_weight

    @torch.no_grad()
    def forward(
        self,
        q_sig_flat,
        gt_sigs_norm,
        gt_pad_mask,
        q_seed_logits_flat=None,
    ):
        B, num_queries, _ = q_sig_flat.shape
        matched_query_mask = torch.zeros((B, num_queries), dtype=torch.bool, device=q_sig_flat.device)
        matched_gt_indices = torch.full((B, num_queries), -1, dtype=torch.long, device=q_sig_flat.device)

        for b in range(B):
            valid_gt_idx = torch.where(gt_pad_mask[b])[0]
            if valid_gt_idx.numel() == 0 or num_queries == 0:
                continue

            sim = pairwise_similarity(
                q_sig_flat[b],
                gt_sigs_norm[b, valid_gt_idx],
                metric=self.similarity_metric,
            )
            cost = 1.0 - sim
            if q_seed_logits_flat is not None and self.seed_cost_weight != 0.0:
                seed_cost = 1.0 - torch.sigmoid(q_seed_logits_flat[b])
                cost = cost + self.seed_cost_weight * seed_cost.unsqueeze(-1)
            cost = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            if len(row_ind) == 0:
                continue

            row_ind_t = torch.as_tensor(row_ind, device=q_sig_flat.device, dtype=torch.long)
            col_ind_t = valid_gt_idx[
                torch.as_tensor(col_ind, device=q_sig_flat.device, dtype=torch.long)
            ]
            matched_query_mask[b, row_ind_t] = True
            matched_gt_indices[b, row_ind_t] = col_ind_t

        return matched_query_mask, matched_gt_indices

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "similarity_metric: {}".format(self.similarity_metric),
            "seed_cost_weight: {}".format(self.seed_cost_weight),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
