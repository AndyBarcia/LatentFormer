# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from collections import OrderedDict, defaultdict

import torch
from scipy.optimize import linear_sum_assignment

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator

from mask2former.modeling.similarity import pairwise_distance


class LatentFormerSignatureEvaluator(DatasetEvaluator):
    def __init__(self, similarity_metric, output_dir=None):
        self.similarity_metric = similarity_metric
        self._output_dir = output_dir
        self.reset()

    def reset(self):
        self._stats = defaultdict(lambda: defaultdict(list))

    def process(self, inputs, outputs):
        del inputs
        for output in outputs:
            diagnostics = output.get("latentformer_signature_eval")
            if diagnostics is None:
                continue
            gt_signatures = diagnostics["gt_signatures"].float()
            det_signatures = diagnostics["det_signatures"].float()
            num_gt = int(gt_signatures.shape[0])
            bucket = f"gt_{num_gt:03d}"
            self._add_image_stats("all", gt_signatures, det_signatures)
            self._add_image_stats(bucket, gt_signatures, det_signatures)

    def _add_image_stats(self, bucket, gt_signatures, det_signatures):
        if gt_signatures.shape[0] >= 2:
            gt_dist = pairwise_distance(
                gt_signatures,
                gt_signatures,
                metric=self.similarity_metric,
            )
            rows, cols = torch.triu_indices(
                gt_dist.shape[0],
                gt_dist.shape[1],
                offset=1,
                device=gt_dist.device,
            )
            self._stats[bucket]["gt_gt"].extend(gt_dist[rows, cols].tolist())

        if gt_signatures.shape[0] == 0 or det_signatures.shape[0] == 0:
            return

        det_gt_dist = pairwise_distance(
            det_signatures,
            gt_signatures,
            metric=self.similarity_metric,
        )
        cost = det_gt_dist.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_det = set(row_ind.tolist())
        if len(row_ind) > 0:
            self._stats[bucket]["matched_det_gt"].extend(
                cost[row_ind, col_ind].astype(float).tolist()
            )

        unmatched_det = [
            det_idx for det_idx in range(det_signatures.shape[0]) if det_idx not in matched_det
        ]
        if unmatched_det:
            nearest_unmatched = det_gt_dist[unmatched_det].min(dim=1).values
            self._stats[bucket]["unmatched_det_gt"].extend(nearest_unmatched.tolist())

    def evaluate(self):
        local_stats = {
            bucket: dict(metric_values)
            for bucket, metric_values in self._stats.items()
        }
        all_stats = comm.gather(local_stats, dst=0)
        if not comm.is_main_process():
            return {}

        merged = defaultdict(lambda: defaultdict(list))
        for worker_stats in all_stats:
            for bucket, metric_values in worker_stats.items():
                for metric_name, values in metric_values.items():
                    merged[bucket][metric_name].extend(values)

        results = OrderedDict()
        for bucket in sorted(merged.keys()):
            for metric_name in ("gt_gt", "matched_det_gt", "unmatched_det_gt"):
                values = torch.tensor(merged[bucket].get(metric_name, []), dtype=torch.float32)
                prefix = f"{bucket}/{metric_name}"
                results[f"{prefix}_count"] = int(values.numel())
                if values.numel() == 0:
                    results[f"{prefix}_avg"] = float("nan")
                    results[f"{prefix}_dev"] = float("nan")
                else:
                    results[f"{prefix}_avg"] = float(values.mean().item())
                    results[f"{prefix}_dev"] = float(values.std(unbiased=False).item())

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            with open(os.path.join(self._output_dir, "latentformer_signature_stats.json"), "w") as handle:
                json.dump(results, handle, indent=2)

        return {"latentformer_signature": results}
