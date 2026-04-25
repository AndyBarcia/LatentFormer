# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from collections import OrderedDict, defaultdict

import torch
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator

from mask2former.modeling.similarity import pairwise_distance


def _signature_bucket_name(num_gt):
    if num_gt == 0:
        return "gt_000"
    bucket_start = ((num_gt - 1) // 10) * 10 + 1
    bucket_end = bucket_start + 9
    return f"gt_{bucket_start:03d}_{bucket_end:03d}"


class LatentFormerSignatureEvaluator(DatasetEvaluator):
    def __init__(self, similarity_metric, output_dir=None):
        self.similarity_metric = similarity_metric
        self._output_dir = output_dir
        self._logger = logging.getLogger(__name__)
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
            det_seed_scores = diagnostics.get("det_seed_scores")
            if det_seed_scores is not None:
                det_seed_scores = det_seed_scores.float()
            num_gt = int(gt_signatures.shape[0])
            bucket = _signature_bucket_name(num_gt)
            self._add_image_stats("all", gt_signatures, det_signatures, det_seed_scores)
            self._add_image_stats(bucket, gt_signatures, det_signatures, det_seed_scores)

    def _add_image_stats(self, bucket, gt_signatures, det_signatures, det_seed_scores=None):
        num_gt = int(gt_signatures.shape[0])
        num_det = int(det_signatures.shape[0])

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
            self._stats[bucket]["num_gt"].append(num_gt)
            self._stats[bucket]["num_det"].append(num_det)
            self._stats[bucket]["num_matched_det"].append(0)
            self._stats[bucket]["num_unmatched_det"].append(num_det)
            self._stats[bucket]["num_missed_gt"].append(num_gt)
            if det_seed_scores is not None and num_det > 0:
                self._stats[bucket]["unmatched_det_seed_score"].extend(det_seed_scores.tolist())
            return

        det_gt_dist = pairwise_distance(
            det_signatures,
            gt_signatures,
            metric=self.similarity_metric,
        )
        cost = det_gt_dist.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_det = set(row_ind.tolist())
        matched_gt = set(col_ind.tolist())
        unmatched_det = [
            det_idx for det_idx in range(det_signatures.shape[0]) if det_idx not in matched_det
        ]
        missed_gt = [
            gt_idx for gt_idx in range(gt_signatures.shape[0]) if gt_idx not in matched_gt
        ]

        self._stats[bucket]["num_gt"].append(num_gt)
        self._stats[bucket]["num_det"].append(num_det)
        self._stats[bucket]["num_matched_det"].append(len(matched_det))
        self._stats[bucket]["num_unmatched_det"].append(len(unmatched_det))
        self._stats[bucket]["num_missed_gt"].append(len(missed_gt))
        if len(row_ind) > 0:
            self._stats[bucket]["matched_det_gt"].extend(
                cost[row_ind, col_ind].astype(float).tolist()
            )
            if det_seed_scores is not None:
                matched_scores = det_seed_scores[row_ind]
                self._stats[bucket]["matched_det_seed_score"].extend(matched_scores.tolist())

        if unmatched_det:
            nearest_unmatched = det_gt_dist[unmatched_det].min(dim=1).values
            self._stats[bucket]["unmatched_det_gt"].extend(nearest_unmatched.tolist())
            if det_seed_scores is not None:
                self._stats[bucket]["unmatched_det_seed_score"].extend(
                    det_seed_scores[unmatched_det].tolist()
                )

        if missed_gt:
            nearest_missed = det_gt_dist[:, missed_gt].min(dim=0).values
            self._stats[bucket]["missed_gt_det"].extend(nearest_missed.tolist())

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
        table_rows = []
        for bucket in sorted(merged.keys()):
            for metric_name in (
                "num_gt",
                "num_det",
                "num_matched_det",
                "num_unmatched_det",
                "num_missed_gt",
                "gt_gt",
                "matched_det_gt",
                "unmatched_det_gt",
                "missed_gt_det",
                "matched_det_seed_score",
                "unmatched_det_seed_score",
            ):
                values = torch.tensor(merged[bucket].get(metric_name, []), dtype=torch.float32)
                prefix = f"{bucket}/{metric_name}"
                results[f"{prefix}_count"] = int(values.numel())
                if values.numel() == 0:
                    avg = dev = min_value = p50 = max_value = float("nan")
                else:
                    avg = float(values.mean().item())
                    dev = float(values.std(unbiased=False).item())
                    min_value = float(values.min().item())
                    p50 = float(values.quantile(0.5).item())
                    max_value = float(values.max().item())

                results[f"{prefix}_avg"] = avg
                results[f"{prefix}_dev"] = dev
                results[f"{prefix}_min"] = min_value
                results[f"{prefix}_p50"] = p50
                results[f"{prefix}_max"] = max_value
                table_rows.append(
                    [
                        bucket,
                        metric_name,
                        int(values.numel()),
                        avg,
                        dev,
                        min_value,
                        p50,
                        max_value,
                    ]
                )

        if table_rows:
            headers = ["Bucket", "Metric", "Count", "Avg", "Dev", "Min", "P50", "Max"]
            table = tabulate(
                table_rows,
                headers=headers,
                tablefmt="pipe",
                floatfmt=".3f",
                stralign="center",
                numalign="center",
            )
            self._logger.info("LatentFormer Signature Evaluation Results:\n" + table)

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            with open(os.path.join(self._output_dir, "latentformer_signature_stats.json"), "w") as handle:
                json.dump(results, handle, indent=2)

        return {"latentformer_signature": results}
