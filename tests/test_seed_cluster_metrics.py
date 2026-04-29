import torch

from mask2former.modeling.seed_cluster_metrics import compute_seed_cluster_precision_recall
from mask2former.modeling.seed_selection import ThresholdPrecisionRecallMLP


def test_seed_cluster_metrics_counts_duplicate_and_unmatched_components():
    query_signatures = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.0, 0.8],
            ]
        ]
    )
    query_seed_logits = torch.tensor([[10.0, 10.0, -10.0, 10.0]])
    matched_query_mask = torch.tensor([[True, True, True, False]])
    matched_gt_indices = torch.tensor([[0, 1, 2, -1]])

    metrics = compute_seed_cluster_precision_recall(
        query_signatures=query_signatures,
        query_seed_logits=query_seed_logits,
        matched_query_mask=matched_query_mask,
        matched_gt_indices=matched_gt_indices,
        seed_threshold=0.5,
        duplicate_threshold=0.8,
        metric="cosine",
    )

    assert metrics["tp"].item() == 1.0
    assert metrics["fp"].item() == 1.0
    assert metrics["fn"].item() == 2.0
    assert torch.isclose(metrics["precision"], torch.tensor([[[0.5]]])).all()
    assert torch.isclose(metrics["recall"], torch.tensor([[[1.0 / 3.0]]])).all()


def test_seed_cluster_metrics_sweeps_thresholds():
    query_signatures = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ]
    )
    query_seed_logits = torch.tensor([[10.0, -10.0]])
    matched_query_mask = torch.tensor([[True, True]])
    matched_gt_indices = torch.tensor([[0, 1]])

    metrics = compute_seed_cluster_precision_recall(
        query_signatures=query_signatures,
        query_seed_logits=query_seed_logits,
        matched_query_mask=matched_query_mask,
        matched_gt_indices=matched_gt_indices,
        seed_threshold=torch.tensor([0.5, 1.0]),
        duplicate_threshold=torch.tensor([0.0, 0.5]),
        metric="cosine",
    )

    assert metrics["tp"].shape == (1, 2, 2)
    assert metrics["tp"][0, 0].tolist() == [1.0, 1.0]
    assert metrics["fp"][0, 0].tolist() == [0.0, 0.0]
    assert metrics["fn"][0, 0].tolist() == [1.0, 1.0]
    assert metrics["tp"][0, 1].tolist() == [0.0, 0.0]
    assert metrics["fn"][0, 1].tolist() == [2.0, 2.0]


def test_seed_cluster_metrics_aggregates_batch_micro_scores():
    query_signatures = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    query_seed_logits = torch.tensor([[10.0, -10.0], [10.0, 10.0]])
    matched_query_mask = torch.tensor([[True, False], [False, True]])
    matched_gt_indices = torch.tensor([[0, -1], [-1, 0]])

    metrics = compute_seed_cluster_precision_recall(
        query_signatures=query_signatures,
        query_seed_logits=query_seed_logits,
        matched_query_mask=matched_query_mask,
        matched_gt_indices=matched_gt_indices,
        seed_threshold=0.5,
        duplicate_threshold=0.5,
        metric="cosine",
    )

    assert metrics["total_tp"].item() == 2.0
    assert metrics["total_fp"].item() == 1.0
    assert metrics["total_fn"].item() == 0.0
    assert torch.isclose(metrics["micro_precision"], torch.tensor([[2.0 / 3.0]])).all()
    assert torch.isclose(metrics["micro_recall"], torch.tensor([[1.0]])).all()


def test_threshold_precision_recall_mlp_predicts_per_threshold_pair():
    predictor = ThresholdPrecisionRecallMLP(hidden_dim=8)

    pred = predictor(
        torch.tensor([0.3, 0.5]),
        torch.tensor([0.6, 0.8, 0.9]),
    )

    assert pred.shape == (2, 3, 2)
    assert torch.all(pred >= 0.0)
    assert torch.all(pred <= 1.0)
