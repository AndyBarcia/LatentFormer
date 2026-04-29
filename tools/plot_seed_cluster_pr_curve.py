#!/usr/bin/env python3
"""Plot LatentFormer seed-cluster threshold precision/recall predictions."""

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_latentformer_config, add_maskformer2_config
from mask2former.modeling.seed_selection import ThresholdPrecisionRecallMLP


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a LatentFormer checkpoint, sweep seed/duplicate thresholds through "
            "the learned PR predictor, and plot precision-recall with the Pareto front."
        )
    )
    parser.add_argument("checkpoint", help="Path to a Detectron2 checkpoint .pth file.")
    parser.add_argument(
        "--config-file",
        default="",
        help="Config YAML. If omitted, the script tries to find one beside the checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output image path. Defaults to <checkpoint-dir>/seed_cluster_pr_curve.png.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Number of threshold values per axis. Defaults to config PLOT_NUM_POINTS or 200.",
    )
    parser.add_argument(
        "--seed-threshold-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Seed threshold sweep range. Defaults to config range or 0 1.",
    )
    parser.add_argument(
        "--duplicate-threshold-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Duplicate threshold sweep range. Defaults to config range or 0 1.",
    )
    parser.add_argument(
        "--top-rq",
        type=int,
        default=1,
        help="Number of highest-RQ threshold points to mark.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the tiny MLP forward pass. Defaults to cpu.",
    )
    return parser.parse_args()


def find_config_file(checkpoint_path):
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve().parent
    configs = sorted(checkpoint_dir.glob("*.yaml")) + sorted(checkpoint_dir.glob("*.yml"))
    return str(configs[0]) if configs else ""


def load_pr_config(config_file):
    if not config_file:
        return None
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_latentformer_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    return cfg.MODEL.LATENT_FORMER.SEED_CLUSTER_PR


def get_sweep_settings(args, pr_cfg):
    num_points = args.num_points
    if num_points is None:
        num_points = int(pr_cfg.PLOT_NUM_POINTS) if pr_cfg is not None else 200
    if num_points <= 1:
        raise ValueError("--num-points must be greater than 1")

    seed_range = args.seed_threshold_range
    if seed_range is None:
        seed_range = pr_cfg.SEED_THRESHOLD_RANGE if pr_cfg is not None else (0.0, 1.0)

    duplicate_range = args.duplicate_threshold_range
    if duplicate_range is None:
        duplicate_range = (
            pr_cfg.DUPLICATE_THRESHOLD_RANGE if pr_cfg is not None else (0.0, 1.0)
        )

    return int(num_points), tuple(float(v) for v in seed_range), tuple(float(v) for v in duplicate_range)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def extract_pr_mlp_state(state_dict):
    suffixes = (
        "threshold_pr_mlp.net.0.weight",
        "seed_cluster_pr_mlp.net.0.weight",
    )
    matches = [
        key
        for key in state_dict
        if any(key.endswith(suffix) for suffix in suffixes)
    ]
    if not matches:
        raise KeyError("Could not find threshold PR MLP weights in checkpoint.")

    first_key = sorted(matches, key=len)[0]
    prefix = first_key[: -len("net.0.weight")]
    mlp_state = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            mlp_state[key[len(prefix) :]] = value

    required = {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"}
    missing = sorted(required - set(mlp_state))
    if missing:
        raise KeyError(f"Incomplete threshold PR MLP state; missing: {missing}")
    return mlp_state, prefix.rstrip(".")


def pareto_frontier_indices(precision, recall):
    order = torch.argsort(recall, descending=True)
    best_precision = precision.new_tensor(-1.0)
    pareto = []
    for idx in order.tolist():
        if precision[idx] > best_precision + 1e-7:
            pareto.append(idx)
            best_precision = precision[idx]
    if not pareto:
        return torch.empty(0, dtype=torch.long)
    pareto = torch.as_tensor(pareto, dtype=torch.long)
    return pareto[torch.argsort(recall[pareto])]


def predict_grid(predictor, num_points, seed_range, duplicate_range, device):
    seed_thresholds = torch.linspace(seed_range[0], seed_range[1], num_points, device=device)
    duplicate_thresholds = torch.linspace(
        duplicate_range[0],
        duplicate_range[1],
        num_points,
        device=device,
    )
    predictor.to(device)
    predictor.eval()
    with torch.no_grad():
        predictions = predictor(seed_thresholds, duplicate_thresholds).detach().cpu()

    seed_grid, duplicate_grid = torch.meshgrid(
        seed_thresholds.detach().cpu(),
        duplicate_thresholds.detach().cpu(),
        indexing="ij",
    )
    precision = predictions[..., 0].reshape(-1)
    recall = predictions[..., 1].reshape(-1)
    rq = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
    return {
        "precision": precision,
        "recall": recall,
        "rq": rq,
        "seed": seed_grid.reshape(-1),
        "duplicate": duplicate_grid.reshape(-1),
    }


def plot_curve(grid, pareto, top_rq, output_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision = grid["precision"]
    recall = grid["recall"]
    rq = grid["rq"]
    seed = grid["seed"]
    duplicate = grid["duplicate"]
    top_k = min(max(int(top_rq), 0), rq.numel())
    top_indices = torch.topk(rq, k=top_k).indices if top_k else torch.empty(0, dtype=torch.long)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=170)
    scatter = ax.scatter(
        recall.numpy(),
        precision.numpy(),
        c=rq.numpy(),
        s=6,
        alpha=0.24,
        cmap="magma",
        linewidths=0,
        label="Threshold grid",
    )
    if pareto.numel() > 0:
        ax.plot(
            recall[pareto].numpy(),
            precision[pareto].numpy(),
            color="#1f77b4",
            linewidth=2.1,
            label=f"Pareto front ({pareto.numel()} pts)",
        )
        ax.scatter(
            recall[pareto].numpy(),
            precision[pareto].numpy(),
            color="#1f77b4",
            s=14,
            zorder=3,
        )

    if top_indices.numel() > 0:
        ax.scatter(
            recall[top_indices].numpy(),
            precision[top_indices].numpy(),
            marker="*",
            s=180,
            color="#ffd400",
            edgecolors="#1a1a1a",
            linewidths=0.7,
            zorder=5,
            label=f"Highest RQ ({top_indices.numel()} pts)",
        )
        for rank, idx in enumerate(top_indices.tolist(), start=1):
            ax.annotate(
                f"#{rank} RQ={rq[idx]:.3f}\nseed={seed[idx]:.3f}, dup={duplicate[idx]:.3f}",
                xy=(float(recall[idx]), float(precision[idx])),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=7,
                color="#1a1a1a",
            )

    ax.set_xlabel("Predicted recall")
    ax.set_ylabel("Predicted precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted RQ")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())
    config_file = args.config_file or find_config_file(checkpoint_path)
    pr_cfg = load_pr_config(config_file)
    num_points, seed_range, duplicate_range = get_sweep_settings(args, pr_cfg)

    state_dict = load_checkpoint(checkpoint_path)
    mlp_state, mlp_prefix = extract_pr_mlp_state(state_dict)
    hidden_dim = int(mlp_state["net.0.weight"].shape[0])
    predictor = ThresholdPrecisionRecallMLP(hidden_dim=hidden_dim)
    predictor.load_state_dict(mlp_state)

    grid = predict_grid(predictor, num_points, seed_range, duplicate_range, args.device)
    pareto = pareto_frontier_indices(grid["precision"], grid["recall"])

    output_path = args.output
    if not output_path:
        output_path = str(Path(checkpoint_path).parent / "seed_cluster_pr_curve.png")
    os.makedirs(os.path.dirname(str(Path(output_path).resolve())), exist_ok=True)

    title = "LatentFormer threshold PR predictions"
    plot_curve(grid, pareto, args.top_rq, output_path, title)

    best_idx = int(grid["rq"].argmax())
    print(f"Loaded predictor: {mlp_prefix} (hidden_dim={hidden_dim})")
    if config_file:
        print(f"Config: {config_file}")
    print(f"Grid: {num_points} x {num_points}")
    print(f"Seed threshold range: {seed_range[0]:.6g} .. {seed_range[1]:.6g}")
    print(f"Duplicate threshold range: {duplicate_range[0]:.6g} .. {duplicate_range[1]:.6g}")
    print(
        "Best RQ: "
        f"rq={grid['rq'][best_idx]:.6f}, "
        f"precision={grid['precision'][best_idx]:.6f}, "
        f"recall={grid['recall'][best_idx]:.6f}, "
        f"seed={grid['seed'][best_idx]:.6f}, "
        f"duplicate={grid['duplicate'][best_idx]:.6f}"
    )
    print(f"Pareto points: {pareto.numel()}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
