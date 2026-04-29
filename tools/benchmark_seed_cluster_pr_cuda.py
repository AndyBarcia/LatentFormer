import argparse
import time
from pathlib import Path

import torch


def _load_native_op():
    import importlib.util

    op_dir = Path(__file__).resolve().parents[1] / "mask2former/modeling/seed_selection_ops"
    native_path = next(op_dir.glob("LatentFormerSeedSelection*.so"))
    spec = importlib.util.spec_from_file_location("LatentFormerSeedSelection", native_path)
    native_op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_op)
    return native_op


def _time_call(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _check(cpu_result, cuda_result):
    names = (
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "total_tp",
        "total_fp",
        "total_fn",
        "micro_precision",
        "micro_recall",
    )
    for name, cpu_value, cuda_value in zip(names, cpu_result, cuda_result):
        lhs = cpu_value.detach().cpu()
        rhs = cuda_value.detach().cpu()
        if not torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5):
            diff = (lhs - rhs).abs().max().item()
            raise AssertionError(f"{name} mismatch, max abs diff={diff}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--seed-thresholds", type=int, default=3)
    parser.add_argument("--duplicate-thresholds", type=int, default=3)
    parser.add_argument("--matched-frac", type=float, default=0.15)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    native_op = _load_native_op()

    query_signatures_cpu = torch.randn(args.batch_size, args.queries, args.channels)
    query_seed_logits_cpu = torch.randn(args.batch_size, args.queries)
    matched_query_mask_cpu = torch.rand(args.batch_size, args.queries) < args.matched_frac
    matched_gt_indices_cpu = torch.arange(args.queries).expand(args.batch_size, -1).clone()
    seed_thresholds_cpu = torch.linspace(0.1, 0.9, args.seed_thresholds)
    duplicate_thresholds_cpu = torch.linspace(0.1, 0.9, args.duplicate_thresholds)

    query_signatures_cuda = query_signatures_cpu.cuda()
    query_seed_logits_cuda = query_seed_logits_cpu.cuda()
    matched_query_mask_cuda = matched_query_mask_cpu.cuda()
    matched_gt_indices_cuda = matched_gt_indices_cpu.cuda()
    seed_thresholds_cuda = seed_thresholds_cpu.cuda()
    duplicate_thresholds_cuda = duplicate_thresholds_cpu.cuda()

    def cpu_call():
        return native_op.seed_cluster_precision_recall_forward(
            query_signatures_cpu,
            query_seed_logits_cpu,
            matched_query_mask_cpu,
            matched_gt_indices_cpu,
            seed_thresholds_cpu,
            duplicate_thresholds_cpu,
            args.metric,
            1e-6,
            0.1,
        )

    def cuda_call():
        return native_op.seed_cluster_precision_recall_forward(
            query_signatures_cuda,
            query_seed_logits_cuda,
            matched_query_mask_cuda,
            matched_gt_indices_cuda,
            seed_thresholds_cuda,
            duplicate_thresholds_cuda,
            args.metric,
            1e-6,
            0.1,
        )

    cpu_result = cpu_call()
    cuda_result = cuda_call()
    torch.cuda.synchronize()
    _check(cpu_result, cuda_result)

    cpu_ms = _time_call(cpu_call, args.warmup, args.iters)
    cuda_ms = _time_call(cuda_call, args.warmup, args.iters)

    print(f"B={args.batch_size} Q={args.queries} C={args.channels}")
    print(f"threshold grid: {args.seed_thresholds} seed x {args.duplicate_thresholds} duplicate")
    print(f"matched queries: {matched_query_mask_cpu.sum(dim=1).tolist()}")
    print(f"CPU native:  {cpu_ms:.3f} ms")
    print(f"CUDA native: {cuda_ms:.3f} ms")
    print(f"speedup:     {cpu_ms / cuda_ms:.2f}x")


if __name__ == "__main__":
    main()
