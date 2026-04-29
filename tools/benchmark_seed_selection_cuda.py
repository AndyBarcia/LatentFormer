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


def _valid_outputs(result):
    signatures, mask, scores = result
    outputs = []
    for b in range(scores.shape[0]):
        valid_scores = scores[b, mask[b]].detach().cpu()
        valid_signatures = signatures[b, mask[b]].detach().cpu()
        order = torch.argsort(valid_scores)
        outputs.append((valid_scores[order], valid_signatures[order]))
    return outputs


def _check_scores(cpu_result, cuda_result):
    cpu_outputs = _valid_outputs(cpu_result)
    cuda_outputs = _valid_outputs(cuda_result)
    for idx, ((lhs_scores, lhs_signatures), (rhs_scores, rhs_signatures)) in enumerate(
        zip(cpu_outputs, cuda_outputs)
    ):
        if (
            lhs_scores.numel() != rhs_scores.numel()
            or not torch.allclose(lhs_scores, rhs_scores, atol=1e-6, rtol=1e-6)
            or not torch.allclose(lhs_signatures, rhs_signatures, atol=1e-6, rtol=1e-6)
        ):
            raise AssertionError(
                f"batch {idx}: CPU selected {lhs_scores.numel()} seeds, "
                f"CUDA selected {rhs_scores.numel()} seeds"
            )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--seed-threshold", type=float, default=0.2)
    parser.add_argument("--duplicate-threshold", type=float, default=0.8)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    native_op = _load_native_op()

    query_signatures_cpu = torch.randn(args.batch_size, args.queries, args.channels)
    query_seed_logits_cpu = torch.randn(args.batch_size, args.queries)
    query_signatures_cuda = query_signatures_cpu.cuda()
    query_seed_logits_cuda = query_seed_logits_cpu.cuda()

    def cpu_call():
        return native_op.clustering_seed_selection_forward(
            query_signatures_cpu,
            query_seed_logits_cpu,
            args.seed_threshold,
            args.duplicate_threshold,
            args.metric,
            1e-6,
            0.1,
        )

    def cuda_call():
        return native_op.clustering_seed_selection_forward(
            query_signatures_cuda,
            query_seed_logits_cuda,
            args.seed_threshold,
            args.duplicate_threshold,
            args.metric,
            1e-6,
            0.1,
        )

    cpu_result = cpu_call()
    cuda_result = cuda_call()
    torch.cuda.synchronize()
    _check_scores(cpu_result, cuda_result)

    cpu_ms = _time_call(cpu_call, args.warmup, args.iters)
    cuda_ms = _time_call(cuda_call, args.warmup, args.iters)
    cpu_counts = [int(mask.sum()) for mask in cpu_result[1]]
    cuda_shape = tuple(cuda_result[0].shape)

    print(f"B={args.batch_size} Q={args.queries} C={args.channels}")
    print(f"thresholds: seed={args.seed_threshold} duplicate={args.duplicate_threshold} metric={args.metric}")
    print(f"selected seeds per image: {cpu_counts}")
    print(f"CPU native:  {cpu_ms:.3f} ms")
    print(f"CUDA native: {cuda_ms:.3f} ms")
    print(f"speedup:     {cpu_ms / cuda_ms:.2f}x")
    print(f"CUDA output shape: {cuda_shape}")


if __name__ == "__main__":
    main()
