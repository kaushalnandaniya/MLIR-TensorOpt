"""
Unit tests for benchmarking harness and visualization plots.
"""

from pathlib import Path
import pytest
import torch
from mlir_tensoropt.benchmarks.benchmark_latency import (
    benchmark_operator_latency,
    plot_benchmark_results,
    BenchmarkResult,
)

def test_benchmark_operator_latency():
    """Verify operator benchmarking produces valid timing and stats."""
    def dummy_op():
        a = torch.ones(8, 8)
        return torch.matmul(a, a)

    result = benchmark_operator_latency(
        dummy_op,
        name="DummyMatmul",
        warmup_iters=5,
        benchmark_iters=10,
        batch_size=1,
    )
    assert result.p50_us > 0
    assert result.p95_us >= result.p50_us
    assert result.throughput_samples_per_sec > 0


def test_plot_benchmark_results(tmp_path):
    """Verify matplotlib plot generation without displaying GUI window."""
    results = [
        BenchmarkResult(
            name="Baseline",
            warmup_iters=5,
            benchmark_iters=10,
            p50_us=100.0,
            p95_us=110.0,
            p99_us=120.0,
            mean_us=102.0,
            std_us=5.0,
            throughput_samples_per_sec=10000.0,
            speedup_over_baseline=1.0,
        ),
        BenchmarkResult(
            name="MLIR-TensorOpt",
            warmup_iters=5,
            benchmark_iters=10,
            p50_us=70.0,
            p95_us=75.0,
            p99_us=80.0,
            mean_us=71.0,
            std_us=3.0,
            throughput_samples_per_sec=14285.0,
            speedup_over_baseline=1.43,
        ),
    ]
    out_file = tmp_path / "test_benchmark.png"
    ret = plot_benchmark_results(results, str(out_file))
    assert Path(ret).exists()
    assert Path(ret).stat().st_size > 0
