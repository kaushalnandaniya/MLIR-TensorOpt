"""
Benchmarking harness for latency, throughput, and speedup profiling.
Measures p50, p95, p99 latencies and throughput against PyTorch eager execution.
"""

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
from tabulate import tabulate
from rich.console import Console
from rich.table import Table

@dataclass
class BenchmarkResult:
    name: str
    warmup_iters: int
    benchmark_iters: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    std_us: float
    throughput_samples_per_sec: float
    speedup_over_baseline: float = 1.0


def benchmark_operator_latency(
    fn: Callable[[], Any],
    name: str = "Operator",
    warmup_iters: int = 50,
    benchmark_iters: int = 200,
    batch_size: int = 1,
    baseline_latency_us: float | None = None,
) -> BenchmarkResult:
    """
    Benchmarks execution latency of a callable function with warmup and percentile statistics.
    """
    # 1. Warmup runs
    for _ in range(warmup_iters):
        fn()

    # 2. Timed benchmark runs
    latencies = []
    for _ in range(benchmark_iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)  # Microseconds

    latencies_arr = np.array(latencies)
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    mean_lat = float(np.mean(latencies_arr))
    std_lat = float(np.std(latencies_arr))

    throughput = (batch_size / (mean_lat / 1e6)) if mean_lat > 0 else 0.0
    speedup = (baseline_latency_us / p50) if baseline_latency_us and baseline_latency_us > 0 else 1.0

    return BenchmarkResult(
        name=name,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        p50_us=p50,
        p95_us=p95,
        p99_us=p99,
        mean_us=mean_lat,
        std_us=std_lat,
        throughput_samples_per_sec=throughput,
        speedup_over_baseline=speedup,
    )


def print_rich_benchmark_table(results: List[BenchmarkResult]) -> None:
    """Renders a styled Rich table of benchmark results."""
    console = Console()
    table = Table(
        title="MLIR-TensorOpt: Latency & Throughput Benchmark",
        header_style="bold magenta",
    )
    table.add_column("Pipeline Configuration", justify="left")
    table.add_column("p50 (µs)", justify="right")
    table.add_column("p95 (µs)", justify="right")
    table.add_column("p99 (µs)", justify="right")
    table.add_column("Throughput (items/s)", justify="right")
    table.add_column("Speedup Factor", justify="center")

    for r in results:
        speedup_style = "bold green" if r.speedup_over_baseline >= 1.0 else "yellow"
        table.add_row(
            r.name,
            f"{r.p50_us:.1f}",
            f"{r.p95_us:.1f}",
            f"{r.p99_us:.1f}",
            f"{r.throughput_samples_per_sec:.1f}",
            f"[{speedup_style}]{r.speedup_over_baseline:.2f}x[/{speedup_style}]",
        )
    console.print(table)
