"""
Benchmarking and profiling suite for MLIR-TensorOpt.
"""

from .benchmark_latency import benchmark_operator_latency, BenchmarkResult

__all__ = [
    "benchmark_operator_latency",
    "BenchmarkResult",
]
