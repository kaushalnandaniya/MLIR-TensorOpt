# IMPLEMENTATION_ROADMAP.md: Step-by-Step Developer & Agent Guide

> **NOTICE FOR INCOMING AGENT**:
> You are building `MLIR-TensorOpt`, an end-to-end MLIR deep learning compiler and lowering pipeline.
> This document specifies the exact phases, code architecture, testing methodology, and commands to implement this project from scratch.

---

## 🧭 High-Level Phase Overview

- [x] **Phase 0**: Project Specification, Architecture Design & Alignment *(Completed)*
- [x] **Phase 1**: Environment & Dependencies Setup (LLVM/MLIR toolchain, Python environment) *(Completed)*
- [x] **Phase 2**: Project Skeleton & Build System (CMakeLists.txt, include/, lib/, tools/, python/) *(Completed)*
- [x] **Phase 3**: Frontend Importer (PyTorch Model $\to$ MLIR Torch/Linalg Dialect) *(Completed)*
- [x] **Phase 4**: Custom Optimization Passes (Attention Fusion, Loop Tiling, Vectorization) *(Completed)*
- [x] **Phase 5**: Execution Engine & Lowering to LLVM JIT *(Completed)*
- [x] **Phase 6**: Numerical Divergence Verification Suite (FP32 vs BF16 vs FP8) *(Completed)*
- [x] **Phase 7**: Performance Benchmarking Suite (Latency, Throughput, Cache Locality) *(Completed)*
- [x] **Phase 8**: Testing, Documentation, Git Repo & CI Setup *(Completed)*

---

## 🛠️ Phase 1: Environment & Toolchain Strategy

### Recommendation for Fast & Reliable Execution
Building the full `llvm-project` with MLIR from source takes 45-90 minutes and consumes ~25 GB of disk space.
**Recommended Approach**:
1. Use **prebuilt MLIR wheels** or `torch-mlir`:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install torch-mlir -f https://llvm.github.io/torch-mlir/package-index/
   pip install numpy tabulate rich matplotlib pytest
   ```
2. Or use the official LLVM MLIR Python bindings / system LLVM:
   ```bash
   # On macOS via Homebrew:
   brew install llvm
   # Or via prebuilt mlir python package:
   pip install mlir-wheels
   ```

---

## 💻 Phase 2: Project Directory Skeleton

Create the following file hierarchy:

```bash
mkdir -p include/mlir_tensoropt/Passes
mkdir -p lib/Passes
mkdir -p python/mlir_tensoropt/{frontend,passes,verification,benchmarks}
mkdir -p tests
```

---

## 🔬 Phase 3: Frontend Importer (`python/mlir_tensoropt/frontend/`)

1. **`models.py`**:
   - Define reference PyTorch modules:
     - `MultiHeadAttentionBlock`: Implements standard scaled dot-product attention with QKV projection, scaling, and causal mask.
     - `FeedForwardBlock`: Two-layer MLP with activation (GELU/ReLU) and LayerNorm.
2. **`export.py`**:
   - Ingests the PyTorch module.
   - Traces the model graph (via PyTorch FX or `torch_mlir.compile`).
   - Emits MLIR IR in `linalg-on-tensors` dialect.

---

## ⚡ Phase 4: Custom Passes & Rewrite Patterns

Implement the passes in `python/mlir_tensoropt/passes/` (and corresponding C++ passes in `lib/Passes/`):

1. **`attention_fusion.py`**:
   - Identifies the chain of ops: `Linear(QKV) -> Reshape -> MatMul(Q, K) -> Div(sqrt(d)) -> Mask -> Softmax -> MatMul(Attn, V)`.
   - Replaces with a single fused kernel representation.
2. **`loop_tiling.py`**:
   - Configures tile sizes: e.g., Tile $M=64, N=64, K=32$.
   - Applies tiling pass using `mlir.dialects.linalg.tile` or pass manager pipeline:
     ```
     convert-linalg-to-affine-loops
     affine-loop-tile{tile-size=64}
     ```
3. **`bufferize_and_vectorize.py`**:
   - Applies `one-shot-bufferize` to convert tensors to destination-passing memrefs.
   - Vectorizes loops to target SIMD widths (128-bit / 256-bit).

---

## 🎯 Phase 5: Lowering & JIT Execution Engine

Pipeline passes down to LLVM dialect:
```
linalg-to-llvm
memref-to-llvm
func-to-llvm
reconcile-unrealized-casts
```
Instantiate `ExecutionEngine` to JIT-compile and execute the compiled kernel natively.

---

## 🧪 Phase 6: Numerical Verification (`python/mlir_tensoropt/verification/`)

1. **`divergence_checker.py`**:
   - Feeds identical input tensors $\mathbf{X}$ to:
     a) Reference PyTorch Eager.
     b) Compiled MLIR-TensorOpt engine (in FP32, BF16, and FP8).
   - Computes:
     - Max Absolute Difference: $\max |Y_{	ext{compiled}} - Y_{	ext{ref}}|$
     - Root Mean Square Error (RMSE)
     - Cosine Similarity
   - Generates a styled terminal summary table using `tabulate` / `rich`.

---

## 📊 Phase 7: Benchmarking Harness (`python/mlir_tensoropt/benchmarks/`)

1. **`benchmark_latency.py`**:
   - Runs 100 warmup iterations and 1,000 timed iterations.
   - Measures:
     - Latency (p50, p95, p99 in microseconds).
     - Throughput (tokens/second or GFLOPS).
     - Speedup factor over PyTorch baseline ($X.X	imes$).
   - Plots latency distributions and memory footprint comparisons.

---

## 🚀 Phase 8: Final Polish & Git Setup

1. Initialize Git repository:
   ```bash
   git init
   git add .
   git commit -m "feat: initial implementation of MLIR-TensorOpt compiler pipeline"
   ```
2. Verify all unit tests pass:
   ```bash
   pytest tests/
   ```
3. Run the complete pipeline demo:
   ```bash
   python python/mlir_tensoropt/demo.py
   ```
