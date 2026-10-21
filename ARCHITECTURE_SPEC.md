# ARCHITECTURE_SPEC.md: Technical Specification for MLIR-TensorOpt

## 1. System Objective

`MLIR-TensorOpt` is a domain-specific machine learning compiler and code-generation pipeline. Its objective is to ingest high-level deep learning representations (specifically Transformer Multi-Head Attention, Batched GEMM, and MLP blocks) from PyTorch and optimize their hardware execution through systematic MLIR dialect transformations.

---

## 2. Dialect Layering & Lowering Hierarchy

The compiler operates across 5 progressive dialect levels:

```
[PyTorch Model / PyTorch FX Graph]
       │
       ▼ (Frontend: torch-mlir / custom TorchScript importer)
[Torch Dialect (`torch`)]
       │
       ▼ (Pass: Torch-to-Linalg / Torch-to-Tosa Conversion)
[Linalg on Tensors (`linalg`, `tensor`)]  <--- CORE OPTIMIZATION ZONE
       │  • Attention Fusion (QKV + Mask + Softmax + MatMul)
       │  • Hierarchical Loop Tiling (L1/SRAM Locality)
       │  • Elementwise & Layout Transformations
       ▼ (Pass: One-Shot Bufferization)
[Linalg on MemRefs (`memref`, `affine`)]  <--- BUFFER & MEMORY PLANNING
       │  • In-place buffer assignment (Zero-Copy)
       │  • Deallocation & memory lifetime minimization
       ▼ (Pass: Linalg-to-Vector Lowering)
[Vector Dialect (`vector`)]               <--- SIMD & VECTOR ENGINE
       │  • 128/256/512-bit vectorization
       │  • Hardware masked transfers
       ▼ (Pass: Final LLVM Lowering)
[LLVM Dialect (`llvm`)]                   <--- CODEGEN & JIT
       │
       ▼ (ExecutionEngine / JIT)
[Native Machine Code Execution]
```

---

## 3. Core Optimization Passes

### Pass 1: `AttentionFusionPass`
- **Objective**: Standard eager evaluation of Multi-Head Attention executes $4$ separate DRAM memory trips:
  1. $\mathbf{Q} \mathbf{K}^T$ matrix multiplication $	o$ writes attention logits to memory.
  2. Scale by $rac{1}{\sqrt{d_k}}$ and apply causal mask $	o$ reads and writes scaled logits.
  3. Softmax across sequence dimension $	o$ reads and writes attention probabilities.
  4. Probabilities $	imes \mathbf{V}$ matrix multiplication $	o$ writes final output.
- **Compiler Strategy**:
  - Detect the composite pattern via `PatternMatchAndRewrite` on `linalg.generic` / `linalg.matmul`.
  - Fuse scaling and masking directly into the reduction loop of the first GEMM.
  - Implement online Softmax (Milakov & Gimelshein / FlashAttention formulation) inside a single fused kernel to maintain intermediate activations entirely inside registers or fast on-chip SRAM.

### Pass 2: `TilingAndLocalityPass`
- **Objective**: Large matrix multiplications ($\mathbf{M} 	imes \mathbf{K} 	imes \mathbf{N}$) exceed CPU L1 cache ($32	ext{ KB} - 64	ext{ KB}$) or accelerator on-chip SRAM partition sizes, causing memory pipeline stalls.
- **Compiler Strategy**:
  - Target: `linalg.matmul` or `linalg.generic`.
  - Tile dimensions $[M, N, K]$ with configurable tile sizes (e.g., $M=64, N=64, K=32$).
  - Emit outer loops as `scf.for` (or `scf.forall` for multi-threaded/accelerator execution) and inner compute kernels as unrolled micro-kernels.
  - Generates code that guarantees all read operands fit strictly into L1 Data Cache for the duration of the tile execution.

### Pass 3: `OneShotBufferizationPass`
- **Objective**: Values in the `tensor` dialect represent purely immutable values. Executing code on hardware requires explicit, mutable memory addresses (`memref`).
- **Compiler Strategy**:
  - Utilize MLIR's **Comprehensive One-Shot Bufferize** framework.
  - Enforce Destination-Passing Style (DPS).
  - Perform static read-after-write (RAW) alias analysis to verify which output tensors can safely overwrite input tensors in-place, eliminating temporary allocation (`memref.alloc`) and reducing dynamic memory footprints to near zero.

### Pass 4: `VectorizePass`
- **Objective**: Lower scalar and 1D loop nests into hardware vector operations.
- **Compiler Strategy**:
  - Transform `linalg` operations operating on contiguous slices into `vector.transfer_read`, `vector.contract` (vectorized multiply-accumulate), and `vector.transfer_write`.
  - Target SIMD vector widths (AVX-512, AVX2, ARM NEON, or AI accelerator vector execution units).

---

## 4. Multi-Precision & Numerical Drift Verification

A critical responsibility of an AI systems and compiler engineer (specifically emphasized by Cerebras) is validating that compiler optimizations do not degrade numerical accuracy.

### Precision Formats Supported:
1. **FP32 (IEEE 754 Single Precision)**: Baseline reference (23 mantissa, 8 exponent).
2. **BF16 (Brain Floating Point)**: 7 mantissa, 8 exponent (preserves FP32 dynamic range, reduced precision).
3. **FP8 (E4M3 and E5M2)**: 8-bit floating point for cutting-edge accelerator inference.

### Metrics Computed:
- **Maximum Absolute Difference**:
  $$\Delta_{\max} = \max_{i} |y_i^{	ext{compiled}} - y_i^{	ext{reference}}|$$
- **Relative Error**:
  $$\epsilon_{	ext{rel}} = rac{\| \mathbf{y}^{	ext{compiled}} - \mathbf{y}^{	ext{reference}} \|_2}{\| \mathbf{y}^{	ext{reference}} \|_2}$$
- **Root Mean Square Error (RMSE)**:
  $$	ext{RMSE} = \sqrt{rac{1}{N} \sum_{i=1}^{N} (y_i^{	ext{compiled}} - y_i^{	ext{reference}})^2}$$
- **Cosine Similarity**:
  $$	ext{CosineSim} = rac{\mathbf{y}^{	ext{compiled}} \cdot \mathbf{y}^{	ext{reference}}}{\| \mathbf{y}^{	ext{compiled}} \|_2 \|\mathbf{y}^{	ext{reference}}\|_2}$$

### Tolerance Thresholds:
- FP32: $\Delta_{\max} < 10^{-5}$, CosineSim $> 0.999999$
- BF16: $\Delta_{\max} < 10^{-2}$, CosineSim $> 0.9995$
- FP8: $\Delta_{\max} < 5 	imes 10^{-2}$, CosineSim $> 0.995$
