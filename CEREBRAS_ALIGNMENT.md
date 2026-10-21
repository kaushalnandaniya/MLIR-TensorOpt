# CEREBRAS_ALIGNMENT.md: Technical Alignment & Interview Guide for Cerebras Systems

## 1. Context: Cerebras Systems & The Full Stack ML Engineer Role

Cerebras Systems builds the **Wafer-Scale Engine (WSE)**, the largest and fastest AI processor in the world. Unlike traditional GPUs (which rely on external High Bandwidth Memory / HBM and suffer from memory bus bottlenecks), Cerebras integrates:
- **900,000 AI-optimized compute cores** on a single wafer.
- **44 Gigabytes of on-chip SRAM** directly adjacent to the cores.
- **21 Petabytes per second of memory bandwidth** ($>10	imes$ faster than any GPU).

### What Cerebras Software Engineers Do:
Because the architecture is unique, Cerebras engineers cannot use NVIDIA CUDA. They built an end-to-end compiler stack (using **MLIR**, **LLVM**, and CSL - Cerebras Software Language).
Their day-to-day job is:
1. Taking bleeding-edge PyTorch models (Llama 3, DeepSeek, MoE, Whisper).
2. Lowering them through custom MLIR dialects onto the 900,000-core fabric.
3. Fusing operations to keep intermediate data inside on-chip SRAM.
4. Tiling computation across cores so no core stalls waiting for memory.
5. Benchmarking numerical drift when running in BF16 and FP8.

---

## 2. Direct Mapping: Cerebras Job Description vs. MLIR-TensorOpt

| Cerebras Job Requirement | How `MLIR-TensorOpt` Directly Addresses It |
| :--- | :--- |
| **"Own model bring-up from reference to execution"** | Frontend importer ingest PyTorch Transformer Attention and MLP blocks and converts them into structured MLIR IR. |
| **"Develop MLIR-based lowering paths and transformation passes"** | Custom C++/Python rewrite patterns lowering `torch` dialect $	o$ `linalg` $	o$ `vector` $	o$ `llvm`. |
| **"Optimize performance through operator fusion, layout transformations, and memory planning"** | `AttentionFusionPass` fuses QKV projections, scale-mask-softmax, and GEMM; loop tiling optimizes for SRAM/L1 cache reuse. |
| **"Validate numerical correctness across precision formats (FP32, BF16, FP8)"** | Automated drift verification framework computing RMSE, Max Absolute Error, and Cosine Similarity against PyTorch eager baselines. |
| **"Contributions to MLIR, LLVM, or inference frameworks"** | A functional, tested GitHub repository utilizing the official MLIR / LLVM compiler infrastructure. |

---

## 3. Interview Talking Points & Technical Q&A

### Q1: Why use MLIR instead of writing directly in LLVM IR or C++?
> **Answer**: *"LLVM IR is designed for scalar, general-purpose computation. Once a high-level tensor operation like `torch.matmul(A, B)` or `Softmax` is lowered to LLVM IR, it gets decomposed into nested pointer arithmetic and branches, losing high-level structural semantics (shapes, broadcast rules, memory access strides). In contrast, MLIR allows us to perform high-level transformations—like operator fusion, hierarchical loop tiling, and memory footprint planning—at the `linalg` and `tensor` dialect levels where the mathematical structure of the tensor operation is fully preserved."*

### Q2: What is the benefit of Operator Fusion on accelerator hardware?
> **Answer**: *"On modern AI accelerators, compute is cheap, but memory traffic is the primary bottleneck. For example, in an unfused Attention block, the result of the first MatMul is written to memory, read back to apply scale and mask, written to memory, read back to compute Softmax, and written again. Each roundtrip consumes precious memory bandwidth and introduces pipeline stalls. By fusing these operations into a single kernel loop, intermediate tensors remain resident inside processor registers or local L1/SRAM cache, completely eradicating off-chip memory traffic and boosting throughput."*

### Q3: How do you identify and diagnose numerical divergence in lower precision formats (BF16, FP8)?
> **Answer**: *"BF16 preserves the 8-bit dynamic range of FP32 but reduces the mantissa to 7 bits, making it susceptible to precision loss in long accumulations. FP8 (e.g., E4M3 or E5M2) has even tighter dynamic range and quantization noise. In `MLIR-TensorOpt`, we implement a layer-by-layer divergence probe that checks maximum absolute difference, relative error, and cosine similarity against FP32 eager baselines. If divergence exceeds threshold, we isolate the specific lowering pass or operator (e.g., Softmax normalization denominator underflowing) and apply numerically stable reformulations like online Softmax or scaled reductions."*

### Q4: What is Destination-Passing Style (DPS) and why is One-Shot Bufferization important?
> **Answer**: *"In MLIR, operations on the `tensor` dialect have purely functional, value semantics (immutable mathematical objects). However, physical hardware needs to write to explicit memory buffers (`memref`). One-Shot Bufferization analyzes the entire computation graph in destination-passing style, determining where an output buffer can safely reuse an input buffer without causing read-after-write conflicts. This eliminates unnecessary dynamic memory allocations (`memref.alloc`) and keeps memory utilization strictly deterministic."*

---

## 4. How to Feature This on Your Resume Once Built

Once you build and test this project, you can confidently include it in your CV:

```latex
\textbf{\textcolor{primaryColor}{MLIR-TensorOpt: Deep Learning Compiler Pipeline}} \hfill \href{https://github.com/kaushalnandaniya/MLIR-TensorOpt}{\faGithub} \\
\textit{C++, MLIR, LLVM, \texttt{torch-mlir}, Dialects, Loop Tiling, Numerical Verification}
\begin{itemize}[leftmargin=1.1em, itemsep=0.5pt, topsep=1pt, parsep=0pt, partopsep=0pt]
    \item Developed custom \textbf{MLIR rewrite patterns} and transformation passes lowering PyTorch tensor ops (Attention, GEMM, LayerNorm) through \textbf{linalg} dialects to hardware-specific execution mappings.
    \item Implemented \textbf{operator fusion} (QKV projection + scaled dot-product attention) and loop tiling passes, eliminating redundant memory roundtrips and optimizing tensor layouts for on-chip cache hierarchies.
    \item Validated numerical correctness and accuracy impact across \textbf{FP32, BF16, and FP8} precision formats, isolating numerical drift and diagnosing lowering bugs against reference PyTorch eager execution.
\end{itemize}
```
