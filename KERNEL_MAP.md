# KERNEL_MAP.md

> Low-level mapping of Transformer operations to actual GPU kernels.
>
> Purpose:
> Identify exactly what runs on the GPU, how it is implemented, and where optimization is possible.

---

# 1. High-Level Mapping

| Transformer Component | Kernel Type | Typical Implementation |
|----------------------|-------------|------------------------|
| QKV Projection | GEMM | CUTLASS / cuBLAS |
| Attention | Fused Attention Kernel | FlashAttention |
| Softmax | Fused Kernel | FlashAttention / Triton |
| MLP (FFN) | GEMM + Activation | CUTLASS + fused GELU/SwiGLU |
| LayerNorm / RMSNorm | Reduction Kernel | Triton / fused CUDA |
| RoPE (Rotary Embedding) | Elementwise Kernel | Triton / custom CUDA |
| KV Cache Write | Memory Copy Kernel | Custom / vLLM paging |
| KV Cache Read | Memory Gather Kernel | PagedAttention |
| Sampling | CPU or CUDA Kernel | Python / Triton / custom CUDA |

---

# 2. Attention Block Breakdown

## Operation Flow

```
Input Hidden States
   ↓
QKV Projection (GEMM)
   ↓
Reshape / Split Heads
   ↓
RoPE (Rotary Embeddings)
   ↓
Attention Score (Q × K^T)
   ↓
Softmax
   ↓
Weighted Sum (Attention × V)
   ↓
Output Projection (GEMM)
```

---

## Kernel Mapping

### QKV Projection

- Kernel Type: GEMM
- Backends:
  - cuBLASLt
  - CUTLASS
- Bottleneck Type:
  - Compute-bound (tensor cores)

### Optimization Targets

- Tensor core utilization
- Weight layout (NHWC vs NCHW)
- Fusion with bias add

---

### RoPE (Rotary Positional Embedding)

- Kernel Type: Elementwise
- Current Implementations:
  - Triton
  - CUDA custom kernel

### Optimization Targets

- Fuse into QKV projection
- Reduce memory round-trips
- Vectorized complex arithmetic

---

### Attention Score (Q × Kᵀ)

- Kernel Type: GEMM-like / fused kernel
- Implementation:
  - FlashAttention

### Bottlenecks

- Memory bandwidth
- Global memory traffic
- Intermediate tensor storage

### Optimization Targets

- Tile size tuning
- Shared memory optimization
- Register reuse

---

### Softmax

- Kernel Type: Reduction + elementwise
- Typically fused in FlashAttention

### Optimization Targets

- Fuse with attention kernel
- Reduce intermediate writes
- Avoid global memory round-trips

---

### Weighted Sum (Attention × V)

- Kernel Type: GEMM-like fused operation

### Optimization Targets

- Fuse into FlashAttention kernel
- Improve memory coalescing
- Improve warp efficiency

---

# 3. MLP / Feed Forward Network

## Operation Flow

```
Hidden State
   ↓
Linear Layer 1 (GEMM)
   ↓
Activation (GELU / SwiGLU)
   ↓
Linear Layer 2 (GEMM)
```

---

## Kernel Mapping

### Linear Layers

- Kernel Type: GEMM
- Backend:
  - cuBLASLt
  - CUTLASS

### Optimization Targets

- Tensor core saturation
- Weight fusion
- Batch size tuning

---

### Activation Function

- GELU / SwiGLU

### Kernel Type

- Elementwise

### Optimization Targets

- Fuse into GEMM epilogue
- Reduce memory traffic
- Triton fusion

---

# 4. LayerNorm / RMSNorm

- Kernel Type: Reduction + normalization
- Implementation:
  - Triton kernels
  - CUDA custom kernels

---

## Bottlenecks

- Memory bandwidth bound
- Synchronization heavy
- Multiple passes over tensor

---

## Optimization Targets

- Fuse into adjacent GEMMs
- Single-pass reduction
- Register accumulation

---

# 5. KV Cache System

## Write Path

```
Key/Value tensors → KV Cache memory
```

### Kernel Type

- Memory copy / store kernel

### Bottlenecks

- Non-contiguous writes
- Allocation overhead
- Cache fragmentation

### Optimization Targets

- PagedAttention improvements
- Contiguous layout
- Prefetching

---

## Read Path

```
KV Cache → Attention kernel
```

### Kernel Type

- Gather + memory load

### Bottlenecks

- Random memory access
- L2 cache misses
- DRAM bandwidth saturation

### Optimization Targets

- Cache locality improvement
- Block-wise layout
- Better paging strategy

---

# 6. Sampling Kernel

## Operation Flow

```
Logits
  ↓
Top-K / Top-P filter
  ↓
Softmax (optional)
  ↓
Random sampling / argmax
```

---

## Implementation Types

### CPU Implementation (common baseline)

- Python / NumPy / PyTorch CPU

### GPU Implementation (optimal)

- Triton kernel
- CUDA fused sampling kernel

---

## Bottlenecks

- CPU-GPU sync
- Data transfer overhead
- Python overhead per token

---

## Optimization Targets

- Fully GPU-resident sampling
- Fuse top-k + softmax + sampling
- Avoid host round trips

---

# 7. Decode Loop Kernel Graph

Each token generation step:

```
[Layer 0]
  QKV GEMM
  RoPE
  Attention (FlashAttention)
  MLP
  KV cache update

[Layer 1]
  repeat...

[Layer N]
  repeat...
```

---

## Key Insight

Decode is NOT one kernel.

It is a repeated kernel graph.

Optimization must consider:

- launch overhead
- memory reuse
- cache reuse
- kernel fusion potential

---

# 8. Kernel Execution Bottleneck Types

| Bottleneck Type | Cause | Fix |
|----------------|------|-----|
| Compute-bound | GEMM saturation | Tensor core tuning |
| Memory-bound | KV cache | Layout optimization |
| Latency-bound | Kernel launches | CUDA Graphs |
| CPU-bound | Sampling/tokenizer | GPU offload |
| Sync-bound | CPU-GPU sync | Async pipeline |

---

# 9. Fusion Opportunities Map

## High Impact Fusion Candidates

### Attention Block

- QKV projection + bias
- RoPE + Q/K transform
- Softmax + attention score
- Attention + V multiplication

---

### MLP Block

- GEMM + activation
- Activation + second GEMM bias

---

### Norm Layers

- RMSNorm + residual add
- LayerNorm + bias

---

# 10. CUDA Graph Opportunities

Best candidates:

- Decode loop (highest ROI)
- MLP + Attention per layer
- KV cache update cycle

Goal:

Eliminate kernel launch overhead entirely during decode.

---

# 11. Memory Access Patterns

## Optimal Pattern

- Coalesced access
- Contiguous KV blocks
- Shared memory reuse
- Register-resident intermediate results

## Problem Patterns

- Strided KV cache access
- Random memory reads
- Excess global memory writes

---

# 12. Optimization Priority Map

| Component | Priority | Reason |
|-----------|----------|--------|
| Decode Loop | P0 | Dominates runtime |
| KV Cache | P0 | Memory bottleneck |
| Attention | P0 | Compute + memory heavy |
| Scheduler | P0 | Controls GPU utilization |
| MLP | P1 | High compute load |
| Sampling | P1 | Hidden CPU bottleneck |
| Prefill | P2 | One-time cost |

---

# 13. Mental Model

Think of the GPU pipeline as:

```
GEMM Engine → Attention Engine → Memory Engine → Repeat Loop
```

And everything else is:

- scheduling
- memory movement
- synchronization

---

# 14. Key Insight

Most performance gains do NOT come from faster kernels.

They come from:

- fewer kernels
- better memory layout
- better reuse
- less synchronization

---

# 15. Next Step

👉 Run baseline trace  
👉 Map actual kernels (Nsight) to this document  
👉 Identify top 3 slowest kernels  
👉 Target first optimization pass  

---
