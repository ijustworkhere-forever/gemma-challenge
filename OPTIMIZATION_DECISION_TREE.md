# OPTIMIZATION_DECISION_TREE.md

> A deterministic decision system for choosing what to optimize next.
>
> Rule:
> Never guess. Always follow the tree.

---

# 1. Start Here

## Q1: Is GPU utilization < 85%?

### YES → Go to A (Underutilized GPU)

### NO → Go to Q2

---

# A. Underutilized GPU Path

## Likely Causes:

- Scheduler inefficiency
- CPU bottleneck
- Tokenizer bottleneck
- Sampling overhead
- I/O blocking

---

## Actions:

- Profile CPU usage
- Check queue depth
- Measure batching efficiency
- Inspect tokenization cost

---

## Fixes:

- Improve scheduler
- Increase batching efficiency
- Parallelize CPU work
- Move sampling to GPU

---

# Q2: Is memory bandwidth near max?

### YES → Go to B (Memory-bound)

### NO → Go to Q3

---

# B. Memory-bound Path

## Likely Causes:

- KV cache inefficiency
- Attention memory traffic
- Poor cache locality

---

## Actions:

- Nsight Compute memory analysis
- L2 cache hit rate check
- DRAM throughput inspection

---

## Fixes:

- KV cache layout optimization
- PagedAttention tuning
- FlashAttention improvements
- Reduce memory transfers

---

# Q3: Are kernels small and fragmented?

### YES → Go to C (Latency-bound)

### NO → Go to Q4

---

# C. Latency-bound Path

## Likely Causes:

- Too many kernel launches
- Decode loop inefficiency
- No CUDA Graph usage

---

## Actions:

- Count kernel launches per token
- Inspect decode loop
- Measure launch overhead

---

## Fixes:

- CUDA Graphs
- Kernel fusion
- Reduce per-token launches

---

# Q4: Is CPU usage high?

### YES → Go to D (CPU-bound)

### NO → Go to Q5

---

# D. CPU-bound Path

## Likely Causes:

- Tokenizer bottleneck
- Sampling overhead
- Python GIL
- Serialization

---

## Fixes:

- Move sampling to GPU
- Parallel tokenization
- Rust/C++ tokenizer
- Async pipeline

---

# Q5: Is GPU usage high but TPS still low?

### YES → Go to E (Compute-bound)

### NO → Re-profile

---

# E. Compute-bound Path

## Likely Causes:

- Inefficient kernels
- Low tensor core utilization
- Suboptimal GEMM shapes

---

## Fixes:

- FlashAttention tuning
- CUTLASS optimization
- Triton kernel rewrite
- FP16/BF16 tuning

---

# 2. Summary Map

| Condition | Action |
|----------|--------|
| Low GPU util | Fix scheduler |
| High memory | Fix KV cache |
| Many kernels | Use CUDA Graphs |
| High CPU | Offload work |
| High compute | Optimize kernels |

---

# 3. Golden Rule

Only optimize the bottleneck at the current node.

Do NOT jump ahead.

---

# 4. Exit Condition

You only leave this tree when:

- GPU > 95% utilization
- No single dominant bottleneck remains
