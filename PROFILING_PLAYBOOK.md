# PROFILING_PLAYBOOK.md

> A practical guide to profiling the Gemma inference stack.
>
> Goal:
> Identify real bottlenecks (not symptoms) using deterministic, repeatable profiling.

---

# 0. Core Principle

Never optimize without a trace.

If you didn’t measure it in Nsight / profiler output, it does not exist.

---

# 1. Profiling Stack Overview

We use three layers of profiling:

| Tool | Purpose |
|------|--------|
| Nsight Systems | Whole system timeline (CPU + GPU + sync) |
| Nsight Compute | Kernel-level performance analysis |
| PyTorch Profiler | Operator-level breakdown |

---

# 2. Profiling Workflow (Always Follow This Order)

```
1. Run baseline benchmark
2. Capture Nsight Systems trace
3. Identify macro bottlenecks
4. Drill into kernels with Nsight Compute
5. Map results to architecture
6. Decide optimization target
```

---

# 3. Nsight Systems (nsys)

## 3.1 Purpose

Shows the full execution timeline:

- CPU threads
- GPU kernels
- CUDA API calls
- synchronization points

---

## 3.2 How to Run

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=gemma_trace \
  python benchmark.py
```

---

## 3.3 What to Look For

### GPU Idle Time

If GPU is idle:

- Scheduler issue
- CPU bottleneck
- Input starvation

---

### Kernel Launch Frequency

If too many small kernels:

- Decode inefficiency
- Missing CUDA Graphs
- Lack of fusion

---

### CPU → GPU Gaps

If gaps exist:

- Tokenizer bottleneck
- Python overhead
- Blocking I/O

---

### Overlapping Execution

Check:

- Are CPU and GPU overlapping?
- Or strictly serialized?

---

## 3.4 Red Flags

- Long CPU gaps before GPU starts
- Frequent synchronization calls
- Tiny fragmented kernels
- Low GPU utilization (<70%)

---

# 4. Nsight Compute (ncu)

## 4.1 Purpose

Deep dive into individual GPU kernels.

---

## 4.2 How to Run

```bash
ncu --set full \
  --target-processes all \
  python benchmark.py
```

Or target specific kernels:

```bash
ncu --kernel-name regex:attention \
  python benchmark.py
```

---

## 4.3 Key Metrics

### Compute Utilization

- SM occupancy
- Tensor core utilization
- FLOPS achieved vs peak

---

### Memory Metrics

- DRAM throughput
- L2 hit rate
- Global load efficiency

---

### Execution Efficiency

- Warp execution efficiency
- Divergence
- Stalls

---

## 4.4 Interpretation Guide

### Case 1: Low SM occupancy

Cause:
- Too few threads
- Small batch size
- Kernel launch inefficiency

Fix:
- Increase batch
- Fuse kernels
- Use CUDA Graphs

---

### Case 2: High DRAM usage

Cause:
- KV cache bottleneck
- Poor memory layout
- Attention inefficiency

Fix:
- Improve cache locality
- PagedAttention tuning
- Reduce memory traffic

---

### Case 3: Low tensor core usage

Cause:
- Non-optimal GEMM shapes
- Precision mismatch
- Kernel fallback

Fix:
- Adjust shapes
- Use FP16/BF16 properly
- CUTLASS tuning

---

# 5. PyTorch Profiler

## 5.1 Purpose

High-level operator timing.

---

## 5.2 How to Run

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    run_model()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 5.3 What to Look For

- Slow Python ops
- Unexpected CPU bottlenecks
- Small but frequent ops
- Synchronization points

---

# 6. Bottleneck Classification System

Every issue must be classified:

| Type | Symptom | Fix Strategy |
|------|--------|-------------|
| Compute-bound | High GPU usage, slow | Optimize kernels |
| Memory-bound | DRAM saturated | KV cache optimization |
| Latency-bound | Many small kernels | CUDA Graphs |
| CPU-bound | GPU idle | Tokenizer/sampling fix |
| Sync-bound | Frequent stalls | Async pipeline |

---

# 7. Decode Loop Profiling (Critical Path)

Focus here first.

Each token generation step:

```
Attention
MLP
RMSNorm
KV Cache Update
Sampling
```

---

## What to Measure

- Time per token
- Kernel launch overhead per token
- Memory traffic per token
- CPU sync per token

---

## Expected Pattern

If optimized:

- Few large kernels
- High GPU occupancy
- Minimal CPU interaction

If NOT optimized:

- Many tiny kernels
- GPU idle between steps
- High launch overhead

---

# 8. Scheduler Profiling

## What to Measure

- Queue wait time
- Batch formation time
- GPU idle gaps

---

## Red Flags

- GPU < 90% utilization
- Frequent empty queues
- Uneven batch sizes

---

# 9. KV Cache Profiling

## Metrics

- L2 cache hit rate
- DRAM throughput
- Allocation frequency
- Memory fragmentation

---

## Red Flags

- Random memory access patterns
- High bandwidth saturation
- Cache misses dominating runtime

---

# 10. Sampling Profiling

Often ignored — but critical.

## What to Check

- CPU usage per token
- Sync between GPU → CPU → GPU
- Top-k / top-p cost

---

## Red Flags

- CPU spike every token
- Frequent GPU synchronization
- Python-heavy sampling loop

---

# 11. GPU Utilization Checklist

Target: >95%

If below:

- Is scheduler starving GPU?
- Is decode batch too small?
- Are kernels too fragmented?
- Is CPU blocking?

---

# 12. Profiling Strategy (Important)

Never do:

- Random optimization
- Guess-based kernel changes
- Multi-variable changes

Always do:

1. Profile
2. Hypothesize
3. Change ONE thing
4. Re-profile
5. Compare

---

# 13. Golden Rule

> “If you can’t see it in Nsight, you can’t fix it.”

---

# 14. Output Artifacts (Every Run Must Save)

Each profiling run should produce:

```
/profiling/
  nsight_trace.qdrep
  nsight_report.csv
  torch_profile.json
  benchmark_results.json
  notes.md
```

---

# 15. Final Objective Mapping

| Layer | Tool | Goal |
|------|------|------|
| System | Nsight Systems | Find macro bottlenecks |
| Kernel | Nsight Compute | Optimize GPU usage |
| Model | Torch Profiler | Understand ops |
| Benchmark | Custom script | Measure TPS/PPL |

---

# 16. Next Step

👉 Run baseline benchmark  
👉 Capture full Nsight Systems trace  
👉 Identify top 3 bottlenecks  
👉 Move to targeted optimization phase  

---
