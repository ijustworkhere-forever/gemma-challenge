# BASELINE_REPORT_TEMPLATE.md

> Standardized report for all baseline and post-optimization benchmarks.
>
> Rule:
> If results are not in this format, they are not comparable.

---

# 1. Experiment Metadata

## Run ID

```
exp-0000
```

## Date

YYYY-MM-DD

## Commit Hash

```
abcdef123
```

## Engine

- vLLM / TensorRT-LLM / SGLang / Custom

## Model

- gemma-4-E4B-it (or relevant variant)

## Hardware

- GPU:
- CPU:
- RAM:
- CUDA Version:
- Driver Version:

---

# 2. Benchmark Configuration

## Inference Settings

- Batch Size:
- Max Context Length:
- Temperature:
- Top-K:
- Top-P:
- Precision (FP16 / BF16 / FP8):

---

## Runtime Flags

```
(add CLI flags or config here)
```

---

# 3. Performance Results

## Core Metrics

| Metric | Value |
|------|------|
| TPS (Tokens/sec) | |
| TTFT (Time to First Token) | |
| Latency (avg) | |
| P95 Latency | |
| GPU Utilization | |
| VRAM Usage | |
| PPL (Perplexity) | |

---

## Stability

- Run Variance (%):
- Number of Runs Averaged:

---

# 4. Profiling Summary

## Nsight Systems Summary

- GPU Busy Time:
- GPU Idle Time:
- CPU Bottlenecks:
- Sync Events:

---

## Nsight Compute Summary

- SM Occupancy:
- Tensor Core Utilization:
- DRAM Throughput:
- L2 Cache Hit Rate:

---

## PyTorch Profiler Summary

- Top CPU Ops:
- Top GPU Ops:
- Sync Overhead:

---

# 5. Bottleneck Classification

Primary Bottleneck:

- [ ] Compute-bound
- [ ] Memory-bound
- [ ] Latency-bound
- [ ] CPU-bound
- [ ] Sync-bound

Secondary Bottleneck:

-

---

# 6. Observations

- What looked slow?
- What surprised you?
- What did profiler disagree with?

---

# 7. Hypothesis for Next Improvement

- What do you think will improve TPS?
- Why?

---

# 8. Next Action

- [ ] Scheduler
- [ ] KV Cache
- [ ] CUDA Graphs
- [ ] Kernel Fusion
- [ ] Sampling
- [ ] Other: _______

---

# 9. Raw Artifacts

Links:

- Nsight trace:
- Torch profile:
- Logs:
- Charts:

---
