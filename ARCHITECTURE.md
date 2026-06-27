# ARCHITECTURE.md

> High-level system design of the Gemma inference pipeline.
>
> This document explains:
>
> - How inference actually flows
> - Where compute happens
> - Where memory bottlenecks appear
> - Where optimization opportunities exist

---

# 1. System Overview

The inference stack can be broken into 5 major stages:

```
Request
  ↓
Tokenizer (CPU)
  ↓
Prefill (GPU)
  ↓
KV Cache Write/Read (GPU Memory)
  ↓
Decode Loop (GPU)
  ↓
Sampling + Post-processing (CPU/GPU hybrid)
  ↓
Streaming Output
```

---

# 2. End-to-End Pipeline

## 2.1 Request Layer

Responsible for:

- Receiving input prompts
- Managing concurrent requests
- Queueing / scheduling
- Batching decisions

### Key Characteristics

- CPU-bound under high concurrency
- Highly sensitive to scheduling strategy
- Direct impact on GPU utilization

### Optimization Targets

- Scheduler efficiency
- Queue latency
- Batch formation strategy

---

## 2.2 Tokenization (CPU)

Responsible for:

- Converting text → token IDs

### Bottlenecks

- Python overhead
- Single-thread execution
- GIL contention

### Optimization Opportunities

- Parallel tokenization
- Rust/C++ tokenizer backend
- Pre-tokenization caching

---

## 2.3 Prefill Phase (GPU)

Responsible for:

- Processing full prompt
- Building initial KV cache
- Running full attention over sequence

```
Prompt Tokens → Transformer Layers → KV Cache
```

### Characteristics

- Compute-heavy
- High memory bandwidth usage
- Often long-running compared to decode

### Bottlenecks

- Attention kernel efficiency
- Memory bandwidth saturation
- Large matrix multiplications

### Optimization Opportunities

- FlashAttention tuning
- Kernel fusion
- Tensor core utilization
- Sequence packing

---

## 2.4 KV Cache System

This is one of the most critical subsystems.

Responsible for:

- Storing key/value tensors per layer
- Reusing computation across decode steps

```
Layer Output → KV Cache → Next Token Attention
```

### Characteristics

- Memory bandwidth bound
- Highly sensitive to layout
- Fragmentation affects performance

### Bottlenecks

- Non-contiguous memory access
- Cache paging inefficiency
- L2 cache misses
- Allocation overhead

### Optimization Opportunities

- PagedAttention improvements
- Cache layout redesign (contiguous vs paged)
- FP8 / compressed KV cache
- Better memory reuse strategy

---

## 2.5 Decode Loop (GPU)

This is the critical throughput stage.

Runs repeatedly for each generated token:

```
For each token:
    Attention(KV Cache)
    MLP
    LayerNorm
    Update KV Cache
```

### Characteristics

- Latency sensitive
- Repeated execution (dominates runtime)
- Kernel launch overhead matters

### Bottlenecks

- Kernel launch overhead
- Inefficient scheduling
- Small batch inefficiency
- Memory stalls

### Optimization Opportunities

- CUDA Graphs (huge win potential)
- Kernel fusion
- Continuous batching
- Tensor core optimization
- Reduced launch frequency

---

## 2.6 Sampling (CPU/GPU hybrid)

Responsible for:

- Selecting next token
- Applying:
  - Top-K
  - Top-P
  - Temperature scaling

### Characteristics

- CPU-heavy in many stacks
- Often overlooked bottleneck
- Happens every token

### Bottlenecks

- Python overhead
- CPU-GPU synchronization
- Unfused operations

### Optimization Opportunities

- Move sampling to GPU
- Fuse sampling kernel
- Vectorized top-k/top-p
- Eliminate CPU round trips

---

## 2.7 Output Streaming

Responsible for:

- Sending tokens to client
- Managing partial responses

### Characteristics

- Network + CPU bound
- Can introduce stalls if blocking

### Optimization Opportunities

- Async streaming
- Buffering strategies
- Non-blocking IO

---

# 3. Full Execution Timeline

```
TIME →
│
│ Request arrives
│
├── Tokenization (CPU)
│
├── Prefill (GPU heavy compute burst)
│
├── KV cache initialized
│
├── Decode loop starts
│     ├── Attention
│     ├── MLP
│     ├── Sampling
│     └── KV update
│     (repeats per token)
│
└── Output streaming
```

---

# 4. Performance Hotspots (Expected)

Ranked by likely impact:

## #1 Decode Loop

- Dominates total runtime
- Repeats per token
- Sensitive to every micro-optimization

## #2 KV Cache

- Memory bandwidth bottleneck
- Layout determines performance ceiling

## #3 Scheduler

- Controls GPU utilization
- Can cause idle GPU time

## #4 Prefill

- Heavy compute burst
- Important but less dominant than decode

## #5 Sampling

- Often underestimated
- Can become CPU bottleneck

---

# 5. Bottleneck Classification Guide

Use this to interpret profiler results:

| Symptom | Likely Cause |
|----------|-------------|
| Low GPU usage | Scheduler issue |
| High memory bandwidth | KV cache / attention |
| High kernel launch count | Decode inefficiency |
| High CPU usage | Tokenizer / sampling |
| Long TTFT | Prefill inefficiency |
| Poor TPS scaling | Decode bottleneck |

---

# 6. Optimization Levers Map

## Scheduler Layer

- Continuous batching
- Request ordering
- Queue management

---

## Compute Layer

- FlashAttention tuning
- Kernel fusion
- Triton kernels
- TensorRT-LLM optimizations

---

## Memory Layer

- KV cache layout
- Paging strategy
- Allocation optimization
- Memory reuse

---

## Execution Layer

- CUDA Graphs
- Async execution
- Stream overlap

---

## Sampling Layer

- GPU sampling kernels
- Fused top-k/top-p
- Reduced synchronization

---

# 7. Key Insight

Most TPS improvements do NOT come from faster math.

They come from:

- fewer kernel launches
- better memory locality
- better scheduling
- fewer CPU/GPU sync points

---

# 8. Mental Model

Think of the system as:

```
GPU = expensive compute engine

CPU = traffic controller

KV Cache = memory highway

Scheduler = traffic lights

Decode loop = main workload loop
```

If any layer is inefficient, the entire system slows down.

---

# 9. Optimization Strategy Reminder

Before changing anything:

1. Identify bottleneck layer
2. Confirm with profiler
3. Change ONE variable
4. Measure again
5. Keep or revert

---

# 10. Current Unknowns

These are critical unanswered questions:

- What dominates decode time: compute or memory?
- Is scheduler causing GPU underutilization?
- How expensive is KV cache access per token?
- Is sampling visible in runtime profile?
- Where does kernel launch overhead peak?

---

# 11. Next Step

👉 Run baseline benchmark  
👉 Capture full Nsight trace  
👉 Map real runtime to this architecture  
👉 Identify first optimization target  

---
