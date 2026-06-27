# IDEAS.md

> Working notebook for optimization ideas, hypotheses, and future experiments.
>
> Nothing should be considered "truth" until benchmarked.
>
> Status Legend:
>
> - ⏳ Not Started
> - 🔬 Investigating
> - ✅ Proven
> - ❌ Rejected
> - 🚀 Implemented
>
> Priority:
>
> - P0 = Critical
> - P1 = High
> - P2 = Medium
> - P3 = Nice to Have

---

# Current Baseline

| Metric | Value |
|----------|------|
| TPS | |
| TTFT | |
| PPL | |
| GPU Utilization | |
| VRAM | |
| Engine | |

---

# P0 Ideas

## KV Cache Investigation

Status: ⏳

Priority: P0

Hypothesis

KV cache memory layout is limiting decode throughput due to cache misses.

Questions

- Are reads contiguous?
- Are writes coalesced?
- Can layout be improved?
- Can paging be reduced?
- Is fragmentation occurring?

Experiments

- Compare paged vs contiguous
- Measure L2 hit rate
- Profile memory bandwidth

Expected Gain

5–20%

Notes

---

## CUDA Graphs

Status: ⏳

Priority: P0

Hypothesis

Kernel launch overhead becomes significant during decode.

Experiments

- Capture decode graph
- Replay graph
- Compare TPS

Questions

- Static shapes?
- Dynamic shapes?
- Graph recreation cost?

Expected Gain

5–10%

Notes

---

## Scheduler Improvements

Status: ⏳

Priority: P0

Hypothesis

Current scheduler leaves GPU idle between requests.

Investigate

- Continuous batching
- Dynamic batching
- Request ordering
- Decode grouping

Metrics

- GPU idle time
- Queue wait time
- TPS

---

# P1 Ideas

## FlashAttention

Status: ⏳

Questions

- FA2
- FA3
- Tile tuning
- Occupancy

Expected Gain

3–8%

---

## Kernel Fusion

Status: ⏳

Candidates

- RMSNorm
- Rotary Embeddings
- Sampling
- LayerNorm
- Activation

Expected Gain

2–6%

---

## Quantization

Status: ⏳

Test

- FP16
- BF16
- FP8
- INT8

Watch

- PPL
- Accuracy
- TPS

---

## Torch Compile

Status: ⏳

Evaluate

- torch.compile()

Compare

- compile time
- decode speed
- memory usage

---

# P2 Ideas

## CPU Profiling

Potential Bottlenecks

- Tokenizer
- Python GIL
- Serialization
- Async Runtime
- HTTP Server

Questions

- Can tokenizer run in parallel?
- Can sampling move to C++?
- Is Python overhead measurable?

---

## Memory Allocator

Investigate

- Fragmentation
- Allocation frequency
- CUDA allocator tuning

---

## Tensor Core Utilization

Goal

95%+

Questions

- Are tensor cores saturated?
- Mixed precision opportunities?
- Register pressure?

---

## Sampling Optimization

Investigate

- Top-K
- Top-P
- Temperature
- Argmax path

Can this be fused?

---

# P3 Ideas

## Multi-Stream Decode

Idea

Overlap

- Host work
- Device work
- Copies

Potential

Small gain

---

## Async Tokenizer

Idea

Tokenize while previous request is decoding.

---

## Output Pipeline

Investigate

- Serialization
- JSON generation
- HTTP chunking

---

# Research Topics

Read

- FlashAttention papers
- PagedAttention paper
- TensorRT-LLM implementation
- vLLM scheduler
- SGLang scheduler
- CUTLASS documentation
- Triton language docs

---

# Interesting Metrics

Collect

- SM Occupancy
- Warp Efficiency
- Tensor Core Usage
- L2 Cache Hit Rate
- DRAM Throughput
- Register Usage
- Shared Memory
- Kernel Launch Count
- CPU Wait Time
- GPU Wait Time

---

# Optimization Wishlist

- Zero CPU bottlenecks
- >95% GPU utilization
- Zero unnecessary kernel launches
- Fully asynchronous pipeline
- Continuous batching
- CUDA Graph replay
- Efficient KV cache
- Minimal memory fragmentation

---

# Crazy Ideas 💡

These may be terrible.

Write them down anyway.

---

### Idea

Status:

Description:

Reason it might work:

Why it probably won't:

Benchmark Result:

---

### Idea

Status:

Description:

Reason it might work:

Why it probably won't:

Benchmark Result:

---

### Idea

Status:

Description:

Reason it might work:

Why it probably won't:

Benchmark Result:

---

# Experiments to Revisit

Ideas that showed promise but weren't finished.

| Experiment | Reason |
|------------|--------|
| | |
| | |
| | |

---

# Dead Ends

Document failed ideas to avoid repeating them.

| Idea | Reason Rejected |
|------|-----------------|
| | |
| | |

---

# Performance Timeline

| Date | TPS | PPL | Notes |
|------|-----|-----|------|
| | | | |

---

# Biggest Unknowns

These are questions we currently cannot answer.

- Why is decode slower than expected?
- Is the scheduler saturating the GPU?
- Is memory bandwidth the primary bottleneck?
- Is attention compute-bound or memory-bound?
- What limits throughput on long contexts?
- Are Python components visible in the critical path?

---

# Winning Opportunities

Potential areas where competitors may overlook optimizations.

- Better scheduler heuristics
- KV cache locality
- CUDA Graph replay
- Kernel fusion
- Smarter batching
- Memory allocator tuning
- Decode pipeline overlap
- CPU runtime optimization
- Custom Triton kernels
- Better profiling methodology

---

# Rules

- Every idea gets benchmarked.
- Never optimize without profiling.
- Never trust a single benchmark run.
- Record failures—they're just as valuable as successes.
- Keep hypotheses separate from verified results.
- Revisit old ideas after major architectural changes.
