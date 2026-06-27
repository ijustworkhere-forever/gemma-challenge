# REFERENCES.md

> A curated list of papers, repositories, articles, and tools relevant to the Gemma Challenge.
>
> Every entry should answer:
>
> - What is it?
> - Why does it matter?
> - How can it help us?

---

# Official Challenge

## Gemma Challenge

Status: ⭐ Required Reading

Purpose

The official competition documentation.

Read Before

Everything.

Notes

- Review evaluation methodology
- Study submission requirements
- Understand PPL constraints
- Keep up with rule changes

---

# Core Papers

## Attention Is All You Need

Importance

★★★★★

Topics

- Transformer architecture
- Self-attention
- Decoder design

Potential Value

Understanding attention bottlenecks.

---

## FlashAttention

Importance

★★★★★

Topics

- IO-aware attention
- Memory optimization

Potential Value

Essential for understanding modern attention kernels.

Questions

- Can the implementation be tuned?
- Are tile sizes optimal?

---

## FlashAttention-2

Importance

★★★★★

Topics

- Better work partitioning
- Parallelism
- Occupancy improvements

Potential Value

May expose optimization opportunities beyond defaults.

---

## FlashAttention-3

Importance

★★★★☆

Topics

- Hopper optimizations
- New scheduling

Potential Value

Ideas may transfer even if hardware differs.

---

## PagedAttention (vLLM)

Importance

★★★★★

Topics

- KV cache management
- Memory fragmentation

Potential Value

Likely one of the biggest optimization areas.

Questions

- Can cache locality improve?
- Better page sizing?

---

## CUDA Graphs

Importance

★★★★★

Topics

- Kernel launch reduction

Potential Value

Decode optimization.

---

## CUTLASS Documentation

Importance

★★★★☆

Topics

- GEMM
- Tensor cores
- Kernel templates

Potential Value

Reference when writing custom kernels.

---

## Triton Language

Importance

★★★★★

Topics

- GPU kernel development

Potential Value

Primary language for custom kernels.

---

# Repositories

---

## vLLM

Purpose

Current baseline inference engine.

Investigate

- Scheduler
- KV cache
- Decode loop
- Sampling

---

## TensorRT-LLM

Purpose

Performance comparison.

Focus

- Kernel fusion
- Quantization
- CUDA Graphs

---

## SGLang

Purpose

Alternative serving architecture.

Focus

- Scheduler
- Continuous batching

---

## llama.cpp

Purpose

Learn from highly optimized inference implementation.

Focus

- Sampling
- Quantization
- Memory layout

---

## Gemma PyTorch

Purpose

Official model implementation.

Investigate

- Decoder
- Attention
- Layer implementations

---

# GPU References

Read

- CUDA Programming Guide
- CUDA Best Practices
- Nsight Systems
- Nsight Compute
- NVIDIA Occupancy Calculator

---

# Performance Engineering

Read

- Brendan Gregg
- Systems Performance
- Flame Graphs
- Linux Perf
- Roofline Analysis

---

# Profiling

Tools

- Nsight Systems
- Nsight Compute
- Torch Profiler
- perf
- nvidia-smi
- CUPTI

Questions

- Where is time spent?
- Compute or memory bound?
- Kernel launch overhead?
- Tensor core utilization?

---

# Benchmarking

Rules

- Same benchmark
- Same settings
- Same hardware
- Multiple runs
- Record averages
- Record variance

---

# Interesting GitHub Issues

| Repository | Topic | Status |
|------------|-------|--------|
| vLLM | | |
| FlashAttention | | |
| Triton | | |
| CUTLASS | | |

---

# Things Worth Watching

- New Triton releases
- FlashAttention updates
- TensorRT-LLM releases
- CUDA releases
- PyTorch releases
- Gemma Challenge leaderboard

---

# Research Questions

- Is decode compute bound?
- Is decode memory bound?
- Can KV cache locality improve?
- Is continuous batching optimal?
- Are launch overheads measurable?
- Is speculative decoding allowed?
- What causes throughput collapse at long context?

---

# Personal Reading Queue

- [ ] FlashAttention paper
- [ ] FlashAttention-2 paper
- [ ] PagedAttention paper
- [ ] CUDA Graphs documentation
- [ ] CUTLASS tutorials
- [ ] Triton tutorials
- [ ] vLLM architecture
- [ ] TensorRT-LLM architecture
- [ ] Gemma technical report
