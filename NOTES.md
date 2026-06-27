# NOTES.md

> Daily engineering journal.
>
> This is intentionally informal.
>
> Record observations, theories, profiler screenshots, dead ends, and random ideas.
>
> Think of this as your lab notebook.

---

# 2026-06-26

## Project Started

Repository created.

Initial documentation added:

- PLAN.md
- IDEAS.md
- EXPERIMENTS.md
- DECISIONS.md
- REFERENCES.md
- NOTES.md

Primary Goal

Understand where inference time is spent before making any optimizations.

---

## Initial Thoughts

Don't optimize blindly.

Profiling should drive every decision.

Most competitors will likely tweak configuration values.

Instead, understand the architecture.

---

## Immediate Tasks

- Clone baseline implementation
- Run benchmark
- Record baseline
- Capture Nsight trace
- Capture Torch profiler
- Read challenge rules completely

---

## Questions

How much time is spent in

- Prefill?
- Decode?
- Sampling?
- Python?
- CUDA launches?
- KV cache?

---

## Things To Watch

GPU utilization.

If utilization is below 95%, ask why.

---

## Interesting Observations

None yet.

---

## Potential Risks

- Overfitting benchmark
- Breaking PPL
- Chasing noisy benchmarks
- Optimizing wrong bottleneck

---

## Reminder

Measure.

Don't assume.

---

# Scratchpad

Random ideas belong here before moving to IDEAS.md.

---

- Could decode be replayed using CUDA Graphs?
- Should scheduler become priority-aware?
- Profile tokenizer separately.
- Compare long vs short context throughput.
- Measure effect of batch size.
- Measure context length scaling.
- Investigate CPU wakeups.

---

# Profiler Checklist

Before every optimization:

- Capture Nsight Systems
- Capture Nsight Compute
- Save traces
- Record benchmark
- Commit code

---

# Daily Summary Template

## Date

### What Changed

-

### What Was Learned

-

### Biggest Bottleneck

-

### Biggest Surprise

-

### Tomorrow

-

### New Ideas

-
