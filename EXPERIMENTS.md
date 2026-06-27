# EXPERIMENTS.md

> Every optimization attempt gets recorded here.
>
> Goal:
> Build a reproducible history of what was tried, what worked, and what didn't.
>
> Rule #1:
> Never trust memory.
>
> Rule #2:
> If it wasn't benchmarked, it didn't happen.

---

# Experiment Template

---

## Experiment ID

EXP-0001

---

### Date

YYYY-MM-DD

---

### Git Commit

```
abcdef123
```

---

### Author

Name

---

### Branch

feature/example

---

### Objective

Describe exactly what you're trying to improve.

Example:

Increase decode throughput by reducing CUDA launch overhead.

---

### Hypothesis

Example:

CUDA Graph replay will reduce kernel launch latency enough to improve TPS by 5–10%.

---

### Files Changed

```
engine/decode.py

scheduler.py

kv_cache.py
```

---

### Configuration

Engine:

Model:

Precision:

Batch Size:

Max Context:

GPU:

Driver:

CUDA Version:

Torch Version:

---

### Benchmark Dataset

Official Benchmark

or

Custom Dataset

---

### Before

TPS:

TTFT:

Latency:

GPU Utilization:

VRAM:

PPL:

---

### After

TPS:

TTFT:

Latency:

GPU Utilization:

VRAM:

PPL:

---

### Difference

TPS

+

%

Latency

-

%

GPU Utilization

+

%

Memory

+

or

-

---

### Profiler Findings

Interesting observations.

Example

- Decode kernel launch reduced
- Higher occupancy
- L2 hit rate increased
- Python overhead unchanged

---

### Unexpected Behavior

Anything surprising?

---

### Problems

Did anything regress?

---

### Screenshots

Profiler

Nsight

Charts

Links

---

### Decision

Choose one

✅ Keep

❌ Revert

🔬 Needs More Testing

---

### Lessons Learned

What should future you remember?

---

# Experiment Index

| ID | Status | TPS Change | Decision |
|----|----------|------------|----------|
| EXP-0001 | | | |
| EXP-0002 | | | |
| EXP-0003 | | | |
| EXP-0004 | | | |

---

# Best Improvements So Far

| Rank | Experiment | TPS Gain |
|-------|------------|-----------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

# Biggest Failures

Failures are valuable.

| Experiment | Reason |
|------------|--------|
| | |
| | |

---

# Things To Retry Later

Sometimes improvements fail because another bottleneck exists.

List them here.

- Example
- Example
- Example
