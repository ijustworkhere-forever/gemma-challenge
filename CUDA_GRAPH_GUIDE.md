# CUDA_GRAPH_GUIDE.md

> Practical guide to applying CUDA Graphs to the Gemma decode loop.
>
> Goal:
> Eliminate kernel launch overhead during autoregressive decoding.

---

# 1. Why CUDA Graphs Matter

In decode loops:

- Each token = multiple kernel launches
- Launch overhead accumulates massively
- GPU often waits on CPU dispatch

CUDA Graphs fix this by:

> Recording execution once → replaying without CPU involvement

---

# 2. Best Candidate: Decode Loop

CUDA Graphs apply ONLY when:

- Execution graph is stable
- Shapes are consistent
- No dynamic branching per token

Decode loop satisfies this.

---

# 3. Standard Decode Loop

Without CUDA Graphs:

```
for token in sequence:
    attention()
    mlp()
    norm()
    sampling()
    kv_update()
```

Each step = kernel launch overhead

---

# 4. CUDA Graph Version

## Step 1: Warmup Run

You must run once normally:

- Allocate buffers
- Initialize KV cache
- Stabilize shapes

---

## Step 2: Capture Graph

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// decode step
run_attention();
run_mlp();
run_norm();
run_kv_update();

cudaStreamEndCapture(stream, &graph);
```

---

## Step 3: Instantiate Graph

```cpp
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
```

---

## Step 4: Replay per token

```cpp
cudaGraphLaunch(graphExec, stream);
```

---

# 5. What Must Be Static

CUDA Graphs REQUIRE:

- Fixed tensor shapes
- Fixed batch size (or padded batches)
- No Python branching inside graph
- Stable memory allocations

---

# 6. What Breaks CUDA Graphs

These will invalidate capture:

- Dynamic sequence length changes
- Dynamic batching changes per step
- Memory allocation during decode
- CPU-side branching per token

---

# 7. Integration Strategy (Real Systems Approach)

## Option A: Full Decode Graph (Best)

Capture entire decode step per token.

Pros:
- Maximum speedup
- Minimal CPU involvement

Cons:
- Requires strict shape control

---

## Option B: Partial Graph

Only capture:

- Attention
- MLP
- Norm

Keep sampling + scheduling outside graph.

Pros:
- Easier to implement

Cons:
- Slight CPU overhead remains

---

# 8. Expected Gains

| Optimization | Gain |
|-------------|------|
| Remove kernel launches | 5–15% |
| Reduce CPU sync | additional 2–5% |
| Better batching stability | 1–5% |

---

# 9. Profiling Before/After

## Before CUDA Graphs

- Many kernel launches per token
- CPU-GPU synchronization per step
- High latency variance

---

## After CUDA Graphs

- Single launch per decode step
- Near-zero CPU overhead per token
- Stable TPS

---

# 10. Common Mistakes

❌ Capturing graph with dynamic shapes  
❌ Reallocating memory inside loop  
❌ Including Python logic inside graph  
❌ Changing batch size mid-stream  

---

# 11. Debug Checklist

If CUDA Graphs don't work:

- Did shapes stay constant?
- Did memory allocations occur inside loop?
- Did CPU touch GPU buffers mid-run?
- Are kernels actually inside graph?

---

# 12. Integration Points in This Repo

Connect to:

- `KERNEL_MAP.md` → identify which kernels to capture
- `ARCHITECTURE.md` → locate decode loop boundary
- `PROFILING_PLAYBOOK.md` → validate improvements
- `EXPERIMENTS.md` → track gains

---

# 13. Final Rule

CUDA Graphs are NOT an optimization.

They are a *pipeline transformation*.

If applied correctly:

> decode becomes a replay loop instead of execution loop

---

# 14. Next Step

👉 Profile decode loop kernel launches  
👉 Confirm shape stability  
👉 Capture first graph experiment  
👉 Measure TPS delta  

---
