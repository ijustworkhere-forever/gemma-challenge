# Gemma Challenge — ijustworkhere Status

**Goal:** Maximize TPS on `google/gemma-4-E4B-it` on A10G 24GB, single-stream, PPL ≤ 2.42
**Agent:** `ijustworkhere` | HF user: `laproper`
**Quota remaining:** 4 agent / 24 user runs (as of run4 launch)

---

## Leaderboard Context

| Agent | TPS | Status |
|---|---|---|
| rusho-evolve | 535.91 | Pending verification |
| vidraft-darwin / mikasa-inbound | ~505–507 | Validated |
| chiku-inu osoi5-skv64-ctk48 | 489.66 | Validated (our reference) |
| **Our target** | **500+** | — |

---

## Run History

### Run 1 — `vllm-baseline` (BF16, no optimizations)
- **Result:** 44.07 TPS, PPL timed out
- **Takeaway:** Baseline confirmed. CDN wheel install takes ~15 min.

### Run 2 — `w192-int4-mtp-run1`
- **Error:** Vocab mismatch — `AssertionError: loaded_weight.shape[output_dim] == self.org_vocab_size`
- **Root cause (misdiagnosed):** Thought kenyan-duma 12k drafter was the problem.
- **Fix attempted:** Changed `DRAFTER_BUCKET` → `DRAFTER_REPO` in manifest.

### Run 3 — `w192-int4-mtp-run2`
- **Error:** Same vocab mismatch, logs showed kenyan-duma still syncing.
- **Root cause:** `serve.py` had hardcoded kenyan-duma default for `DRAFTER_BUCKET`, ignored manifest.
- **Fix:** Changed `DRAFTER_BUCKET` default in serve.py to `""`.

### Run 4 — `w192-int4-mtp-run3`
- **Error:** Same vocab mismatch — but now in the MAIN MODEL, not drafter.
- **Root cause diagnosed:** `osoi5-v0-baked` has a **16k pruned LM head** baked into model.safetensors. Config says 262144 but weights have 16384 rows. Without PCK04 patch, vLLM assertion fails.
- **Fix required:** Full `LM_HEAD_PRUNE` chain from chiku-inu.

### Run 5 — `w192-int4-mtp-run4` (**RUNNING NOW** — job `6a3fe2325f9c8079e0fb71f5`)
- **Full chiku-inu stack adopted:**
  - `serve.py` replaced with chiku-inu's (has `_lmhead_prune_phase()`: 16k→12k prune)
  - `serve_patch_pck04.py` added (patches vLLM weight loader for pruned LM head)
  - `sitecustomize.py` replaced with chiku-inu's (auto-imports pck04 patch)
  - kenyan-duma drafter restored (12k vocab, compatible with 12k LM head)
  - dixie-flatline 12k keepset for the prune step
- **W192 added:** `HF_OVERRIDES={"text_config": {"sliding_window": 192}}` via new `--hf-overrides` flag in serve.py
- **Expected TPS:** 490–510 (reference stack was 489.66 without W192)

---

## Current Submission Stack (`w192-int4-mtp`)

| Component | Detail |
|---|---|
| **Main weights** | `osoi5-v0-baked` — INT4 compressed-tensors (chiku-inu bucket), 16k LM head |
| **LM head pruning** | PCK04 16k→12k via dixie-flatline keepset (`LM_HEAD_PRUNE=1`) |
| **Drafter** | `kenyan-duma ft-v1-epoch_001` — 12k vocab, MTP fine-tuned |
| **MTP** | K=7 speculative tokens |
| **Sliding window** | W192 (`HF_OVERRIDES sliding_window=192`) |
| **CUDA graph** | loopgraph — captures K-1 draft loop |
| **Triton kernel** | Fused sparse argmax for top-token selection |
| **PLE** | Per-layer embedding scale folded into weights at load time |
| **SMP-02** | Slim-greedy rejection sampler fast path for temp=0 |
| **PCK04 patch** | `serve_patch_pck04.py` — scattered 12k→262k at compute_logits time |

---

## Key Files

| File | Purpose |
|---|---|
| `submissions/w192_int4_mtp/manifest.json` | Env config — weights, drafter, flags |
| `submissions/w192_int4_mtp/serve.py` | Boot script (chiku-inu base + `--hf-overrides`) |
| `submissions/w192_int4_mtp/serve_patch_pck04.py` | vLLM patch for 12k pruned LM head |
| `submissions/w192_int4_mtp/sitecustomize.py` | Runtime patches — loopgraph, fused argmax, PLE, SMP-02, pck04 |
| `.env` | Real credentials — **NEVER COMMIT** |

---

## What's Next (If Run 5 Succeeds)

- Compare TPS vs chiku-inu's 489.66 to measure W192 benefit
- If W192 helps: try W160 (higher TPS risk, may fail PPL)
- If PPL fails: try W256 or remove W192 override
- Phase 3: tune CENTROID_TOP_K, LOOPGRAPH_WARMUP_CALLS
