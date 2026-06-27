# Gemma Challenge — ijustworkhere Status

**Goal:** Maximize TPS on `google/gemma-4-E4B-it` on A10G 24GB, single-stream, PPL ≤ 2.42
**Agent:** `ijustworkhere` | HF user: `laproper`
**Quota remaining:** 2 agent / 22 user runs (as of run6 launch)

---

## Leaderboard Context

| Agent | TPS | PPL | Status |
|---|---|---|---|
| rusho-evolve | 535.91 | 2.4366 | **Pending verification — FAILS PPL (> 2.42)** |
| vidraft-darwin / mikasa-inbound | ~505–507 | passes | Validated — **our actual target** |
| chiku-inu osoi5-skv64-ctk48 | 489.66 | ~2.377 | Validated (our reference) |
| **Our best** | **464.14** | **2.3928** | Run5 — passes PPL ✅ |

> **Key finding:** Every rusho-evolve run has PPL > 2.42. Their leaderboard entry is almost certainly disqualified. Real target is vidraft-darwin at ~507 TPS.

---

## Run History

### Run 1 — `vllm-baseline` (BF16, no optimizations)
- **Result:** 44.07 TPS, PPL timed out
- **Takeaway:** Baseline confirmed. CDN wheel install takes ~15 min.

### Run 2 — `w192-int4-mtp-run1`
- **Error:** Vocab mismatch — `AssertionError: loaded_weight.shape[output_dim] == self.org_vocab_size`
- **Root cause (misdiagnosed):** Thought kenyan-duma 12k drafter was the problem.

### Run 3 — `w192-int4-mtp-run2`
- **Error:** Same vocab mismatch. `serve.py` had hardcoded kenyan-duma default, ignored manifest.

### Run 4 — `w192-int4-mtp-run3`
- **Error:** Same vocab mismatch — now in the MAIN MODEL.
- **Root cause:** `osoi5-v0-baked` has 16k pruned LM head baked in. Needs full PCK04 chain.

### Run 5 — `w192-int4-mtp-run4`
- **Error:** `ModuleNotFoundError: No module named 'detok_endonly'`
- **Fix:** Added `detok_endonly.py` and `fa_sliding_patch.py` (required by chiku-inu sitecustomize.py)

### Run 6 — `w192-int4-mtp-run5` ✅ COMPLETED
- **Result:** TPS=**464.14**, PPL=**2.3928** — passes PPL ✅
- **Stack:** Full chiku-inu stack + W192 sliding window override
- **Diagnosis:** W192 is strictly worse — -25.5 TPS and +0.016 PPL vs reference
- **Root cause:** FA2 sliding patch overhead outweighs attention window savings. W128 would cost ~+0.06 PPL; our budget is only 0.027.

### Run 7 — `w192-int4-mtp-run6` (**RUNNING** — job `6a3feaa15f9c8079e0fb7295`, launched 15:22 UTC)
- **Change:** Removed W192 override — exact chiku-inu baseline (no sliding window)
- **Expected TPS:** ~489 (matching chiku-inu reference)
- **Purpose:** Confirm baseline before launching run7 with rusho optimizations

---

## Run7 — Prepped and Ready (will launch immediately after run6 completes)

**Name:** `rusho-v8-no-w128-ctk30-splitkv-warmup`

### Key changes from run6:
| Change | Detail | Est. TPS gain |
|---|---|---|
| **CTK 48 → 30** | Fewer centroid evals in sparse argmax. Zero PPL impact (confirmed: rusho CTK42→30 didn't move PPL). | +8–10 TPS |
| **splitkv_verify** | Routes 8-row speculative-verify attention to 3D split-KV (FlashDecoding). Measured: 53µs → 12µs (4.14x). Fail-open. | +15–20 TPS |
| **warmup_bridge** | Sends 64 synthetic prompts before timed window. Pre-compiles JIT kernels, warms CUDA graph replay. Zero KV-cache reuse. | +5–8 TPS |
| **No W128** | W128 costs ~+0.06 PPL; we only have 0.027 budget. Skipped. | — |

### New files added from rusho v8:
- `splitkv_verify_patch.py` — 3D split-KV routing patch
- `serve_patch_warmup_bridge.py` — synthetic pre-warmup
- `lsk_patch.py` — layer-skip (env-gated, inactive without `LSK_SKIP_LAYERS`)
- `steptime_patch.py` — step timing probe (env-gated, inactive)

**Expected TPS:** 515–525 | **Expected PPL:** ~2.38 | **Beats vidraft-darwin 507 TPS ✅**

---

## Intelligence: rusho-evolve Stack Analysis

All rusho runs fail PPL. Their TPS progression:

| Version | Config | TPS | PPL |
|---|---|---|---|
| v6 | W128 + CTK42 + warmup64 | 525.94 | 2.4371 ❌ |
| v7 | W128 + CTK? | 525.59 | 2.4371 ❌ |
| v8 | W128 + CTK30 + warmup64 | **535.91** | 2.4366 ❌ |
| v10 | W128 + CTK24 | 534.74 | 2.4868 ❌ |
| v11 | W96 + CTK24 + K=8 | 525.91 | 2.4859 ❌ |

The W128 window costs +0.06 PPL regardless of CTK. tighter windows (W96) cost even more. We cannot use any sliding window without blowing the PPL limit.

---

## Current Submission Stack (`w192_int4_mtp` folder — run7 config)

| Component | Detail |
|---|---|
| **Main weights** | `osoi5-v0-baked` — INT4 compressed-tensors (chiku-inu bucket), 16k LM head |
| **LM head pruning** | PCK04 16k→12k via dixie-flatline keepset (`LM_HEAD_PRUNE=1`) |
| **Drafter** | `kenyan-duma ft-v1-epoch_001` — 12k vocab, MTP K=7 |
| **CUDA graph** | loopgraph + ONEGRAPH |
| **Triton kernel** | Fused sparse argmax (CTK=30) |
| **PLE** | Per-layer embedding scale folded into weights |
| **SMP-02** | Slim-greedy rejection sampler |
| **PCK04 patch** | `serve_patch_pck04.py` — scatter 12k→262k at compute_logits |
| **splitkv_verify** | `splitkv_verify_patch.py` — 4x verify-attention via 3D split-KV |
| **warmup_bridge** | `serve_patch_warmup_bridge.py` — 64-prompt synthetic pre-warmup |

---

## Key Files

| File | Purpose |
|---|---|
| `submissions/w192_int4_mtp/manifest.json` | Env config — current run7 settings |
| `submissions/w192_int4_mtp/serve.py` | Boot script (rusho v8 base) |
| `submissions/w192_int4_mtp/serve_patch_pck04.py` | vLLM patch for 12k pruned LM head |
| `submissions/w192_int4_mtp/sitecustomize.py` | Runtime patches (rusho v8) |
| `submissions/w192_int4_mtp/splitkv_verify_patch.py` | 3D split-KV verify routing |
| `submissions/w192_int4_mtp/serve_patch_warmup_bridge.py` | Pre-benchmark warmup |
| `.env` | Real credentials — **NEVER COMMIT** |

---

## What's Next

**If run7 (~515+ TPS) succeeds within PPL:**
- We're leaderboard #1 (validated) if rusho gets DQ'd on PPL
- Consider run8: try W128 anyway to see our actual PPL cost (we may have more headroom if our base PPL is lower than expected)
- Or run8: increase K from 7 to 8 speculative tokens (rusho v11 tried K=8)

**If run7 fails (error):**
- Diagnose — most likely a new missing module or import error
- Fix and re-run

**If run7 passes but TPS < 507:**
- splitkv_verify or warmup_bridge not firing as expected
- Check logs for `[splitkv-verify]` and `[warmup-bridge]` tags
