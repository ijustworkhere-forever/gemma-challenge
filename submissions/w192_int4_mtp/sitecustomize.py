"""Patch vLLM Gemma4 MTP drafting with a CUDA graph loop replay.

This file is loaded by the vLLM child process through PYTHONPATH. It intentionally
does not patch PLE. Pupa's serve.py patches PLE textfast and scale-folds through
the installed vLLM source so the fold can be verified fail-closed at load time.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from copy import copy
from typing import Any


LOOPGRAPH_TARGET = "vllm.v1.spec_decode.gemma4"
RUNNER_TARGET = "vllm.v1.worker.gpu_model_runner"
TOP_TOKEN_TARGET = "vllm.model_executor.models.gemma4_mtp"
LOOPGRAPH_WARMUP_CALLS = int(os.environ.get("LOOPGRAPH_WARMUP_CALLS", "48"))
LOOPGRAPH_REQUIRE_CAPTURE = os.environ.get("LOOPGRAPH_REQUIRE_CAPTURE") == "1"
LOOPGRAPH_PINGPONG_SLOTS = max(1, int(os.environ.get("LOOPGRAPH_PINGPONG_SLOTS", "1")))
FUSED_SPARSE_ARGMAX = os.environ.get("FUSED_SPARSE_ARGMAX", "1") == "1"
FUSED_SPARSE_ARGMAX_REQUIRE = os.environ.get("FUSED_SPARSE_ARGMAX_REQUIRE") == "1"
FUSED_SPARSE_ARGMAX_BLOCK = int(os.environ.get("FUSED_SPARSE_ARGMAX_BLOCK", "16"))
_FUSED_SPARSE_ARGMAX_KERNELS: Any | None = None
_LOOPGRAPH_SLOT_EVENTS_BY_PTR: dict[int, Any] = {}
_LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR: dict[int, bool] = {}


def _call_base_propose(base_propose: Any, self: Any, kwargs: dict[str, Any]) -> Any:
    return base_propose(self, **kwargs)


def _build_static_buffers(self: Any, state: dict[str, Any], cad: Any) -> None:
    import torch

    device = self.device
    token_count = self.num_speculative_tokens
    state["outputs"] = [
        torch.zeros((1, token_count), dtype=torch.int64, device=device)
        for _ in range(LOOPGRAPH_PINGPONG_SLOTS)
    ]
    state["out"] = state["outputs"][0]
    state["next_slot"] = 0
    state["_pupa_loopgraph_slot_events"] = [
        torch.cuda.Event(blocking=False) for _ in state["outputs"]
    ]
    for output, event in zip(
        state["outputs"], state["_pupa_loopgraph_slot_events"], strict=True
    ):
        _LOOPGRAPH_SLOT_EVENTS_BY_PTR[output.data_ptr()] = event
        _LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR[output.data_ptr()] = False
    state["seq_lens"] = torch.zeros_like(cad.seq_lens[:1])
    state["block_tables"] = {}

    static_cad = copy(cad)
    static_cad.seq_lens = state["seq_lens"]
    static_cad.num_actual_tokens = 1
    static_cad.max_query_len = 1
    static_cad.max_seq_len = self.max_model_len
    static_cad.slot_mapping = self._slot_mapping_buffer[:1]
    static_cad.query_start_loc = self.arange[:2]

    per_layer_metadata = {}
    for group in self.draft_attn_groups:
        group_id = group.kv_cache_group_id
        source = self._per_group_block_tables.get(group_id, cad.block_table_tensor)[:1]
        block_size = group.get_metadata_builder().kv_cache_spec.block_size
        width = max(source.shape[1], -(-self.max_model_len // block_size))
        static_block_table = torch.zeros((1, width), dtype=source.dtype, device=device)
        state["block_tables"][group_id] = static_block_table

        group_cad = copy(static_cad)
        group_cad.block_table_tensor = static_block_table
        metadata = group.get_metadata_builder().build_for_drafting(
            common_attn_metadata=group_cad,
            draft_index=1,
        )
        for layer_name in group.layer_names:
            per_layer_metadata[layer_name] = metadata
    state["metadata"] = per_layer_metadata


def _refresh_static_buffers(self: Any, state: dict[str, Any], cad: Any) -> None:
    state["seq_lens"].copy_(cad.seq_lens[:1])
    for group_id, static_block_table in state["block_tables"].items():
        source = self._per_group_block_tables.get(group_id, cad.block_table_tensor)[:1]
        width = min(source.shape[1], static_block_table.shape[1])
        static_block_table[:, :width].copy_(source[:, :width])


def _run_graph_body(self: Any, state: dict[str, Any]) -> None:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context

    token_count = self.num_speculative_tokens
    output = state["out"]
    with set_forward_context(
        state["metadata"],
        self.vllm_config,
        num_tokens=1,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping=self._get_slot_mapping(1),
    ):
        for index in range(token_count - 1):
            self.input_ids[:1].copy_(output[0, index : index + 1])
            last_hidden, backbone_hidden = self.model(
                input_ids=self.input_ids[:1],
                positions=self._get_positions(1),
                inputs_embeds=None,
                hidden_states=self.hidden_states[:1],
            )
            self.hidden_states[:1].copy_(backbone_hidden[:1])
            token = self.model.get_top_tokens(last_hidden[:1])
            output[0, index + 1 : index + 2].copy_(token)


def _select_loopgraph_output_slot(state: dict[str, Any]) -> Any:
    import torch

    outputs = state.get("outputs")
    if not outputs:
        return state["out"]

    slot_index = int(state.get("next_slot", 0))
    output_slot = outputs[slot_index]
    event = _LOOPGRAPH_SLOT_EVENTS_BY_PTR.get(output_slot.data_ptr())
    event_recorded = _LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR.get(
        output_slot.data_ptr(), False
    )
    if event is not None and event_recorded:
        torch.cuda.current_stream().wait_event(event)

    state["out"] = output_slot
    state["active_slot"] = slot_index
    state["next_slot"] = (slot_index + 1) % len(outputs)
    return output_slot


def _prime_loopgraph_outputs(state: dict[str, Any], first_token: Any) -> None:
    for output in state.get("outputs", [state["out"]]):
        output[0, 0:1].copy_(first_token)


def _capture_graph(self: Any, state: dict[str, Any]) -> None:
    import torch

    graphs = []
    for output in state.get("outputs", [state["out"]]):
        state["out"] = output
        for _ in range(2):
            _run_graph_body(self, state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_graph_body(self, state)
        graphs.append(graph)
    state["graphs"] = graphs
    state["graph"] = graphs[0]
    state["out"] = state.get("outputs", [state["out"]])[0]


def _is_loopgraph_eligible(self: Any, state: dict[str, Any], cad: Any) -> bool:
    return (
        not state["failed"]
        and self.num_speculative_tokens > 1
        and not self.parallel_drafting
        and not self._enable_probabilistic_draft_probs
        and not self.supports_mm_inputs
        and not self.uses_mrope
        and self.constant_draft_positions
        and cad.batch_size() == 1
    )


def _raise_or_fallback(exc: Exception) -> None:
    if LOOPGRAPH_REQUIRE_CAPTURE:
        raise RuntimeError("LOOPGRAPH_REQUIRE_CAPTURE=1 but capture failed") from exc


def _apply_loopgraph_patch(module: Any) -> None:
    import torch

    from vllm.forward_context import set_forward_context

    proposer_cls = module.Gemma4Proposer
    base_propose = proposer_cls.propose

    def propose(
        self: Any,
        target_token_ids: Any,
        target_positions: Any,
        target_hidden_states: Any,
        next_token_ids: Any,
        token_indices_to_sample: Any,
        common_attn_metadata: Any,
        sampling_metadata: Any,
        mm_embed_inputs: Any = None,
        num_rejected_tokens_gpu: Any = None,
        slot_mappings: Any = None,
    ) -> Any:
        kwargs = {
            "target_token_ids": target_token_ids,
            "target_positions": target_positions,
            "target_hidden_states": target_hidden_states,
            "next_token_ids": next_token_ids,
            "token_indices_to_sample": token_indices_to_sample,
            "common_attn_metadata": common_attn_metadata,
            "sampling_metadata": sampling_metadata,
            "mm_embed_inputs": mm_embed_inputs,
            "num_rejected_tokens_gpu": num_rejected_tokens_gpu,
            "slot_mappings": slot_mappings,
        }
        state = self.__dict__.setdefault(
            "_pupa_loopgraph",
            {"calls": 0, "graph": None, "failed": False},
        )
        if not _is_loopgraph_eligible(self, state, common_attn_metadata):
            return _call_base_propose(base_propose, self, kwargs)

        state["calls"] += 1
        if state["graph"] is None and state["calls"] <= LOOPGRAPH_WARMUP_CALLS:
            return _call_base_propose(base_propose, self, kwargs)

        self._last_draft_probs = None
        token_count = self.num_speculative_tokens
        num_tokens, token_indices_to_sample, cad = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        _, per_layer_metadata = self.build_per_group_and_layer_attn_metadata(cad)
        cg_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        model_kwargs, slot_map_size = self.build_model_inputs_first_pass(
            num_tokens,
            num_input_tokens,
            mm_embed_inputs,
        )
        with set_forward_context(
            per_layer_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cg_mode,
            slot_mapping=self._get_slot_mapping(slot_map_size, cad.slot_mapping),
        ):
            last_hidden, hidden = self.model(**model_kwargs)

        sample_hidden = last_hidden[token_indices_to_sample]
        positions = self.positions[token_indices_to_sample]
        first_hidden = hidden[token_indices_to_sample]
        self.positions[:1] = positions
        first_token, _ = self._sample_draft_tokens(sample_hidden, sampling_metadata)

        cad.num_actual_tokens = 1
        cad.max_query_len = 1
        cad.query_start_loc = self.arange[:2]
        cad.query_start_loc_cpu = torch.from_numpy(self.token_arange_np[:2]).clone()
        if num_rejected_tokens_gpu is not None:
            cad.seq_lens -= num_rejected_tokens_gpu
            cad._seq_lens_cpu = None
            cad._num_computed_tokens_cpu = None

        if state["graph"] is None and not state["failed"]:
            try:
                _build_static_buffers(self, state, cad)
                _refresh_static_buffers(self, state, cad)
                _prime_loopgraph_outputs(state, first_token)
                self.hidden_states[:1].copy_(first_hidden)
                _capture_graph(self, state)
                print(
                    f"[pupa-loopgraph] captured K-1={token_count - 1} graph "
                    f"at eligible call {state['calls']} "
                    f"with slots={LOOPGRAPH_PINGPONG_SLOTS} (pid {os.getpid()})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                state["failed"] = True
                state["graph"] = None
                print(
                    f"[pupa-loopgraph] capture failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                _raise_or_fallback(exc)

        if state["graph"] is not None:
            output_slot = _select_loopgraph_output_slot(state)
            _refresh_static_buffers(self, state, cad)
            output_slot[0, 0:1].copy_(first_token)
            self.hidden_states[:1].copy_(first_hidden)
            graphs = state.get("graphs")
            graph = graphs[state["active_slot"]] if graphs else state["graph"]
            graph.replay()
            return output_slot

        cg_mode, input_batch_size, batch_size_dp = (
            self._determine_batch_execution_and_padding(1)
        )
        draft_tokens = [first_token]
        hidden_current = first_hidden
        loop_metadata = None
        for index in range(token_count - 1):
            input_ids = draft_tokens[-1].int()
            if index == 0:
                _, loop_metadata = self.build_per_group_and_layer_attn_metadata(
                    cad,
                    draft_index=1,
                )
            self.input_ids[:1] = input_ids
            self.hidden_states[:1] = hidden_current
            kwargs = {
                "input_ids": self.input_ids[:input_batch_size],
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": None,
                "hidden_states": self.hidden_states[:input_batch_size],
            }
            with set_forward_context(
                loop_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_dp,
                cudagraph_runtime_mode=cg_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                last_hidden, hidden = self.model(**kwargs)
            hidden_current = hidden[:1]
            token, _ = self._sample_draft_tokens(last_hidden[:1], sampling_metadata)
            draft_tokens.append(token)
        return torch.stack(draft_tokens, dim=1)

    proposer_cls.propose = propose
    print(
        f"[pupa-loopgraph] patched Gemma4Proposer.propose in pid {os.getpid()} "
        f"(warmup_calls={LOOPGRAPH_WARMUP_CALLS}, "
        f"require_capture={LOOPGRAPH_REQUIRE_CAPTURE})",
        file=sys.stderr,
        flush=True,
    )


def _prewarm_all() -> None:
    import torch
    import sys
    print("[pupa-warmup] starting triton kernel pre-warming...", file=sys.stderr, flush=True)
    
    # 1. Warm up sparse argmax triton kernels
    try:
        hidden_states = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
        lm_head_weight = torch.zeros((256000, 256), dtype=torch.bfloat16, device="cuda")
        top_k_indices = torch.zeros((1, 48), dtype=torch.int64, device="cuda")
        token_ordering = torch.zeros((2048, 128), dtype=torch.int64, device="cuda")
        partial_scores = torch.empty((1, 384), dtype=torch.float32, device="cuda")
        partial_tokens = torch.empty((1, 384), dtype=torch.int64, device="cuda")
        output_tokens = torch.empty((1,), dtype=torch.int64, device="cuda")
        
        triton, blocks_kernel, reduce_kernel = _get_fused_sparse_argmax_kernels()
        blocks_kernel[(1, 384)](
            hidden_states,
            lm_head_weight,
            top_k_indices,
            token_ordering,
            partial_scores,
            partial_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            lm_head_weight.stride(0),
            lm_head_weight.stride(1),
            top_k_indices.stride(0),
            top_k_indices.stride(1),
            partial_scores.stride(0),
            partial_tokens.stride(0),
            VOCAB_PER_CENTROID=128,
            SELECTED_COUNT=6144,
            HIDDEN_SIZE=256,
            BLOCK_SELECTED=16,
            BLOCK_D=256,
            num_warps=8,
        )
        reduce_kernel[(1,)](
            partial_scores,
            partial_tokens,
            output_tokens,
            partial_scores.stride(0),
            partial_tokens.stride(0),
            output_tokens.stride(0),
            NUM_BLOCKS=384,
            BLOCK_BLOCKS=512,
            num_warps=8,
        )
        print("[pupa-warmup] warmed up sparse argmax triton kernels successfully", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[pupa-warmup] sparse argmax triton warmup failed: {e!r}", file=sys.stderr, flush=True)

    # 2. Warm up standard rejection greedy sample kernel
    try:
        try:
            import importlib
            rejection_module = importlib.import_module("vllm.v1.sample.rejection_sampler")
        except Exception:
            rejection_module = sys.modules.get("vllm.v1.sample.rejection_sampler")
            
        if rejection_module is not None:
            rejection_greedy_sample_kernel = getattr(rejection_module, "rejection_greedy_sample_kernel", None)
            if rejection_greedy_sample_kernel is not None:
                output_token_ids = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
                cu_num_draft_tokens = torch.tensor([7], dtype=torch.int32, device="cuda")
                draft_token_ids = torch.zeros(7, dtype=torch.int32, device="cuda")
                target_argmax = torch.zeros(7, dtype=torch.int32, device="cuda")
                bonus_token_ids = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
                
                rejection_greedy_sample_kernel[(1,)](
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    None,
                    7,
                    None,
                    None,
                    SYNTHETIC_MODE=False,
                )
                print("[pupa-warmup] warmed up standard rejection greedy sample kernel successfully", file=sys.stderr, flush=True)
            else:
                print("[pupa-warmup] rejection_greedy_sample_kernel not found in rejection_sampler module", file=sys.stderr, flush=True)
        else:
            print("[pupa-warmup] vllm.v1.sample.rejection_sampler module not found", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[pupa-warmup] standard rejection greedy sample kernel warmup failed: {e!r}", file=sys.stderr, flush=True)


def _apply_loopgraph_copy_event_patch(module: Any) -> None:
    import torch

    runner_cls = module.GPUModelRunner
    original_copy_draft_token_ids_to_cpu = runner_cls._copy_draft_token_ids_to_cpu
    original_init = runner_cls.__init__

    def _copy_draft_token_ids_to_cpu(
        self: Any,
        scheduler_output: Any,
        zeros_only: bool = False,
    ) -> Any:
        draft_token_ids = getattr(self, "_draft_token_ids", None)
        result = original_copy_draft_token_ids_to_cpu(
            self, scheduler_output, zeros_only=zeros_only
        )
        if zeros_only or not torch.is_tensor(draft_token_ids):
            return result

        event = _LOOPGRAPH_SLOT_EVENTS_BY_PTR.get(draft_token_ids.data_ptr())
        copy_stream = getattr(self, "draft_token_ids_copy_stream", None)
        if event is not None and copy_stream is not None:
            with torch.cuda.stream(copy_stream):
                event.record(copy_stream)
            _LOOPGRAPH_SLOT_EVENT_RECORDED_BY_PTR[draft_token_ids.data_ptr()] = True
        return result

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        try:
            _prewarm_all()
        except Exception as e:
            print(f"[pupa-warmup] warmup invocation failed: {e!r}", file=sys.stderr, flush=True)

    runner_cls._copy_draft_token_ids_to_cpu = _copy_draft_token_ids_to_cpu
    runner_cls.__init__ = __init__
    print(
        f"[pupa-loopgraph] patched GPUModelRunner draft-token copy events "
        f"and __init__ (warmup) in pid {os.getpid()} (slots={LOOPGRAPH_PINGPONG_SLOTS})",
        file=sys.stderr,
        flush=True,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def _get_fused_sparse_argmax_kernels() -> Any:
    global _FUSED_SPARSE_ARGMAX_KERNELS
    if _FUSED_SPARSE_ARGMAX_KERNELS is not None:
        return _FUSED_SPARSE_ARGMAX_KERNELS

    import triton
    import triton.language as tl

    @triton.jit
    def _sparse_argmax_blocks_kernel(
        hidden_states,
        lm_head_weight,
        top_centroids,
        token_ordering,
        partial_scores,
        partial_tokens,
        hidden_stride_t,
        hidden_stride_d,
        lm_head_stride_v,
        lm_head_stride_d,
        top_stride_t,
        top_stride_k,
        partial_score_stride_t,
        partial_token_stride_t,
        VOCAB_PER_CENTROID: tl.constexpr,
        SELECTED_COUNT: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_SELECTED: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        token_idx = tl.program_id(0)
        selected_block = tl.program_id(1)

        selected_offsets = selected_block * BLOCK_SELECTED + tl.arange(
            0, BLOCK_SELECTED
        )
        valid_selected = selected_offsets < SELECTED_COUNT
        centroid_slots = selected_offsets // VOCAB_PER_CENTROID
        token_slots = selected_offsets - centroid_slots * VOCAB_PER_CENTROID

        centroid_ids = tl.load(
            top_centroids + token_idx * top_stride_t + centroid_slots * top_stride_k,
            mask=valid_selected,
            other=0,
        )
        vocab_ids = tl.load(
            token_ordering + centroid_ids * VOCAB_PER_CENTROID + token_slots,
            mask=valid_selected,
            other=0,
        )

        d_offsets = tl.arange(0, BLOCK_D)
        valid_d = d_offsets < HIDDEN_SIZE
        hidden = tl.load(
            hidden_states + token_idx * hidden_stride_t + d_offsets * hidden_stride_d,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            lm_head_weight
            + vocab_ids[:, None] * lm_head_stride_v
            + d_offsets[None, :] * lm_head_stride_d,
            mask=valid_selected[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(weights * hidden[None, :], axis=1)
        # The PyTorch sparse path materializes bf16 logits before argmax.
        scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = tl.where(valid_selected, scores, -float("inf"))
        best_score, best_local_idx = tl.max(
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        best_selected = selected_block * BLOCK_SELECTED + best_local_idx
        best_centroid_slot = best_selected // VOCAB_PER_CENTROID
        best_token_slot = best_selected - best_centroid_slot * VOCAB_PER_CENTROID
        best_centroid = tl.load(
            top_centroids + token_idx * top_stride_t + best_centroid_slot * top_stride_k
        )
        best_token = tl.load(
            token_ordering + best_centroid * VOCAB_PER_CENTROID + best_token_slot
        )
        tl.store(
            partial_scores + token_idx * partial_score_stride_t + selected_block,
            best_score,
        )
        tl.store(
            partial_tokens + token_idx * partial_token_stride_t + selected_block,
            best_token,
        )

    @triton.jit
    def _sparse_argmax_reduce_kernel(
        partial_scores,
        partial_tokens,
        output_tokens,
        partial_score_stride_t,
        partial_token_stride_t,
        output_stride_t,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_BLOCKS: tl.constexpr,
    ) -> None:
        token_idx = tl.program_id(0)
        block_offsets = tl.arange(0, BLOCK_BLOCKS)
        valid_blocks = block_offsets < NUM_BLOCKS
        scores = tl.load(
            partial_scores + token_idx * partial_score_stride_t + block_offsets,
            mask=valid_blocks,
            other=-float("inf"),
        )
        _, best_block = tl.max(
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        token = tl.load(
            partial_tokens + token_idx * partial_token_stride_t + best_block
        )
        tl.store(output_tokens + token_idx * output_stride_t, token)

    _FUSED_SPARSE_ARGMAX_KERNELS = (
        triton,
        _sparse_argmax_blocks_kernel,
        _sparse_argmax_reduce_kernel,
    )
    return _FUSED_SPARSE_ARGMAX_KERNELS


def _fallback_sparse_argmax(
    self: Any,
    original_get_top_tokens: Any,
    hidden_states: Any,
    lm_head_weight: Any,
    reason: Exception,
) -> Any:
    if FUSED_SPARSE_ARGMAX_REQUIRE:
        raise RuntimeError(
            "FUSED_SPARSE_ARGMAX_REQUIRE=1 but fusion failed"
        ) from reason
    if not getattr(self, "_pupa_fused_sparse_argmax_warned", False):
        self._pupa_fused_sparse_argmax_warned = True
        print(
            f"[pupa-fused-sparse-argmax] falling back to PyTorch path: {reason!r}",
            file=sys.stderr,
            flush=True,
        )
    return original_get_top_tokens(self, hidden_states, lm_head_weight)


def _apply_fused_top_token_patch(module: Any) -> None:
    import torch

    embedder_cls = module.Gemma4MTPMaskedEmbedder
    original_get_top_tokens = embedder_cls.get_top_tokens

    def _select_and_score_unsorted(self: Any, hidden_states: Any, lm_head_weight: Any):
        num_tokens = hidden_states.shape[0]
        _, top_k_indices = torch.topk(
            self.centroids(hidden_states),
            k=self.centroid_intermediate_top_k,
            dim=-1,
            sorted=False,
        )
        clusters = self.token_ordering.view(
            self.num_centroids,
            self.vocab_size_per_centroid,
        )
        selected = clusters[top_k_indices]
        embeddings = lm_head_weight[selected.reshape(-1)].view(
            num_tokens,
            self.num_selected,
            self.hidden_size,
        )
        logits = torch.einsum("td,tsd->ts", hidden_states, embeddings)
        return logits, selected.view(num_tokens, -1)

    def get_top_tokens_fused(self: Any, hidden_states: Any, lm_head_weight: Any) -> Any:
        if not FUSED_SPARSE_ARGMAX:
            return original_get_top_tokens(self, hidden_states, lm_head_weight)
        try:
            if (
                hidden_states.device.type != "cuda"
                or lm_head_weight.device.type != "cuda"
            ):
                raise RuntimeError("fusion requires CUDA tensors")
            if (
                hidden_states.dtype != torch.bfloat16
                or lm_head_weight.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "fusion currently preserves exact PyTorch argmax only for bf16"
                )
            hidden_size = int(self.hidden_size)
            if hidden_size <= 0 or hidden_size > 1024:
                raise RuntimeError(f"unsupported hidden_size={hidden_size}")

            triton, blocks_kernel, reduce_kernel = _get_fused_sparse_argmax_kernels()
            num_tokens = int(hidden_states.shape[0])
            selected_count = int(self.num_selected)
            block_selected = _next_power_of_2(FUSED_SPARSE_ARGMAX_BLOCK)
            num_blocks = triton.cdiv(selected_count, block_selected)
            reduce_block = _next_power_of_2(num_blocks)
            block_d = _next_power_of_2(hidden_size)

            _, top_k_indices = torch.topk(
                self.centroids(hidden_states),
                k=self.centroid_intermediate_top_k,
                dim=-1,
                sorted=False,
            )
            partial_scores = torch.empty(
                (num_tokens, num_blocks),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            partial_tokens = torch.empty(
                (num_tokens, num_blocks),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            output_tokens = torch.empty(
                (num_tokens,),
                dtype=torch.int64,
                device=hidden_states.device,
            )

            blocks_kernel[(num_tokens, num_blocks)](
                hidden_states,
                lm_head_weight,
                top_k_indices,
                self.token_ordering,
                partial_scores,
                partial_tokens,
                hidden_states.stride(0),
                hidden_states.stride(1),
                lm_head_weight.stride(0),
                lm_head_weight.stride(1),
                top_k_indices.stride(0),
                top_k_indices.stride(1),
                partial_scores.stride(0),
                partial_tokens.stride(0),
                VOCAB_PER_CENTROID=int(self.vocab_size_per_centroid),
                SELECTED_COUNT=selected_count,
                HIDDEN_SIZE=hidden_size,
                BLOCK_SELECTED=block_selected,
                BLOCK_D=block_d,
                num_warps=8,
            )
            reduce_kernel[(num_tokens,)](
                partial_scores,
                partial_tokens,
                output_tokens,
                partial_scores.stride(0),
                partial_tokens.stride(0),
                output_tokens.stride(0),
                NUM_BLOCKS=num_blocks,
                BLOCK_BLOCKS=reduce_block,
                num_warps=8,
            )
            return output_tokens
        except Exception as exc:
            return _fallback_sparse_argmax(
                self,
                original_get_top_tokens,
                hidden_states,
                lm_head_weight,
                exc,
            )

    embedder_cls._select_and_score = _select_and_score_unsorted
    embedder_cls.get_top_tokens = get_top_tokens_fused
    print(
        f"[pupa-fused-sparse-argmax] patched Gemma4MTPMaskedEmbedder top-token path "
        f"in pid {os.getpid()} (enabled={FUSED_SPARSE_ARGMAX}, "
        f"require={FUSED_SPARSE_ARGMAX_REQUIRE}, block={FUSED_SPARSE_ARGMAX_BLOCK})",
        file=sys.stderr,
        flush=True,
    )


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, patch_fn: Any) -> None:
        self._inner = inner
        self._patch_fn = patch_fn

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        self._patch_fn(module)


class _TargetFinder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, patch_fn: Any) -> None:
        self._target = target
        self._patch_fn = patch_fn
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self._target or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self._patch_fn)
        return spec


sys.meta_path.insert(0, _TargetFinder(TOP_TOKEN_TARGET, _apply_fused_top_token_patch))
sys.meta_path.insert(0, _TargetFinder(RUNNER_TARGET, _apply_loopgraph_copy_event_patch))
sys.meta_path.insert(0, _TargetFinder(LOOPGRAPH_TARGET, _apply_loopgraph_patch))
