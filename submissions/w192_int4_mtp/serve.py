#!/usr/bin/env python3
"""
Phase 2: INT4 osoi5 + kenyan-duma MTP K=7 + W192 + loopgraph + fused argmax.
Exact PLE/SMP-02 source patches from combined-opt_antt-r1, extended with
DRAFTER_BUCKET support for the kenyan-duma drafter.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable

WEIGHTS_BUCKET = os.environ.get(
    "WEIGHTS_BUCKET",
    "hf://buckets/gemma-challenge/gemma-chiku-inu/weights/osoi5-v0-baked",
)
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/tmp/osoi5-v0-baked")
DRAFTER_BUCKET = os.environ.get("DRAFTER_BUCKET", "")
DRAFTER_REPO = os.environ.get(
    "DRAFTER_REPO", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
)
LOCAL_DRAFTER_DIR = os.environ.get("LOCAL_DRAFTER_DIR", "/tmp/qat-assistant")
CENTROID_TOP_K = int(os.environ.get("CENTROID_TOP_K", "48"))
JINJA2_VERSION = "3.1.6"
MARKUPSAFE_VERSION = "3.0.3"

TCMALLOC_CANDIDATES = [
    "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",
    "/usr/lib/libtcmalloc_minimal.so.4",
    "/usr/lib64/libtcmalloc_minimal.so.4",
]

Patcher = Callable[[str, pathlib.Path], tuple[str, bool]]


def replace_required(
    source: str,
    *,
    model_path: pathlib.Path,
    label: str,
    old: str,
    new: str,
    marker: str,
) -> tuple[str, bool]:
    if marker in source:
        return source, False
    old_count = source.count(old)
    if old_count != 1:
        raise RuntimeError(
            f"{label} patch pattern count is {old_count} in {model_path}; "
            "refusing to run a silent no-op baseline."
        )
    return source.replace(old, new, 1), True


# ---------------------------------------------------------------------------
# Exact PLE patch strings (from combined-opt_antt-r1/serve.py)
# ---------------------------------------------------------------------------

PLE_TEXT_FAST_PATH_OLD = """        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0,
            input_ids < self.vocab_size_per_layer_input,
        )
        per_layer_inputs_tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )
        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)
"""

PLE_TEXT_FAST_PATH_NEW = """        # Challenge fast path: harness text token IDs are valid PLE IDs.
        # Multimodal serving still maps multimodal positions to token 0 before
        # this call in gemma4_mm.py, so the multimodal PLE contract is retained.
        per_layer_embeds = self.embed_tokens_per_layer(input_ids)
"""

PLE_RUNTIME_SCALE_OLD = (
    "        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer\n"
)
PLE_RUNTIME_SCALE_NEW = (
    "        # PLE scale-fold: embed_scale_per_layer is folded into "
    "embedding weights after load.\n"
)

PLE_GATE_SCRATCH_OLD = """            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
"""

PLE_GATE_SCRATCH_NEW = """            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            # PLE scratch reuse: in-place gate multiply when dtype-preserving.
            if gate.dtype == per_layer_input.dtype:
                gate.mul_(per_layer_input)
                gated_per_layer = gate
            else:
                gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
"""

PLE_COMBINE_SCRATCH_OLD = """        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
"""

PLE_COMBINE_SCRATCH_NEW = """        if per_layer_inputs is None:
            return per_layer_projection
        # PLE scratch reuse: in-place projection add when dtype-preserving.
        if per_layer_projection.dtype == per_layer_inputs.dtype:
            per_layer_projection.add_(per_layer_inputs)
            return per_layer_projection * self.per_layer_input_scale
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
"""

SELF_DECODER_FOLD_ANCHOR = """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
"""

SELF_DECODER_FOLD_METHOD = """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    @torch.inference_mode()
    def fold_per_layer_embed_scale(self) -> None:
        if self.embed_tokens_per_layer is None or self.embed_scale_per_layer is None:
            return
        if getattr(self.embed_tokens_per_layer, "_ple_embed_scale_folded", False):
            return
        if self.hidden_size_per_layer_input != 256:
            raise RuntimeError(
                "PLE scale-fold expected hidden_size_per_layer_input=256, "
                f"got {self.hidden_size_per_layer_input}"
            )
        if self.embed_scale_per_layer.numel() != 1:
            raise RuntimeError("PLE scale-fold expects scalar embed_scale_per_layer")

        scale = float(self.embed_scale_per_layer.item())
        expected_scale = float(self.hidden_size_per_layer_input ** 0.5)
        if scale != expected_scale:
            raise RuntimeError(
                f"PLE scale-fold expected scale {expected_scale}, got {scale}"
            )

        embedding = self.embed_tokens_per_layer
        if hasattr(embedding, "weight_scale"):
            target = embedding.weight_scale
            folded_name = "weight_scale"
        elif hasattr(embedding, "weight"):
            target = embedding.weight
            folded_name = "weight"
        else:
            raise RuntimeError(
                "PLE scale-fold found no weight_scale or weight on "
                "embed_tokens_per_layer"
            )

        if target.dtype != torch.bfloat16:
            raise RuntimeError(
                f"PLE scale-fold expects bf16 {folded_name}, got {target.dtype}"
            )
        if target.device.type != "cuda":
            raise RuntimeError(
                f"PLE scale-fold expects CUDA {folded_name}, got {target.device}"
            )

        target.data.mul_(scale)
        embedding._ple_embed_scale_folded = True
        print(f"[serve] Folded Gemma4 PLE embed scale {scale} into {folded_name}", flush=True)

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
"""

MODEL_DELEGATE_OLD = '''    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Get per-layer embeddings from embed_tokens_per_layer.

        Returns:
            Per-layer embeddings (num_tokens, num_layers,
            hidden_size_per_layer_input)
        """
        return self.self_decoder.get_per_layer_inputs(input_ids)

    def project_per_layer_inputs(
'''

MODEL_DELEGATE_NEW = '''    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Get per-layer embeddings from embed_tokens_per_layer.

        Returns:
            Per-layer embeddings (num_tokens, num_layers,
            hidden_size_per_layer_input)
        """
        return self.self_decoder.get_per_layer_inputs(input_ids)

    def fold_per_layer_embed_scale(self) -> None:
        self.self_decoder.fold_per_layer_embed_scale()

    def project_per_layer_inputs(
'''

LOADER_IMPORT_OLD = "import inspect\n"
LOADER_IMPORT_NEW = "import inspect\nimport os\n"

LOADER_HOOK_OLD = """    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)
"""

LOADER_HOOK_NEW = """    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)

    if os.environ.get("PLE_FOLD_EMBED_SCALE") == "1":
        fold_target_model = os.environ.get("PLE_FOLD_TARGET_MODEL")
        current_model = getattr(model_config, "model", None)
        if fold_target_model and current_model != fold_target_model:
            pass
        else:
            candidates = [
                model,
                getattr(model, "model", None),
                getattr(getattr(model, "language_model", None), "model", None),
            ]
            fold_applied = False
            for candidate in candidates:
                folder = getattr(candidate, "fold_per_layer_embed_scale", None)
                if folder is None:
                    continue
                folder()
                decoder = getattr(candidate, "self_decoder", None)
                embedding = getattr(decoder, "embed_tokens_per_layer", None)
                fold_applied = bool(
                    getattr(embedding, "_ple_embed_scale_folded", False)
                )
                if not fold_applied:
                    raise RuntimeError(
                        "PLE_FOLD_EMBED_SCALE=1 but fold_per_layer_embed_scale "
                        "did not mark embed_tokens_per_layer as folded"
                    )
                break
            if not fold_applied:
                raise RuntimeError(
                    "PLE_FOLD_EMBED_SCALE=1 but no target model candidate "
                    "exposed fold_per_layer_embed_scale"
                )
"""

# ---------------------------------------------------------------------------
# SMP-02 slim-greedy patch strings (from combined-opt_antt-r1/serve.py)
# ---------------------------------------------------------------------------

DIXIE_SMP02_CONST_OLD = "logger = init_logger(__name__)\n"

DIXIE_SMP02_CONST_NEW = """logger = init_logger(__name__)

_DIXIE_SLIM_GREEDY = __import__("os").environ.get("DIXIE_SLIM_GREEDY", "1") == "1"
"""

DIXIE_SMP02_FWD_OLD = "        assert metadata.max_spec_len <= MAX_SPEC_LEN\n"

DIXIE_SMP02_FWD_NEW = """        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # dixie SMP-02: all-greedy fast path. bf16 -> fp32 is an exact,
        # monotonic upcast, so argmax over raw logits is bit-identical to the
        # slow path's argmax over the fp32 copy; the gate guarantees no logits
        # processor / penalty / mask / logprobs request can observe the
        # skipped work. Anything else falls through to the original code.
        if (
            _DIXIE_SLIM_GREEDY
            and sampling_metadata.all_greedy
            and not self.synthetic_mode
            and sampling_metadata.max_num_logprobs is None
            and sampling_metadata.no_penalties
            and not sampling_metadata.bad_words_token_ids
            and sampling_metadata.allowed_token_ids_mask is None
            and (
                sampling_metadata.thinking_budget_state_holder is None
                or not sampling_metadata.thinking_budget_state_holder.has_tracked_requests()
            )
        ):
            dixie_all_argmax = logits.argmax(dim=-1)
            dixie_bonus_token_ids = (
                dixie_all_argmax[metadata.bonus_logits_indices]
                .unsqueeze(1)
                .contiguous()
            )
            dixie_target_argmax = dixie_all_argmax[
                metadata.target_logits_indices
            ].contiguous()
            dixie_batch_size = len(metadata.num_draft_tokens)
            dixie_output_token_ids = torch.full(
                (dixie_batch_size, metadata.max_spec_len + 1),
                PLACEHOLDER_TOKEN_ID,
                dtype=torch.int32,
                device=logits.device,
            )
            rejection_greedy_sample_kernel[(dixie_batch_size,)](
                dixie_output_token_ids,
                metadata.cu_num_draft_tokens,
                metadata.draft_token_ids,
                dixie_target_argmax,
                dixie_bonus_token_ids,
                None,
                metadata.max_spec_len,
                None,
                None,
                SYNTHETIC_MODE=False,
            )
            return SamplerOutput(
                sampled_token_ids=dixie_output_token_ids,
                logprobs_tensors=None,
            )
"""


def patch_gemma4_source(source: str, model_path: pathlib.Path) -> tuple[str, bool]:
    changed_any = False
    for label, old, new, marker in (
        (
            "PLE valid-token fast path",
            PLE_TEXT_FAST_PATH_OLD,
            PLE_TEXT_FAST_PATH_NEW,
            "Challenge fast path: harness text token IDs are valid PLE IDs.",
        ),
        (
            "PLE scale-fold method",
            SELF_DECODER_FOLD_ANCHOR,
            SELF_DECODER_FOLD_METHOD,
            "def fold_per_layer_embed_scale",
        ),
        (
            "PLE runtime scale multiply",
            PLE_RUNTIME_SCALE_OLD,
            PLE_RUNTIME_SCALE_NEW,
            "PLE scale-fold: embed_scale_per_layer is folded into embedding weights",
        ),
        (
            "PLE gate scratch reuse",
            PLE_GATE_SCRATCH_OLD,
            PLE_GATE_SCRATCH_NEW,
            "PLE scratch reuse: in-place gate multiply",
        ),
        (
            "PLE projection-combine scratch reuse",
            PLE_COMBINE_SCRATCH_OLD,
            PLE_COMBINE_SCRATCH_NEW,
            "PLE scratch reuse: in-place projection add",
        ),
        (
            "PLE scale-fold model delegate",
            MODEL_DELEGATE_OLD,
            MODEL_DELEGATE_NEW,
            "self.self_decoder.fold_per_layer_embed_scale()",
        ),
    ):
        source, changed = replace_required(
            source,
            model_path=model_path,
            label=label,
            old=old,
            new=new,
            marker=marker,
        )
        changed_any = changed_any or changed
    return source, changed_any


def patch_loader_utils_source(
    source: str, model_path: pathlib.Path
) -> tuple[str, bool]:
    source, import_changed = replace_required(
        source,
        model_path=model_path,
        label="PLE scale-fold loader os import",
        old=LOADER_IMPORT_OLD,
        new=LOADER_IMPORT_NEW,
        marker="import os",
    )
    source, hook_changed = replace_required(
        source,
        model_path=model_path,
        label="PLE scale-fold loader hook",
        old=LOADER_HOOK_OLD,
        new=LOADER_HOOK_NEW,
        marker="PLE_FOLD_EMBED_SCALE",
    )
    return source, import_changed or hook_changed


def patch_rejection_sampler_source(
    source: str, model_path: pathlib.Path
) -> tuple[str, bool]:
    source, const_changed = replace_required(
        source,
        model_path=model_path,
        label="dixie SMP-02 slim-greedy const",
        old=DIXIE_SMP02_CONST_OLD,
        new=DIXIE_SMP02_CONST_NEW,
        marker="_DIXIE_SLIM_GREEDY",
    )
    source, fwd_changed = replace_required(
        source,
        model_path=model_path,
        label="dixie SMP-02 slim-greedy fast path",
        old=DIXIE_SMP02_FWD_OLD,
        new=DIXIE_SMP02_FWD_NEW,
        marker="dixie SMP-02: all-greedy fast path",
    )
    return source, const_changed or fwd_changed


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def ensure_weights() -> None:
    config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    if os.path.isdir(LOCAL_MODEL_DIR) and os.path.exists(config_path):
        print(f"[serve] weights already at {LOCAL_MODEL_DIR}", flush=True)
        return
    print(f"[serve] syncing weights {WEIGHTS_BUCKET} -> {LOCAL_MODEL_DIR}", flush=True)
    subprocess.run(
        ["hf", "buckets", "sync", WEIGHTS_BUCKET, LOCAL_MODEL_DIR], check=True
    )


def ensure_drafter() -> None:
    config_path = os.path.join(LOCAL_DRAFTER_DIR, "config.json")
    if not os.path.exists(config_path):
        if DRAFTER_BUCKET:
            print(
                f"[serve] syncing drafter {DRAFTER_BUCKET} -> {LOCAL_DRAFTER_DIR}",
                flush=True,
            )
            subprocess.run(
                ["hf", "buckets", "sync", DRAFTER_BUCKET, LOCAL_DRAFTER_DIR],
                check=True,
            )
        else:
            print(
                f"[serve] downloading drafter {DRAFTER_REPO} -> {LOCAL_DRAFTER_DIR}",
                flush=True,
            )
            from huggingface_hub import snapshot_download
            snapshot_download(DRAFTER_REPO, local_dir=LOCAL_DRAFTER_DIR)
    else:
        print(f"[serve] drafter already at {LOCAL_DRAFTER_DIR}", flush=True)

    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    old_top_k = config.get("centroid_intermediate_top_k", 32)
    if old_top_k != CENTROID_TOP_K:
        config["centroid_intermediate_top_k"] = CENTROID_TOP_K
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        print(
            f"[serve] centroid_intermediate_top_k: {old_top_k} -> {CENTROID_TOP_K}",
            flush=True,
        )


def find_tcmalloc() -> str | None:
    # Also check TCMALLOC_HINT from manifest env
    hint = os.environ.get("TCMALLOC_HINT", "")
    if hint and os.path.isfile(hint):
        return hint
    for path in TCMALLOC_CANDIDATES:
        if os.path.isfile(path):
            return path
    for path in glob.glob("/usr/lib/*/libtcmalloc_minimal.so.4"):
        if os.path.isfile(path):
            return path
    return None


def setup_ld_preload() -> None:
    requested = os.environ.get("LD_PRELOAD", "")
    if requested and os.path.isfile(requested.split(":")[0]):
        print(f"[serve] LD_PRELOAD already set: {requested}", flush=True)
        return

    lib = find_tcmalloc()
    if not lib:
        # Try apt-get install
        if shutil.which("apt-get"):
            print("[serve] installing libtcmalloc-minimal4 ...", flush=True)
            subprocess.run(
                ["apt-get", "install", "-y", "-qq", "libtcmalloc-minimal4"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            lib = find_tcmalloc()

    if not lib:
        # Clear any stale/invalid LD_PRELOAD so execvpe doesn't pass bad path
        os.environ.pop("LD_PRELOAD", None)
        print("[serve] WARNING: tcmalloc unavailable, LD_PRELOAD cleared", flush=True)
        return

    os.environ["LD_PRELOAD"] = lib
    print(f"[serve] LD_PRELOAD={lib}", flush=True)


def ensure_benchmark_jinja2() -> None:
    if os.environ.get("PATCH_BENCH_JINJA2") != "1":
        return
    bench_python = pathlib.Path(
        os.environ.get("BENCH_VENV_PYTHON", "/tmp/bench-venv/bin/python")
    )
    if not bench_python.exists():
        return
    check = subprocess.run(
        [str(bench_python), "-c", "import jinja2"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode == 0:
        return
    print(f"[serve] installing jinja2 into {bench_python}", flush=True)
    subprocess.run(
        [
            str(bench_python), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input", "--no-cache-dir",
            f"jinja2=={JINJA2_VERSION}", f"MarkupSafe=={MARKUPSAFE_VERSION}",
        ],
        check=True,
    )


def patch_file(path: pathlib.Path, patcher: Patcher) -> None:
    source = path.read_text(encoding="utf-8")
    patched_source, changed = patcher(source, path)
    if changed:
        path.write_text(patched_source, encoding="utf-8")
        print(f"[serve] patched {path}", flush=True)
    else:
        print(f"[serve] already patched: {path.name}", flush=True)


def patch_ple_sources() -> None:
    if (
        os.environ.get("PLE_ASSUME_VALID_TOKEN_IDS") != "1"
        and os.environ.get("PLE_FOLD_EMBED_SCALE") != "1"
    ):
        return
    os.environ.setdefault("PLE_FOLD_TARGET_MODEL", LOCAL_MODEL_DIR)
    purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    patch_file(
        purelib / "vllm" / "model_executor" / "models" / "gemma4.py",
        patch_gemma4_source,
    )
    patch_file(
        purelib / "vllm" / "model_executor" / "model_loader" / "utils.py",
        patch_loader_utils_source,
    )


def patch_smp02_sources() -> None:
    if os.environ.get("DIXIE_SLIM_GREEDY", "1") != "1":
        return
    purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    patch_file(
        purelib / "vllm" / "v1" / "sample" / "rejection_sampler.py",
        patch_rejection_sampler_source,
    )


def setup_sitecustomize_path() -> None:
    package_dir = str(pathlib.Path(__file__).resolve().parent)
    existing = os.environ.get("PYTHONPATH", "")
    paths = [p for p in existing.split(os.pathsep) if p]
    if package_dir not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([package_dir, *paths])
    print(f"[serve] PYTHONPATH sitecustomize prefix: {package_dir}", flush=True)


def append_env_arg(args: list[str], env_name: str, flag: str) -> None:
    value = os.environ.get(env_name)
    if value:
        args.extend([flag, value])


def main() -> None:
    ensure_benchmark_jinja2()
    ensure_weights()
    setup_ld_preload()
    ensure_drafter()
    patch_ple_sources()
    patch_smp02_sources()
    setup_sitecustomize_path()

    args = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", LOCAL_MODEL_DIR,
        "--served-model-name", os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it"),
        "--host", os.environ.get("HOST", "0.0.0.0"),
        "--port", os.environ.get("PORT", "8000"),
        "--dtype", os.environ.get("DTYPE", "bfloat16"),
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-num-seqs", os.environ.get("MAX_NUM_SEQS", "1"),
        "--performance-mode", os.environ.get("PERFORMANCE_MODE", "interactivity"),
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
    ]

    append_env_arg(args, "MAX_NUM_BATCHED_TOKENS", "--max-num-batched-tokens")
    append_env_arg(args, "SPECULATIVE_CONFIG", "--speculative-config")
    append_env_arg(args, "GENERATION_CONFIG", "--generation-config")
    append_env_arg(args, "OVERRIDE_GENERATION_CONFIG", "--override-generation-config")
    append_env_arg(args, "UVICORN_LOG_LEVEL", "--uvicorn-log-level")
    append_env_arg(args, "HF_OVERRIDES", "--hf-overrides")

    if os.environ.get("DISABLE_LOG_STATS") == "1":
        args.append("--disable-log-stats")

    print("[serve] launching:", " ".join(args), flush=True)
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
