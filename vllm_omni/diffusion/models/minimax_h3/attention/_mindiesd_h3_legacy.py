# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional BF16 Ulysses communication overlap for MiniMax-H3.

Set VLLM_OMNI_MINIMAX_H3_MINDIESD=1 to enable this adapter (default: off).
MINDIESD_OVERLAP selects a persistent communication stream (default: 1).
MINDIESD_HEAD_CHUNK_SIZE is heads per destination rank per chunk (default: 1;
0 selects all heads). MINDIESD_VERIFY_FIRST enables bit-exact checks against
the original Attention route (default: 0). All switches share the prefix
VLLM_OMNI_MINIMAX_H3_. Unsupported configurations retain the original route.
Attention computation remains in the framework's selected local backend.
RainFusion sparse steps reuse its native BF16 consumer and full video geometry;
attention quantization retains the original route.
"""

from __future__ import annotations

import functools
import os
from dataclasses import replace
from importlib import import_module

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_ulysses_mode

logger = init_logger(__name__)
_REQUIRES_MINDIESD = (
    "MiniMax-H3 communication overlap requires the dev-pr657-pr658 "
    "MindIE-SD build with distributed_attention attention_fn support."
)


@functools.cache
def _mindiesd_module(name: str):
    try:
        return import_module(f"mindiesd.{name}")
    except ImportError as exc:
        raise ImportError(_REQUIRES_MINDIESD) from exc


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get("VLLM_OMNI_MINIMAX_H3_" + name)
    if value is None:
        return default
    if value not in ("0", "1"):
        raise ValueError(f"VLLM_OMNI_MINIMAX_H3_{name} must be 0 or 1.")
    return value == "1"


def _d_mode() -> bool:
    value = os.environ.get("MINDIESD_ATTENTION_D", "0")
    if value not in ("0", "1"):
        raise ValueError("MINDIESD_ATTENTION_D must be 0 or 1.")
    return value == "1"


def _graph_signature(impl, plan, metadata, config=None):
    """Describe the unchanged F consumer without modifying RainFusion itself.

    Sparse full-query RF reads geometry/policy scalars only. Dense varlen
    metadata contains changing tensor addresses and stays eager.
    """
    if plan is None or plan.video_spans is None or metadata is None:
        return None
    requested = metadata.extra.get("kv_cache_dtype", impl.rainfusion.precision)
    if requested not in ("float", "bf16"):
        return None
    backend = import_module(type(impl).__module__)
    if not all(hasattr(backend, name) for name in ("_BLOCK_SIZE", "_INNER_PRECISE")):
        return None
    rf = _mindiesd_module("layers.flash_attn.rf_geometry")
    spans = tuple((int(s["start"]), tuple(int(v) for v in s["latent_shape"])) for s in plan.video_spans)
    return (
        "unchanged_F_rf_v2_full",
        "sparse",
        int(plan.used_len),
        spans,
        float(impl.rainfusion.sparsity),
        float(impl.softmax_scale),
        int(backend._BLOCK_SIZE),
        int(backend._INNER_PRECISE),
        "bf16",
        (False if config is not None and config.isolated else bool(rf.rf_pool_split_enabled())),
        os.environ.get("MINDIE_SD_FA_TYPE", ""),
    )


class MindIESDH3Parallel:
    """Model-owned state shared by layers and denoise steps, initialized lazily."""

    def __init__(self, config=None) -> None:
        self.enabled = config.enabled("MINDIESD") if config is not None else _enabled("MINDIESD", default=False)
        self.overlap = config.enabled("MINDIESD_OVERLAP") if config is not None else _enabled("MINDIESD_OVERLAP")
        self.verify_first = (
            config.enabled("MINDIESD_VERIFY_FIRST")
            if config is not None
            else _enabled("MINDIESD_VERIFY_FIRST", default=False)
        )
        chunk_size = int(
            config.value("VLLM_OMNI_MINIMAX_H3_MINDIESD_HEAD_CHUNK_SIZE", "1")
            if config is not None
            else os.environ.get("VLLM_OMNI_MINIMAX_H3_MINDIESD_HEAD_CHUNK_SIZE", "1")
        )
        if chunk_size < 0:
            raise ValueError("MINDIESD_HEAD_CHUNK_SIZE must be nonnegative.")
        self.head_chunk_size = chunk_size or None
        self.state = None
        self.calls = 0
        self.dense_calls = 0
        self.sparse_calls = 0
        self._verified_attention = set()

    @torch.compiler.disable
    def try_attention(self, attention, query, key, value, metadata):
        """Return local BSND output, or None to use the original Attention route."""
        if not self.enabled or query.device.type != "npu" or attention.skip_sequence_parallel:
            return None
        d_mode = _d_mode()
        strategy = attention._get_active_parallel_strategy()
        if not strategy.enabled or strategy.name != "ulysses" or get_ulysses_mode(default="strict") != "strict":
            return None
        if (
            query.ndim != 4
            or query.shape[0] != 1
            or key.shape != query.shape
            or value.shape != query.shape
            or any(t.dtype != torch.bfloat16 or t.device != query.device for t in (key, value))
            or query.dtype != torch.bfloat16
            or attention._active_paged_kv_adapter() is not None
            or getattr(attention, "_has_custom_attention", False)
            or getattr(attention.attention, "causal", False)
        ):
            return None
        # Only H3's single-document, mask-free prefix contract is supported.
        # Unknown extras may carry head-specific tensors that require resharding.
        if metadata is None or metadata.extra.get("h3_num_requests") != 1:
            return None
        if any(
            getattr(metadata, name, None) is not None
            for name in (
                "attn_mask",
                "joint_attn_mask",
                "joint_query",
                "joint_key",
                "joint_value",
                "full_attn_spans",
                "query_ranges",
            )
        ):
            return None
        allowed = {
            "h3_num_requests",
            "valid_kv_length",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "max_seqlen_q",
            "max_seqlen_k",
            "npu_attn_varlen",
            "laser_input_scale",
            "kv_cache_dtype",
        }
        if metadata.extra.keys() - allowed:
            return None
        local_metadata = attention._with_kv_cache_dtype(metadata)
        if local_metadata.extra.get("kv_cache_dtype") not in (None, "float", "auto"):
            return None
        backend = attention.attn_backend.get_name()
        plan = None
        if backend == "RAINFUSION_ATTN":
            plan = attention.attention._resolve_plan(local_metadata)
            if plan is not None:
                # Match the native sparse consumer's precision selection exactly.
                # Transformer INT8 projection is independent of this BF16 route.
                precision = local_metadata.extra.get("kv_cache_dtype", attention.attention.rainfusion.precision)
                if precision not in ("float", "bf16"):
                    return None
        elif backend != "FLASH_ATTN":
            return None
        sp_group = get_sp_group()
        group = sp_group.ulysses_group
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        if world <= 1 or world != sp_group.world_size or sp_group.ring_world_size != 1 or query.shape[2] % world:
            return None
        used_len = metadata.extra.get("valid_kv_length")
        if not isinstance(used_len, int) or not 0 < used_len <= query.shape[1] * world:
            return None
        extra = dict(local_metadata.extra)
        if any(extra.get(name) != used_len for name in ("max_seqlen_q", "max_seqlen_k")):
            return None
        if not extra.get("npu_attn_varlen"):
            return None
        for name in ("cu_seqlens_q", "cu_seqlens_k"):
            cu = extra.get(name)
            if not torch.is_tensor(cu) or cu.ndim != 1 or cu.shape[0] not in (2, 3):
                return None
            extra[name] = cu[:2]
        # Each callback receives the complete valid sequence, with no pad document.
        # Preserve video_layout as-is: sparse geometry is sequence-specific and
        # every head chunk sees the entire valid sequence. Keep kernel policy
        # (including laser_input_scale and KV exemptions) in the native consumer.
        local_metadata = replace(local_metadata, extra=extra, packed_padding=None)

        def attention_fn(q, k, v):
            return attention._run_local_attention(q, k, v, local_metadata)

        parallel = _mindiesd_module("parallel")
        window_fn = getattr(parallel, "rank_window", None)
        if not callable(window_fn):
            raise RuntimeError("H3 BF16 transport requires the paired MindIE-SD rank_window API")
        _, _, local_used = window_fn(rank, world, query.shape[1], used_len)
        if self.state is None:
            self.state = parallel.ParallelState(group=group)
            if self.overlap:
                self.state.make_stream(query.device)
        output_options = {}
        if d_mode and plan is not None and plan.video_spans is not None:
            # The native BF16 consumer owns every data-dependent operation.
            # Only geometry and the final destination are supplied by this adapter.
            local_metadata = replace(
                local_metadata,
                extra={**local_metadata.extra, "mindiesd_rf_geometry_cache": self.state.geom_cache},
            )

            def attention_out_fn(q, k, v, *, out):
                output_metadata = replace(
                    local_metadata,
                    extra={**local_metadata.extra, "mindiesd_rf_output": out},
                )
                return attention._run_local_attention(q, k, v, output_metadata)

            output_options["attention_out_fn"] = attention_out_fn
            if os.environ.get("MINDIESD_ATTENTION_GRAPH", "0") == "1":
                output_options["graph_signature"] = _graph_signature(attention.attention, plan, local_metadata)
        output, stats = parallel.distributed_attention(
            query,
            key,
            value,
            input_layout="BSND",
            ulysses_group=group,
            strategy="ulysses",
            head_chunk_size=self.head_chunk_size,
            used_len=used_len,
            state=self.state,
            attention_fn=attention_fn,
            return_stats=True,
            **output_options,
        )
        if output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
            raise RuntimeError("MindIE-SD attention did not restore local BSND layout.")
        if local_used < query.shape[1]:
            output[:, local_used:].copy_(query[:, local_used:])
        if self.verify_first:
            # Layers and dense/sparse transitions can share identical tensor
            # shapes. Geometry is serialized only for this optional check.
            signature = (attention, backend, tuple(query.shape), query.dtype, used_len, repr(plan), d_mode)
            if signature not in self._verified_attention:
                reference = attention(query, key, value, metadata)
                if not torch.equal(output[:, :local_used], reference[:, :local_used]):
                    raise AssertionError("MindIE-SD parallel attention is not bit-exact on valid rows.")
                self._verified_attention.add(signature)
        self.calls += 1
        if plan is None:
            self.dense_calls += 1
        else:
            self.sparse_calls += 1
        logger.info_once(
            "MiniMax-H3 MindIE-SD BF16 Ulysses attention: route=%s overlap=%s stats=%s",
            "sparse" if plan is not None else "dense",
            self.overlap,
            str(stats),
        )
        return output
