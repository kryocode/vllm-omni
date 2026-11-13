# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 admission and ownership for MindIE-SD distributed attention.

The SDK owns both A2A directions and attention. It is a complete executor,
not a scatter/gather transport. Declined calls have submitted no collectives.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.mindiesd_config import MindIEParallelConfig
from vllm_omni.diffusion.forward_context import get_ulysses_mode
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from mindiesd.parallel import ParallelState

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.layer import Attention


class MindIEParallelResource:
    """Own a caller-supplied SDK state on one live communicator/device.

    The pinned SDK has no close method. Omni drains work and drops its state
    reference at coordinator teardown; cyclic SDK chains are collected there.
    SDK internals and its process-group ownership are never modified.
    """

    def __init__(self, group: dist.ProcessGroup, device: torch.device, config: MindIEParallelConfig):
        from mindiesd.parallel import ParallelState

        self.group = group
        self.device = device
        self.config = config
        self.state: ParallelState | None = ParallelState(
            group=group,
            overlap_comm=config.overlap_comm,
            optimize_attention_pipeline=config.optimize_attention_pipeline,
            bsa_backend=config.bsa_backend,
        )
        self.failed = False
        self.active = False

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spans: list,
        *,
        used_len: int,
        precision: str,
        scale: float,
        sparsity: float,
        pre_rearranged: bool,
        head_major_qkv: bool | None = None,
    ) -> torch.Tensor:
        if self.state is None or self.failed:
            raise RuntimeError("MindIE parallel resource is closed or failed; recreate the worker")
        if self.active:
            raise RuntimeError("concurrent calls cannot share a MindIE parallel resource")
        from mindiesd.parallel import distributed_sparse_attention

        from vllm_omni.diffusion.layers import minimax_h3_acceleration as features

        requested = self.config.head_major_qkv if head_major_qkv is None else head_major_qkv
        self.state.qkv_layout_plan = features.qkv_layout_plan(
            head_count=q.shape[2],
            head_dim=q.shape[3],
            batch_size=q.shape[0],
            world=dist.get_world_size(self.group),
            chunking=q.shape[2] // dist.get_world_size(self.group),
            precision=precision,
            sparse_type=self.config.sparse_type,
            pre_rearranged=pre_rearranged,
            overlap_comm=self.config.overlap_comm,
            optimize_attention_pipeline=self.config.optimize_attention_pipeline,
            requested=requested,
        )
        self.active = True
        try:
            output = distributed_sparse_attention(
                q,
                k,
                v,
                spans,
                group=self.group,
                state=self.state,
                sparse_type=self.config.sparse_type,
                used_len=used_len,
                precision=precision,
                scale=scale,
                sparsity=sparsity,
                pre_rearranged=pre_rearranged,
            )
            features.count("distributed_attention_calls")
            if features.DIAGNOSTICS:
                stats = self.state.last_call_stats
                if stats is None:
                    raise RuntimeError("MindIE diagnostics require successful public last_call_stats")
                actual = stats["actual"]
                for name in (
                    "head_major_qkv_calls",
                    "head_major_qkv_view_calls",
                    "round_robin_merge_calls",
                    "qkv_receive_allocations",
                    "qkv_receive_reuses",
                ):
                    features.count(name, actual[name])
                if requested and not actual["head_major_qkv_calls"]:
                    features.count("head_major_qkv_fallback_calls")
                features.count("bsa_" + actual["bsa_backend"])
                features.count("pre_rearranged_calls" if pre_rearranged else "original_order_calls")
                features.count(
                    "optimized_pipeline_calls"
                    if self.config.optimize_attention_pipeline
                    else "unoptimized_pipeline_calls"
                )
            return output
        except BaseException:
            # No alternate transport or replay after a partial collective.
            self.failed = True
            raise
        finally:
            self.active = False

    def close(self) -> None:
        if self.state is None:
            return
        if self.active:
            raise RuntimeError("cannot close active MindIE parallel resource")
        # Every worker owns one current device. Drain both SDK streams before
        # releasing their buffers; failure leaves state and storage intact.
        with torch.npu.device(self.device):
            current_omni_platform.synchronize()
        self.state = None
        gc.collect()


def _selected_config(impl) -> MindIEParallelConfig | None:
    from vllm_omni.diffusion.layers import minimax_h3_acceleration as features

    config = impl.rainfusion.parallel
    if config is None and features.enabled("dual_stream_attention"):
        optimized = features.enabled("attention_pipeline_optimizations")
        import os

        hoist = optimized and os.environ.get("VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER", "1") == "1"
        return MindIEParallelConfig(
            overlap_comm=True,
            optimize_attention_pipeline=optimized,
            bsa_backend="eagle" if features.enabled("eagle_bsa") else "cann",
            sparse_type="rf_v3",
            reorder_hoist=hoist,
            head_major_qkv=features.head_major_qkv_enabled(),
        )
    if config is None and impl.rainfusion.overlap_comm:
        # Preserve the existing explicit overlap_comm configuration route.
        return MindIEParallelConfig(overlap_comm=True)
    return config


def try_distributed_sparse_attention(
    attention: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: AttentionMetadata,
) -> torch.Tensor | None:
    """Run the full executor, or decline before state allocation/communication."""
    impl, backend, _ = attention.effective_attention()
    if not current_omni_platform.is_npu() or backend.get_name() != "RAINFUSION_ATTN":
        return None
    config = _selected_config(impl)
    if config is None:
        return None
    if (
        attention.causal
        or attention.qkv_layout != "BSND"
        or attention._has_custom_attention
        or attention.is_paged_kv_active()
        or attention.paged_kv_cache_role is not None
        or query.ndim != 4
        or query.shape[0] != 1
        or query.shape != key.shape
        or query.shape != value.shape
        or any(t.dtype != torch.bfloat16 or t.device != query.device for t in (query, key, value))
        or metadata.full_attn_spans is not None
        or metadata.attn_mask is not None
    ):
        return None
    if any(
        t is not None
        for t in (metadata.joint_query, metadata.joint_key, metadata.joint_value, metadata.joint_attn_mask)
    ):
        return None
    if any(
        metadata.extra.get(name) is not None
        for name in (
            "minimax_h3_reference_kv_post_parallel",
            "minimax_h3_compact_reference_rows",
            "gate_compress",
            "vdn_window",
        )
    ):
        return None
    strategy = attention._get_active_parallel_strategy()
    if strategy.name != "ulysses" or get_ulysses_mode(default="strict") != "strict":
        return None
    coordinator = strategy._sp_group
    group = strategy._ulysses_pg
    if (
        group is None
        or not dist.is_initialized()
        or coordinator.ring_world_size != 1
        or coordinator.allgather_world_size != 1
        or strategy._scatter_idx != 2
        or strategy._gather_idx != 1
        or strategy._ulysses_a2a_permute
    ):
        return None
    world = dist.get_world_size(group)
    if world <= 1 or query.shape[2] % world:
        return None
    effective_metadata = attention._with_kv_cache_dtype(metadata)
    plan = impl._resolve_plan(effective_metadata)
    if plan is None or not 0 < plan.used_len <= world * query.shape[1]:
        return None
    precision = effective_metadata.extra.get("kv_cache_dtype", impl.rainfusion.precision)
    if precision == "float":
        precision = "bf16"
    if precision not in ("bf16", "mix"):
        return None
    pre_rearranged = bool(getattr(metadata.video_layout, "pre_rearranged", False))
    if (pre_rearranged or config.bsa_backend == "eagle") and precision != "bf16":
        raise ValueError("MindIE RF reorder hoist and Eagle BSA require BF16 precision")
    spans = plan.video_spans or [{"start": plan.prefix_len, "latent_shape": list(plan.latent_shape)}]
    resource = coordinator.resources.get_or_create(
        ("mindiesd", group, query.device),
        lambda: MindIEParallelResource(group, query.device, config),
    )
    fixed = ("overlap_comm", "optimize_attention_pipeline", "bsa_backend", "sparse_type")
    if any(getattr(resource.config, name) != getattr(config, name) for name in fixed):
        raise RuntimeError("MindIE parallel configuration must agree across layers sharing a communicator/device")
    return resource.execute(
        query,
        key,
        value,
        spans,
        used_len=plan.used_len,
        precision=precision,
        scale=impl.softmax_scale,
        sparsity=impl.rainfusion.sparsity,
        pre_rearranged=pre_rearranged,
        head_major_qkv=config.head_major_qkv,
    )
