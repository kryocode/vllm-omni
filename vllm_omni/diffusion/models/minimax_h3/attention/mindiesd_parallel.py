# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Admit MiniMax sparse calls before Omni's Ulysses all-to-all."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm_omni.diffusion.forward_context import get_ulysses_mode
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.layer import Attention


def try_distributed_sparse_attention(
    attention: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: AttentionMetadata,
) -> torch.Tensor | None:
    """Return local output, or decline before communication and preserve Omni's path.

    MindIE owns both all-to-all directions on admitted calls. Its errors propagate:
    retrying through Omni after a collective could desynchronize the ranks.
    """
    if (
        not current_omni_platform.is_npu()
        or attention.attn_backend.get_name() != "RAINFUSION_ATTN"
        or attention.causal
        or attention.qkv_layout != "BSND"
        or attention._has_custom_attention
        or attention.is_paged_kv_active()
        or attention.paged_kv_cache_role is not None
        or query.ndim != 4
        or query.shape[0] != 1
        or query.shape != key.shape
        or query.shape != value.shape
        or any(tensor.dtype != torch.bfloat16 or tensor.device != query.device for tensor in (query, key, value))
    ):
        return None
    if any(
        tensor is not None
        for tensor in (metadata.joint_query, metadata.joint_key, metadata.joint_value, metadata.joint_attn_mask)
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
    ):
        return None
    world = dist.get_world_size(group)
    if world <= 1 or query.shape[2] % world:
        return None

    # Reuse the backend's step/layer/video gates and the layer's effective
    # attention precision, including quantization exclusions. Transformer int8
    # config controls linear layers and does not participate in this choice.
    effective_metadata = attention._with_kv_cache_dtype(metadata)
    impl = attention.attention
    plan = impl._resolve_plan(effective_metadata)
    if plan is None or not 0 < plan.used_len <= world * query.shape[1]:
        return None
    precision = effective_metadata.extra.get("kv_cache_dtype", impl.rainfusion.precision)
    if precision == "float":
        precision = "bf16"
    if precision not in ("bf16", "mix"):
        return None
    spans = plan.video_spans or [{"start": plan.prefix_len, "latent_shape": list(plan.latent_shape)}]

    from mindiesd.parallel import ParallelState, distributed_sparse_attention

    # Every layer's Ulysses strategy shares this caller-owned coordinator. Keep
    # one state per live communicator/device, so mapping and chain caches survive
    # layer/step boundaries without a module-global cache or a new wrapper class.
    states = getattr(coordinator, "_mindiesd_parallel_states", None)
    if states is None:
        states = coordinator._mindiesd_parallel_states = {}
    state_key = (group, query.device)
    overlap_comm = getattr(impl.rainfusion, "overlap_comm", False)
    state = states.get(state_key)
    if state is None:
        state = states[state_key] = ParallelState(group=group, overlap_comm=overlap_comm)
    elif state.overlap_comm != overlap_comm:
        raise RuntimeError("overlap_comm must stay identical across layers sharing a communicator/device")

    return distributed_sparse_attention(
        query,
        key,
        value,
        spans,
        group=group,
        used_len=plan.used_len,
        state=state,
        precision=precision,
        scale=impl.softmax_scale,
        sparsity=impl.rainfusion.sparsity,
    )
