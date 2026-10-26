# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Thin H3 adapter for PR793 BF16 attention and optional archived dual-stream scheduling."""

import os

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_ulysses_mode


class MindIESDH3PR793:
    """One model-owned state shared by DiT layers; one overlap switch, no Graph/DLO integration."""

    def __init__(self):
        value = os.environ.get("VLLM_OMNI_MINIMAX_H3_PR793_NONQUANT", "0")
        if value not in ("0", "1"):
            raise ValueError("VLLM_OMNI_MINIMAX_H3_PR793_NONQUANT must be 0 or 1")
        self.enabled = value == "1"
        value = os.environ.get("VLLM_OMNI_MINIMAX_H3_PR793_OVERLAP", "1")
        if value not in ("0", "1"):
            raise ValueError("VLLM_OMNI_MINIMAX_H3_PR793_OVERLAP must be 0 or 1")
        self.overlap = value == "1"
        self.state = None

    def try_attention(self, attention, q, k, v, metadata, *, num_requests=1):
        if not self.enabled:
            return None
        from mindiesd.parallel import ParallelState, distributed_sparse_attention, rank_window
        from mindiesd.utils.get_platform import is_a5_device

        if q.device.type != "npu" or is_a5_device():
            raise ValueError("PR793 nonquant requires A2/A3 NPU")
        if get_ulysses_mode(default="strict") != "strict" or getattr(attention, "use_ring", False):
            raise ValueError("PR793 nonquant requires strict Ulysses without ring")
        impl = attention.attention
        if type(impl).__name__ != "RainFusionAttentionImpl":
            raise ValueError("PR793 nonquant requires native RainFusion attention")
        sp = get_sp_group()
        group = sp.ulysses_group
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        if int(sp.world_size) != world or int(getattr(sp, "ring_world_size", 1)) != 1:
            raise ValueError("PR793 nonquant does not support hybrid sequence parallelism")
        metadata = attention._with_kv_cache_dtype(metadata)
        precision = metadata.extra.get("kv_cache_dtype", impl.rainfusion.precision)
        plan = impl._resolve_plan(metadata)
        eligible = (
            num_requests == 1
            and q.ndim == 4
            and q.shape == k.shape == v.shape
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.shape[2] % world == 0
            and precision in ("float", "bf16")
            and not impl.causal
            and (plan is None or plan.video_spans is not None)
        )
        signature = None
        if eligible:
            signature = (
                ("dense", self.overlap)
                if plan is None
                else (
                    "sparse",
                    self.overlap,
                    tuple(q.shape),
                    str(q.dtype),
                    plan.used_len,
                    tuple((int(s["start"]), tuple(s["latent_shape"])) for s in plan.video_spans),
                    float(impl.softmax_scale),
                    float(impl.rainfusion.sparsity),
                )
            )
        signatures = [None] * world
        dist.all_gather_object(signatures, signature, group=group)
        if signature is None or any(s != signature for s in signatures):
            raise ValueError("PR793 nonquant rank contracts disagree or request is unsupported")
        if plan is None:
            return None
        if self.state is None:
            self.state = ParallelState(group=group, overlap=self.overlap)
        out = distributed_sparse_attention(
            q,
            k,
            v,
            plan.video_spans,
            group=group,
            state=self.state,
            used_len=plan.used_len,
            scale=impl.softmax_scale,
            sparsity=impl.rainfusion.sparsity,
        )
        _, _, valid = rank_window(rank, world, q.shape[1], plan.used_len)
        if valid < q.shape[1]:
            out[:, valid:].copy_(q[:, valid:])
        return out
