# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Use the existing SDK SNRC producer with its shared physical-layout plan."""


from .mindiesd_parallel import _selected_config


def maybe_head_major_qkv(layer, qkv, rope_table, video_layout, q_weight, k_weight, eps):
    from vllm_omni.diffusion.layers import minimax_h3_acceleration as features

    impl, backend, _ = layer.effective_attention()
    config = _selected_config(impl) if backend.get_name() == "RAINFUSION_ATTN" else None
    if not features.enabled("split_norm_rope_concat") and not (config is not None and config.head_major_qkv):
        return None
    if rope_table is None or qkv.device.type != "npu":
        return None
    strategy = layer._get_active_parallel_strategy()
    heads, dim = layer.num_heads, layer.head_size
    plan = None
    if config is not None and strategy.name == "ulysses":
        plan = features.qkv_layout_plan(
            head_count=heads,
            head_dim=dim,
            batch_size=1,
            world=strategy._sp_group.ulysses_world_size,
            precision=impl.rainfusion.precision,
            sparse_type=config.sparse_type,
            pre_rearranged=bool(getattr(video_layout, "pre_rearranged", False)),
            overlap_comm=config.overlap_comm,
            optimize_attention_pipeline=config.optimize_attention_pipeline,
            requested=config.head_major_qkv,
        )
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope
    from vllm_omni.diffusion.models.minimax_h3.npu.split_norm_rope_concat import maybe_split_norm_rope_concat

    return maybe_split_norm_rope_concat(
        qkv,
        num_heads=heads,
        num_kv_heads=layer.num_kv_heads,
        head_dim=dim,
        q_weight=q_weight,
        k_weight=k_weight,
        rope_table=rope_table,
        eps=eps,
        reference=fused_qk_norm_rope,
        layout_plan=plan,
        enabled=True,
    )
