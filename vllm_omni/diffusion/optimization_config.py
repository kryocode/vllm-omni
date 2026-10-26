# SPDX-License-Identifier: Apache-2.0
"""Typed startup options for H3 execution; no environment or device state."""

H3_EXECUTION_OPTIONS = {
    "diffusion_attn_head_overlap_mode": (("legacy", "pipeline"), "legacy"),
    "diffusion_attn_rf_cache_mode": (("native", "cached"), "native"),
    "minimax_h3_rf_reorder_mode": (("none", "post_embed", "pre_embed"), "none"),
    "diffusion_attn_head_overlap_graph_mode": (("off", "graph1", "graph2"), "off"),
    "dlo_allgather_gate_mode": (("off", "first_head"), "off"),
    "dlo_weight_format": (("nd", "nz"), "nd"),
}


def validate_h3_execution_options(config):
    """Validate before workers start, including canonical offload selectors."""
    values = {}
    for name, (choices, default) in H3_EXECUTION_OPTIONS.items():
        value = getattr(config, name, default)
        if value not in choices:
            raise ValueError(f"--{name.replace('_', '-')} must be one of {', '.join(choices)}")
        values[name] = value
    if values["diffusion_attn_head_overlap_graph_mode"] != "off" and (
        values["diffusion_attn_head_overlap_mode"] != "pipeline" or values["diffusion_attn_rf_cache_mode"] != "cached"
    ):
        raise ValueError(
            "Graph requires --diffusion-attn-head-overlap-mode pipeline and --diffusion-attn-rf-cache-mode cached"
        )
    gate = values["dlo_allgather_gate_mode"] != "off"
    if gate and values["diffusion_attn_head_overlap_mode"] != "pipeline":
        raise ValueError("DLO first_head gate requires --diffusion-attn-head-overlap-mode pipeline")
    if gate or values["dlo_weight_format"] == "nz":
        if getattr(config, "diffusion_offload_config", None) is not None:
            from vllm_omni.diffusion.offloader.config import DIT_COMPONENT, OffloadStrategy, resolve_offload

            resolved = resolve_offload(config)
            allgather = resolved.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE and resolved.uses_allgather(
                DIT_COMPONENT
            )
        else:
            allgather = bool(getattr(config, "enable_distributed_layerwise_offload", False)) and bool(
                getattr(config, "dlo_use_allgather", True)
            )
        if not allgather:
            raise ValueError("DLO weight gate/NZ require distributed layerwise offload with DiT AllGather")
    return values
