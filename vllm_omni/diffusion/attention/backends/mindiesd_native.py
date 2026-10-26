# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framework native consumers and RF-owned geometry for MindIE transport."""

import os
from dataclasses import replace
from importlib import import_module


def rf_cache_policy(config=None):
    """None preserves the existing route; explicit values select RF group caching."""
    value = (
        config.value("MINDIESD_RF_GEOMETRY_CACHE")
        if config is not None and config.isolated
        else os.environ.get("MINDIESD_RF_GEOMETRY_CACHE")
    )
    if value not in (None, "0", "1"):
        raise ValueError("MINDIESD_RF_GEOMETRY_CACHE must be 0 or 1")
    return None if value is None else value == "1"


def rf_execution_policy(config=None):
    """Keep requested policy serializable until the rank agreement completes."""
    if config is not None and config.isolated:
        return tuple(
            (name, config.value(name, "legacy" if name == "MINDIESD_RF_POOLING" else "0"))
            for name in (
                "MINDIESD_RF_GEOMETRY_CACHE",
                "MINDIESD_RF_POOLING",
                "MINDIESD_ATTENTION_FUSED_INPUT",
            )
        )
    return tuple(
        (name, os.environ[name])
        for name in (
            "MINDIESD_RF_GEOMETRY_CACHE",
            "MINDIESD_RF_POOLING",
            "MINDIESD_ATTENTION_FUSED_INPUT",
        )
        if name in os.environ
    )


def _policy_value(value):
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value))
    if isinstance(value, (tuple, list)):
        return tuple(_policy_value(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Native backend route policy must contain host scalar metadata")


class NativeBackendFactory:
    """Own one RF cache shared across model layers, separate from transport state."""

    def __init__(self):
        self.geometry_cache = None

    @staticmethod
    def contract(attention, *, declared=False):
        """Static policy signature gathered once at the shared request boundary."""
        backend = attention.attn_backend.get_name()
        impl = attention.attention
        strategy = (
            getattr(attention, "parallel_strategy", None) if declared else None
        ) or attention._get_active_parallel_strategy()
        rf = getattr(impl, "rainfusion", None)
        rf_policy = (
            None
            if rf is None
            else tuple(
                (name, _policy_value(getattr(rf, name, None)))
                for name in ("precision", "sparsity", "start_step", "end_step", "skip_layers")
            )
        )
        signature = (
            backend,
            bool(getattr(impl, "causal", False)),
            rf_policy,
            bool(strategy.enabled),
            strategy.name,
            bool(attention.skip_sequence_parallel),
            bool(getattr(attention, "_has_custom_attention", False)),
            attention._active_paged_kv_adapter() is not None,
            _policy_value(getattr(impl, "layer_idx", None)),
            _policy_value(getattr(impl, "qkv_layout", None)),
            _policy_value(getattr(impl, "softmax_scale", None)),
            _policy_value(getattr(attention, "_kv_cache_dtype", None)),
            bool(getattr(attention, "_disable_kv_quant", False)),
            _policy_value(getattr(attention, "_kv_cache_skip_steps", None)),
            _policy_value(getattr(attention, "_kv_cache_skip_layers", None)),
        )
        policy = rf_execution_policy(getattr(impl, "_h3_optimization_config", None))
        return signature + (("rf_execution_policy", policy),) if policy else signature

    @staticmethod
    def supports(attention, *, declared=False):
        strategy = (
            getattr(attention, "parallel_strategy", None) if declared else None
        ) or attention._get_active_parallel_strategy()
        backend = attention.attn_backend.get_name()
        return bool(
            strategy.enabled
            and strategy.name == "ulysses"
            and not attention.skip_sequence_parallel
            and attention._active_paged_kv_adapter() is None
            and not getattr(attention, "_has_custom_attention", False)
            and not getattr(attention.attention, "causal", False)
            and backend in ("FLASH_ATTN", "RAINFUSION_ATTN")
            and getattr(attention, "_kv_cache_dtype", None) in (None, "float", "auto")
            and (backend != "RAINFUSION_ATTN" or attention.attention.rainfusion.precision in ("float", "bf16"))
        )

    @staticmethod
    def graph_prefix_reason(attention, layout, used_len):
        """Prove this request cannot enter the unsupported dense prefix path.

        Reuse RF's host geometry resolver; do not reproduce sparse thresholds
        or inspect the current denoise step left by a preceding request.
        """
        if rf_cache_policy(getattr(attention.attention, "_h3_optimization_config", None)) is False:
            return "RF Graph requires prepared static geometry cache"
        if attention.attn_backend.get_name() != "RAINFUSION_ATTN":
            return "prefix Graph requires RF on every main DiT attention consumer"
        impl = attention.attention
        rf = impl.rainfusion
        if rf.sparsity <= 0 or rf.start_step != 0 or rf.end_step != 0:
            return "prefix Graph requires RF active for the whole denoise schedule (start_step=0, end_step=0)"
        layer = getattr(impl, "layer_idx", None)
        if (layer is None and rf.skip_layers) or layer in rf.skip_layers:
            return "prefix Graph cannot cover a skipped or unidentifiable RF layer"
        if getattr(impl, "qkv_layout", None) != "BSND":
            return "prefix Graph requires a declared BSND RF consumer"
        if layout is None or not layout.video_spans:
            return "prefix Graph requires full-query multi-span RF geometry"
        resolver = getattr(impl, "_resolve_multi_span_plan", None)
        if not callable(resolver) or resolver(layout, used_len) is None:
            return "prefix Graph RF geometry would select the dense fallback"
        return None

    def build(self, attention, metadata):
        cache_policy = rf_cache_policy(getattr(attention.attention, "_h3_optimization_config", None))
        name = attention.attn_backend.get_name()
        plan = attention.attention._resolve_plan(metadata) if name == "RAINFUSION_ATTN" else None
        direct_rf = plan is not None and plan.video_spans is not None
        if direct_rf:
            precision = metadata.extra.get("kv_cache_dtype", attention.attention.rainfusion.precision)
            if precision not in ("float", "bf16"):
                raise RuntimeError("Prepared native RF backend changed its BF16 precision contract")
            if cache_policy is not False and self.geometry_cache is None:
                geometry = import_module("mindiesd.layers.flash_attn.rf_geometry")
                self.geometry_cache = geometry.RFGeometryCache()
            extra = dict(metadata.extra)
            if cache_policy is False:
                extra.pop("mindiesd_rf_geometry_cache", None)
            else:
                extra["mindiesd_rf_geometry_cache"] = self.geometry_cache
            metadata = replace(
                metadata,
                extra=extra,
            )
        return NativeAttentionBackend(attention, metadata, plan, direct_rf)


class NativeAttentionBackend:
    """Keep data-dependent dense/RF work in the framework's existing consumer."""

    def __init__(self, attention, metadata, plan, direct_rf):
        self.attention, self.metadata = attention, metadata
        self.plan, self.direct_rf = plan, direct_rf
        self.is_sparse = plan is not None

    def forward(self, q, k, v, *, chunk, out=None):
        metadata = self.metadata
        if self.direct_rf and out is not None:
            metadata = replace(metadata, extra={**metadata.extra, "mindiesd_rf_output": out})
        result = self.attention._run_local_attention(q, k, v, metadata)
        if out is not None and not self.direct_rf:
            out.copy_(result)
            return out
        return result
