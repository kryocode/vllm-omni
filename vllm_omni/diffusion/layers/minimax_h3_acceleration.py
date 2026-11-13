# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physical MiniMax-H3 feature flags and per-request engagement counters."""

from __future__ import annotations

import os
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_REQUEST_LAYOUT: ContextVar[tuple[LayoutSnapshot, dict[tuple, Any]] | None] = ContextVar(
    "h3_request_layout", default=None
)

KNOWN_FEATURES = frozenset(
    {
        "dual_stream_attention",
        "attention_pipeline_optimizations",
        "weight_nz",
        "eagle_bsa",
        "split_norm_rope_concat",
        "modulation_fusions",
        "qbmm_swiglu",
    }
)
configured = os.environ.get("VLLM_OMNI_MINIMAX_H3_ACCELERATION_FEATURES")
FEATURES = frozenset(filter(None, (value.strip() for value in (configured or "").split(","))))
unknown = FEATURES - KNOWN_FEATURES
if unknown:
    raise ValueError(f"Unknown VLLM_OMNI_MINIMAX_H3_ACCELERATION_FEATURES values: {', '.join(sorted(unknown))}")

DIAGNOSTICS = os.environ.get("VLLM_OMNI_MINIMAX_H3_ACCELERATION_DIAGNOSTICS", "0") == "1"
COUNTS: Counter[str] = Counter()


def enabled(feature: str) -> bool:
    """Return whether one independently selectable acceleration feature is enabled."""
    if feature not in KNOWN_FEATURES:
        raise ValueError(f"Unknown MiniMax-H3 feature: {feature!r}")
    return feature in FEATURES


def count(name: str, amount: int = 1) -> None:
    """Record one feature-path engagement in the current request scope."""
    if DIAGNOSTICS:
        COUNTS[name] += amount


def _physical_bool(name: str) -> bool:
    value = os.environ.get(name, "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def timing_enabled() -> bool:
    """Select the optional, symmetric denoise-loop timing observer."""
    return _physical_bool("VLLM_OMNI_MINIMAX_H3_ACCELERATION_TIMING")


def _head_major_qkv_requested() -> bool:
    """Read the physical layout switch afresh, checking producer/consumer dependencies."""
    requested = _physical_bool("VLLM_OMNI_MINIMAX_H3_HEAD_MAJOR_QKV")
    if requested:
        required = {"dual_stream_attention", "attention_pipeline_optimizations", "split_norm_rope_concat"}
        missing = required - FEATURES
        if missing:
            raise ValueError("Head-major QKV requires explicit features: " + ", ".join(sorted(missing)))
    return requested


def _head_major_qkv_verify_requested() -> bool:
    """Read the optional full same-operator storage diagnostic afresh."""
    return _physical_bool("VLLM_OMNI_MINIMAX_H3_HEAD_MAJOR_QKV_VERIFY")


@dataclass(frozen=True)
class LayoutSnapshot:
    requested: bool
    verify: bool


def head_major_qkv_enabled() -> bool:
    current = _REQUEST_LAYOUT.get()
    return current[0].requested if current is not None else _head_major_qkv_requested()


def head_major_qkv_verify_enabled() -> bool:
    current = _REQUEST_LAYOUT.get()
    return current[0].verify if current is not None else _head_major_qkv_verify_requested()


@contextmanager
def request_layout_scope():
    """Read the dynamic physical switches once at each request boundary."""
    if _REQUEST_LAYOUT.get() is not None:
        yield
        return
    snapshot = LayoutSnapshot(_head_major_qkv_requested(), _head_major_qkv_verify_requested())
    token = _REQUEST_LAYOUT.set((snapshot, {}))
    try:
        yield
    finally:
        _REQUEST_LAYOUT.reset(token)


def qkv_layout_plan(
    *,
    head_count,
    head_dim,
    batch_size,
    world,
    precision="bf16",
    sparse_type="rf_v3",
    pre_rearranged=False,
    overlap_comm=True,
    optimize_attention_pipeline=None,
    requested=None,
    chunking=None,
):
    """Share one immutable SDK layout decision between this request's producer and consumer."""
    requested = head_major_qkv_enabled() if requested is None else requested
    optimize = (
        enabled("attention_pipeline_optimizations")
        if optimize_attention_pipeline is None
        else optimize_attention_pipeline
    )
    key = (
        head_count,
        head_dim,
        batch_size,
        world,
        precision,
        sparse_type,
        pre_rearranged,
        overlap_comm,
        optimize,
        requested,
    )
    current = _REQUEST_LAYOUT.get()
    cache = current[1] if current is not None else {}
    if key not in cache:
        try:
            from mindiesd.parallel import plan_qkv_layout
        except ImportError:
            if requested:
                raise RuntimeError("Head-major QKV requires MindIE-SD public plan_qkv_layout")
            return None
        count("sdk_qkv_layout_plan_calls")
        cache[key] = plan_qkv_layout(
            head_count=head_count,
            head_dim=head_dim,
            batch_size=batch_size,
            world=world,
            chunking=(head_count // world if world > 0 else 0) if chunking is None else chunking,
            width=1,
            precision=precision,
            sparse_type=sparse_type,
            pre_rearranged=pre_rearranged,
            overlap_comm=overlap_comm,
            optimize_attention_pipeline=optimize,
            requested=requested,
        )
    return cache[key]
