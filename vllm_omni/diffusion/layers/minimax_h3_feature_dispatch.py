# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physically named MiniMax-H3 runtime dispatch and engagement counters."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from functools import wraps

import torch

from vllm_omni.diffusion.layers import minimax_h3_acceleration

COUNTS: Counter[str] = Counter()


def count(name):
    """Record optional per-process feature diagnostics."""
    if minimax_h3_acceleration.DIAGNOSTICS:
        COUNTS[name] += 1


def gate_norm(reference, residual, gate, branch, weight, shift, scale, indices, eps, *, admitted=True):
    """Dispatch PR790 when modulation fusions are enabled."""
    if not admitted or not minimax_h3_acceleration.enabled("modulation_fusions"):
        count("indexed_gate_norm_modulate_native")
        return reference(residual, gate, branch, weight, shift, scale, indices, eps)
    from mindiesd.fused_ops import eagle_indexed_gate_norm_modulate

    result = eagle_indexed_gate_norm_modulate(
        residual=residual,
        gate=gate,
        branch=branch,
        shift=shift,
        scale=scale,
        indices=indices,
        weight=weight,
        bias=None,
        norm_type=0,
        epsilon=eps,
        out_dtype=residual.dtype,
    )
    count("indexed_gate_norm_modulate_fused")
    count("sdk_f_facade_calls")
    return result


def norm_modulate(reference, x, weight, shift, scale, indices, eps, *, admitted=True):
    """Dispatch PR782 when modulation fusions are enabled."""
    if not admitted or not minimax_h3_acceleration.enabled("modulation_fusions"):
        count("rms_norm_indexed_scale_shift_native")
        return reference(x, weight, shift, scale, indices, eps)
    from mindiesd.fused_ops import rms_norm_indexed_scale_shift

    result = rms_norm_indexed_scale_shift(x=x, weight=weight, shift=shift, scale=scale, indices=indices, eps=eps)
    count("rms_norm_indexed_scale_shift_fused")
    count("sdk_f_facade_calls")
    return result


def indexed_gate(reference, x, gate, other, indices, *, admitted=True):
    """Dispatch PR801 when modulation fusions are enabled."""
    if not admitted or not minimax_h3_acceleration.enabled("modulation_fusions"):
        count("indexed_gate_native")
        return reference(x, gate, other, indices)
    from mindiesd.fused_ops import indexed_gate as operation

    output = torch.empty_like(x)
    operation(output=output, x=x, gate=gate, other=other, indices=indices)
    count("indexed_gate_fused")
    count("sdk_f_facade_calls")
    return output


def request_scope(function):
    """Reset and publish per-request runtime acceleration counters."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with minimax_h3_acceleration.request_layout_scope():
            timing = minimax_h3_acceleration.timing_enabled()
            if not minimax_h3_acceleration.DIAGNOSTICS and not timing:
                return function(*args, **kwargs)
            from vllm_omni.platforms import current_omni_platform

            COUNTS.clear()
            minimax_h3_acceleration.COUNTS.clear()
            if timing:
                current_omni_platform.synchronize()
            begin = time.monotonic()
            result = function(*args, **kwargs)
            if timing:
                current_omni_platform.synchronize()
            end = time.monotonic()
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else int(os.environ.get("RANK", "-1"))
            )
            if timing:
                print(
                    "[MINIMAX_H3_DIT_TIMING] "
                    + json.dumps(
                        {
                            "rank": rank,
                            "begin_monotonic": begin,
                            "end_monotonic": end,
                            "duration_s": end - begin,
                            "nfe": minimax_h3_acceleration.COUNTS.get("model_evaluations"),
                        }
                    ),
                    flush=True,
                )
            data = {
                "rank": rank,
                "features": sorted(minimax_h3_acceleration.FEATURES),
                "head_major_qkv_requested": minimax_h3_acceleration.head_major_qkv_enabled(),
                "head_major_qkv_verify": minimax_h3_acceleration.head_major_qkv_verify_enabled(),
                "runtime_counts": dict(COUNTS),
                "framework_counts": dict(minimax_h3_acceleration.COUNTS),
                "framework_feature_source": minimax_h3_acceleration.__file__,
            }
            print("[MINIMAX_H3_FEATURE_STATS] " + json.dumps(data, sort_keys=True), flush=True)
            return result

    return wrapped
