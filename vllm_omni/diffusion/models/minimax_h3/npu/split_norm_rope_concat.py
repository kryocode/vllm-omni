# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in MiniMax-H3 adapter for MindIE-SD EagleSplitNormRopeConcat."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

_STATE: dict[str, object] = {"calls": 0}
_VERIFIED_SIGNATURES: set[tuple] = set()


def _enabled() -> bool:
    from vllm_omni.diffusion.layers import minimax_h3_acceleration

    return minimax_h3_acceleration.enabled("split_norm_rope_concat")


def _rank() -> str:
    return os.environ.get("RANK", "unknown")


def _rope_caches(rope_table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _STATE.get("rope_table") is not rope_table:
        half = rope_table.shape[-1] // 2
        _STATE["rope_table"] = rope_table
        _STATE["rope_cos"] = rope_table[..., :half].contiguous()
        _STATE["rope_sin"] = rope_table[..., half:].contiguous()
    return _STATE["rope_cos"], _STATE["rope_sin"]


def _cosine_and_norm_ratio(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.float().reshape(-1)
    expected_f = expected.float().reshape(-1)
    actual_norm = torch.linalg.vector_norm(actual_f)
    expected_norm = torch.linalg.vector_norm(expected_f)
    cosine = torch.dot(actual_f, expected_f) / (actual_norm * expected_norm).clamp_min(1e-12)
    ratio = actual_norm / expected_norm.clamp_min(1e-12)
    return float(cosine), float(ratio)


def _verify_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qkv_heads: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    reference: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> None:
    if os.environ.get("VLLM_OMNI_MINIMAX_H3_SNRC_VERIFY", "0").strip() == "0":
        return
    signature = (
        tuple(qkv_heads.shape),
        qkv_heads.dtype,
        num_heads,
        num_kv_heads,
        head_dim,
        tuple(rope_table.shape),
        rope_table.dtype,
        eps,
        q_weight.dtype,
        k_weight.dtype,
        tuple(q.stride()),
        tuple(k.stride()),
    )
    if signature in _VERIFIED_SIGNATURES:
        return
    rows = min(q.shape[0], 512)
    q_ref_view, k_ref_view, v_ref = qkv_heads[:rows].split([num_heads, num_kv_heads, num_kv_heads], dim=1)
    q_ref, k_ref = reference(
        q_ref_view,
        k_ref_view,
        q_weight,
        k_weight,
        rope_table[:rows],
        eps,
    )
    q_cos, q_ratio = _cosine_and_norm_ratio(q[:rows], q_ref)
    k_cos, k_ratio = _cosine_and_norm_ratio(k[:rows], k_ref)
    q_max = float((q[:rows].float() - q_ref.float()).abs().max())
    k_max = float((k[:rows].float() - k_ref.float()).abs().max())
    v_exact = bool(torch.equal(v[:rows], v_ref))
    passed = q_cos >= 0.99 and k_cos >= 0.99 and 0.9 <= q_ratio <= 1.1 and 0.9 <= k_ratio <= 1.1 and v_exact
    print(
        "[H3_PR792_SNRC_PARITY] "
        f"rank={_rank()} rows={rows} q_cos={q_cos:.9f} k_cos={k_cos:.9f} "
        f"q_norm_ratio={q_ratio:.9f} k_norm_ratio={k_ratio:.9f} "
        f"q_max_abs={q_max:.9f} k_max_abs={k_max:.9f} v_exact={int(v_exact)} passed={int(passed)}",
        flush=True,
    )
    if not passed:
        raise RuntimeError("PR792 EagleSplitNormRopeConcat parity gate failed")
    _VERIFIED_SIGNATURES.add(signature)


def maybe_split_norm_rope_concat(
    qkv: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor | None,
    eps: float,
    reference: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    layout_plan=None,
    enabled: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return fused Q/K/V when explicitly enabled, otherwise use the native Q/K/V path."""
    from vllm_omni.diffusion.layers import minimax_h3_acceleration

    head_major_qkv = minimax_h3_acceleration.head_major_qkv_enabled()
    if not (_enabled() if enabled is None else enabled) or rope_table is None:
        return None
    if qkv.device.type != "npu" or qkv.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("PR792 SNRC requires FP16/BF16 NPU tensors")
    if head_dim != 128 or rope_table.shape[-1] not in (32, 64, 96, 128):
        raise RuntimeError(
            f"PR792 SNRC requires head_dim=128 and rotary_dim in 32/64/96/128, got {head_dim}/{rope_table.shape[-1]}"
        )
    qkv_heads = qkv.view(qkv.shape[0], num_heads + 2 * num_kv_heads, head_dim)
    rope_cos, rope_sin = _rope_caches(rope_table)
    from mindiesd.fused_ops import eagle_split_norm_rope_concat as op

    if layout_plan is None:
        # Preserve the original standalone API's physical storage and fallback decision.
        output_bnsd = head_major_qkv and qkv.dtype == torch.bfloat16 and num_heads == num_kv_heads == 56
    else:
        output_bnsd = bool(
            layout_plan is not None
            and layout_plan.supported
            and qkv.dtype == torch.bfloat16
            and num_heads == num_kv_heads == 56
        )
    op_args = dict(
        qkv=qkv_heads,
        q_weight=q_weight,
        k_weight=k_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        split_sizes=[num_heads, num_kv_heads, num_kv_heads],
        norm_type=3,
        rope_mode=1,
        rotary_dim=rope_table.shape[-1],
        epsilon=eps,
    )
    q, k, v = op(**op_args, output_bnsd=output_bnsd)
    if output_bnsd:
        # Native B1/H/S/D -> logical S/H/D with original global head order.
        q, k, v = (tensor.squeeze(0).transpose(0, 1) for tensor in (q, k, v))
    from vllm_omni.diffusion.layers.minimax_h3_feature_dispatch import count

    count("split_norm_rope_concat_fused")
    count("sdk_e_facade_calls")
    count("snrc_head_major_store_calls" if output_bnsd else "snrc_original_store_calls")
    if minimax_h3_acceleration.head_major_qkv_verify_enabled():
        # The same existing operator and full input, with only the storage flag changed.
        reference_qkv = op(**op_args, output_bnsd=not output_bnsd)
        if not output_bnsd:
            reference_qkv = tuple(tensor.squeeze(0).transpose(0, 1) for tensor in reference_qkv)
        for actual, expected in zip((q, k, v), reference_qkv):
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError("head-major QKV dtype/shape gate failed")
            if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                raise RuntimeError("head-major QKV INT16 bit-view equality gate failed")
        count("snrc_storage_flag_exact_comparison_calls")
        print(
            "[MINIMAX_H3_HEAD_MAJOR_QKV_VERIFY] "
            f"rank={_rank()} rows={qkv_heads.shape[0]} output_bnsd={int(output_bnsd)} qkv_bit_exact=1",
            flush=True,
        )
    calls = int(_STATE["calls"]) + 1
    _STATE["calls"] = calls
    if calls == 1:
        print(
            "[H3_PR792_SNRC_ACTUAL] "
            f"rank={_rank()} op=EagleSplitNormRopeConcat shape={tuple(qkv_heads.shape)} "
            f"heads={num_heads},{num_kv_heads},{num_kv_heads} rotary_dim={rope_table.shape[-1]}",
            flush=True,
        )
    if calls % 400 == 0:
        print(f"[H3_PR792_SNRC_COUNT] rank={_rank()} calls={calls}", flush=True)
    _verify_once(
        q,
        k,
        v,
        qkv_heads,
        num_heads,
        num_kv_heads,
        head_dim,
        q_weight,
        k_weight,
        rope_table,
        eps,
        reference,
    )
    return q, k, v
