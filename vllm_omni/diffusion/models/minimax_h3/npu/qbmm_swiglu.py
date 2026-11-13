# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""INT8 fc1 QBMM + FP32 SwiGLU; dynamic quantization and fc2 stay native."""

import json
import os

import torch
import torch_npu

ND_FORMAT = 2
NZ_FORMAT = 29

_VERIFIED_SIGNATURES = set()
_LOGGED_SIGNATURES = set()


def fused_fc1(qx, weight, scale, token, dtype, bias):
    if bias is not None or dtype != torch.bfloat16:
        raise RuntimeError("PR826 H3 adapter requires no bias and BF16 output")
    if weight.ndim != 2 or qx.ndim != 2 or qx.shape[-1] != weight.shape[0]:
        raise RuntimeError("fc1 activation and weight K dimensions must agree")
    weight_format = torch_npu.get_npu_format(weight)
    if weight_format not in (ND_FORMAT, NZ_FORMAT):
        raise RuntimeError("QBMM SwiGLU weights require ND or FRACTAL_NZ storage")
    from mindiesd.fused_ops import eagle_quant_batch_matmul_swiglu

    out = eagle_quant_batch_matmul_swiglu(
        x=qx,
        weight=weight,
        scale=scale,
        offset=None,
        bias=None,
        pertoken_scale=token,
        output_dtype=dtype,
        transpose_x=False,
        transpose_weight=False,
        group_size=0,
    )
    from vllm_omni.diffusion.layers.minimax_h3_feature_dispatch import count

    count("qbmm_swiglu_fused")
    count("sdk_g_facade_calls")
    signature = (tuple(qx.shape), tuple(weight.shape), dtype, weight_format)
    if signature not in _LOGGED_SIGNATURES:
        print(
            "[H3_PR826_ACTUAL] "
            + json.dumps(
                {
                    "rank": int(os.environ.get("RANK", "-1")),
                    "op": "EagleQuantBatchMatmulSwigluWeightNz"
                    if weight_format == NZ_FORMAT
                    else "EagleQuantBatchMatmulSwiglu",
                    "x": list(qx.shape),
                    "weight": list(weight.shape),
                    "format": weight_format,
                    "output": list(out.shape),
                    "dtype": str(out.dtype),
                }
            ),
            flush=True,
        )
        _LOGGED_SIGNATURES.add(signature)
    if signature not in _VERIFIED_SIGNATURES and os.environ.get("VLLM_OMNI_MINIMAX_H3_QBMM_SWIGLU_VERIFY", "0") == "1":
        native = torch_npu.npu_quant_matmul(qx, weight, scale, pertoken_scale=token, output_dtype=dtype)
        native = torch_npu.npu_swiglu(native, dim=-1)
        a = out.float().flatten()
        b = native.float().flatten()
        an = torch.linalg.vector_norm(a)
        bn = torch.linalg.vector_norm(b)
        values = {
            "rank": int(os.environ.get("RANK", "-1")),
            "cosine": float(torch.dot(a, b) / (an * bn).clamp_min(1e-12)),
            "norm_ratio": float(an / bn.clamp_min(1e-12)),
            "max_abs": float((a - b).abs().max()),
            "finite": bool(torch.isfinite(a).all()),
            "bit_exact": bool(torch.equal(out, native)),
        }
        assert values["finite"] and values["cosine"] >= 0.99 and 0.9 <= values["norm_ratio"] <= 1.1, values
        print("[H3_PR826_NUMERICAL_SCREEN] " + json.dumps(values), flush=True)
        _VERIFIED_SIGNATURES.add(signature)
    return out
