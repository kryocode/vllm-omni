# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native eight-card checks for NZ storage and existing SDK fused operators."""

import json
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]


def _native_worker(rank: int, output: str) -> None:
    import torch_npu
    from mindiesd.fused_ops import eagle_quant_batch_matmul_swiglu, eagle_split_norm_rope_concat

    from vllm_omni.diffusion.layers.npu_weight_layout import nz_view_over_buffer, permute_host_weight_to_nz

    torch.npu.set_device(rank)
    torch.npu.config.allow_internal_format = True
    device = torch.device("npu", rank)
    generator = torch.Generator().manual_seed(42 + rank)
    weight_cpu = torch.randint(-8, 8, (64, 128), dtype=torch.int8, generator=generator)
    nd = weight_cpu.to(device)
    owner = permute_host_weight_to_nz(weight_cpu).to(device)
    nz = nz_view_over_buffer(owner, 0, weight_cpu.shape, torch.int8)
    parameter = torch.nn.Parameter(torch.empty_like(nd), requires_grad=False)
    parameter.data = nz
    assert torch_npu.get_npu_format(nz) == torch_npu.get_npu_format(parameter) == 29
    assert torch.equal(parameter.cpu(), weight_cpu)
    x = torch.randint(-8, 8, (256, 64), dtype=torch.int8, generator=generator).to(device)
    scale = torch.ones(128, device=device, dtype=torch.float32)
    token = torch.ones(256, device=device, dtype=torch.float32)
    expected = torch_npu.npu_quant_matmul(x, nd, scale, pertoken_scale=token, output_dtype=torch.bfloat16)
    actual = torch_npu.npu_quant_matmul(x, parameter, scale, pertoken_scale=token, output_dtype=torch.bfloat16)
    assert torch.equal(actual, expected)
    g = eagle_quant_batch_matmul_swiglu(
        x=x,
        weight=parameter,
        scale=scale,
        pertoken_scale=token,
        output_dtype=torch.bfloat16,
        transpose_weight=False,
        group_size=0,
    )
    g_ref = torch_npu.npu_swiglu(expected, dim=-1)
    a, b = g.float().flatten(), g_ref.float().flatten()
    cosine = float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)).clamp_min(1e-12))
    assert torch.isfinite(g).all() and cosine >= 0.99
    qkv = torch.randn((256, 168, 128), generator=generator, dtype=torch.bfloat16).to(device)
    weights = torch.ones(128, dtype=torch.bfloat16, device=device)
    angles = torch.linspace(0, 1, 256 * 48, device=device).reshape(256, 48)
    kwargs = dict(
        qkv=qkv,
        q_weight=weights,
        k_weight=weights,
        rope_cos=angles.cos().bfloat16(),
        rope_sin=angles.sin().bfloat16(),
        split_sizes=[56, 56, 56],
        norm_type=3,
        rope_mode=1,
        rotary_dim=96,
        epsilon=1e-6,
    )
    normal = eagle_split_norm_rope_concat(**kwargs, output_bnsd=False)
    head_major = eagle_split_norm_rope_concat(**kwargs, output_bnsd=True)
    for plain, physical in zip(normal, head_major, strict=True):
        logical = physical.squeeze(0).transpose(0, 1)
        assert plain.shape == logical.shape and plain.dtype == logical.dtype
        assert torch.equal(plain.view(torch.int16), logical.view(torch.int16))
    torch.npu.synchronize()
    Path(output, f"rank{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                nz_format=29,
                host_pack_cpu_roundtrip=True,
                nz_vs_nd_matmul_bit_exact=True,
                qbmm_swiglu_cosine=cosine,
                snrc_storage_flags_bit_exact=True,
            )
        )
        + "\n"
    )


@hardware_test(res={"npu": "A2"}, num_cards=8)
def test_native_nz_and_fused_storage_contracts(tmp_path):
    if not hasattr(torch, "npu") or torch.npu.device_count() < 8:
        pytest.skip("requires eight visible Ascend cards and matching SDK native package")
    pytest.importorskip("mindiesd.fused_ops")
    context = mp.spawn(_native_worker, args=(str(tmp_path),), nprocs=8, join=False)
    try:
        deadline = time.monotonic() + 240
        while not context.join(timeout=min(30, max(0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                pytest.fail("native operator workers exceeded their deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(8)]
    assert {row["rank"] for row in records} == set(range(8))
