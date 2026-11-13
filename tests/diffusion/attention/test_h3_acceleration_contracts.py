# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Storage and public-call regressions for the H3 A–H integration."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.layers import minimax_h3_acceleration as features
from vllm_omni.diffusion.layers import minimax_h3_feature_dispatch as dispatch
from vllm_omni.diffusion.layers.npu_weight_layout import (
    permute_host_weight_to_nz,
    unpermute_host_weight_from_nz,
)
from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("offset", [0, 11008])
@pytest.mark.parametrize("transposed", [False, True])
def test_nz_cast_normalizes_copy_source_offset_and_stride(monkeypatch, offset, transposed):
    from vllm_omni.diffusion.layers import npu_weight_layout as layout

    raw = (torch.arange(offset + 64 * 128) % 19).to(torch.bfloat16)
    weight = raw[offset:].view(64, 128)
    if transposed:
        weight = weight.T
    expected = weight.clone()
    destination = torch.empty_like(weight, memory_format=torch.contiguous_format)
    original_copy = torch.Tensor.copy_
    sources = []

    def checked_copy(tensor, source, *args, **kwargs):
        if tensor is destination:
            assert source.storage_offset() == 0
            assert source.is_contiguous()
            sources.append(source)
        return original_copy(tensor, source, *args, **kwargs)

    native = ModuleType("torch_npu")
    setattr(native, "empty_with_format", lambda *args, **kwargs: destination)
    monkeypatch.setitem(sys.modules, "torch_npu", native)
    monkeypatch.setattr(layout, "_is_npu_weight", lambda tensor: True)
    monkeypatch.setattr(torch.Tensor, "copy_", checked_copy)
    actual = layout.maybe_cast_weight_to_nz(weight)
    assert actual is destination and len(sources) == 1
    assert torch.equal(actual, expected) and torch.equal(weight, expected)


@pytest.mark.parametrize("dtype,shape", [(torch.int8, (64, 96)), (torch.bfloat16, (48, 64)), (torch.float16, (32, 48))])
def test_complete_nz_slot_roundtrip_is_bit_exact(dtype, shape):
    source = (torch.arange(shape[0] * shape[1]).reshape(shape) % 117).to(dtype)
    raw = permute_host_weight_to_nz(source)
    assert torch.equal(unpermute_host_weight_from_nz(raw, shape, dtype), source)
    # Scalar coordinates verify the physical [K1,N1,N0,K0] order independently.
    block_n, block_k = (16, 32) if dtype == torch.int8 else (16, 16)
    for n, k in [(0, 0), (17, 33), (shape[0] - 1, shape[1] - 1)]:
        index = (
            ((k // block_k) * (shape[0] // block_n) + n // block_n) * block_n + n % block_n
        ) * block_k + k % block_k
        assert raw[index] == source[n, k]


@pytest.mark.parametrize("transposed", [False, True])
def test_offload_packs_selected_weights_and_restores_original_cpu_master(transposed):
    weight = torch.nn.Parameter(torch.arange(64 * 64).reshape(64, 64).to(torch.int8), requires_grad=False)
    norm = torch.nn.Parameter(torch.randn(64))
    if transposed:
        weight.data = weight.data.t()
    original_stride = weight.stride()
    expected_weight, expected_norm = weight.detach().clone(), norm.detach().clone()
    params = {"mlp.fc1.weight": weight, "norm.weight": norm}
    flat, metadata = LayerwiseOffloadHook._to_cpu(
        params, {}, False, nz_weight_names={"mlp.fc1.weight"}, weight_nz_mode="reactor"
    )
    assert weight.numel() == norm.numel() == 0
    assert torch.equal(flat[torch.int8], permute_host_weight_to_nz(expected_weight.contiguous()))
    hook = object.__new__(LayerwiseOffloadHook)
    hook.dtype_cpu_flattened_weights, hook.dtype_metadata = flat, metadata
    hook.next_block_parameters, hook.next_block_buffers = params, {}
    hook._weight_nz_mode, hook._nz_weight_names = "reactor", {"mlp.fc1.weight"}
    hook.restore_next_block()
    assert torch.equal(weight, expected_weight) and torch.equal(norm, expected_norm)
    assert weight.stride() == original_stride


def test_invalid_nz_slot_rejects_before_clearing_parameters():
    weight = torch.nn.Parameter(torch.ones(17, 33, dtype=torch.int8), requires_grad=False)
    expected = weight.clone()
    with pytest.raises(ValueError, match="aligned"):
        LayerwiseOffloadHook._to_cpu(
            {"weight": weight}, {}, False, nz_weight_names={"weight"}, weight_nz_mode="reactor"
        )
    assert torch.equal(weight, expected)


def test_disabled_modulation_keeps_native_objects_and_call_signature(monkeypatch):
    monkeypatch.setattr(features, "FEATURES", frozenset())
    args = [object() for _ in range(8)]
    called = []

    def native(*values):
        called.append(values)
        return args[0]

    assert dispatch.gate_norm(native, *args) is args[0]
    assert called == [tuple(args)]


def test_modulation_uses_public_keyword_facade_and_refiner_can_decline(monkeypatch):
    monkeypatch.setattr(features, "FEATURES", frozenset({"modulation_fusions"}))
    calls = []

    def fused_modulation(**kw):
        calls.append(kw)
        return ("residual", "normalized")

    library = SimpleNamespace(eagle_indexed_gate_norm_modulate=fused_modulation)
    monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", library)
    args = [SimpleNamespace(dtype=torch.bfloat16)] + [object() for _ in range(6)] + [1e-6]
    assert dispatch.gate_norm(lambda *a: pytest.fail("native fallback"), *args) == ("residual", "normalized")
    assert calls[0]["residual"] is args[0] and calls[0]["indices"] is args[6]
    assert calls[0]["norm_type"] == 0 and calls[0]["epsilon"] == 1e-6
    assert dispatch.gate_norm(lambda *a: "refiner", *args, admitted=False) == "refiner"
    assert len(calls) == 1


def test_request_scope_snapshots_and_restores_layout_after_failure(monkeypatch):
    monkeypatch.setattr(features, "FEATURES", features.KNOWN_FEATURES)
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_HEAD_MAJOR_QKV", "1")
    with pytest.raises(RuntimeError, match="failed"):
        with features.request_layout_scope():
            assert features.head_major_qkv_enabled()
            monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_HEAD_MAJOR_QKV", "0")
            assert features.head_major_qkv_enabled()
            raise RuntimeError("failed")
    assert not features.head_major_qkv_enabled()


def test_qbmm_passes_native_dynamic_quantized_inputs_without_extra_cast(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/models/minimax_h3/npu/qbmm_swiglu.py"
    fake_npu = SimpleNamespace(get_npu_format=lambda weight: 29)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    spec = importlib.util.spec_from_file_location("h3_qbmm_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qx = torch.zeros(8, 64, dtype=torch.int8)
    weight = torch.zeros(64, 128, dtype=torch.int8)
    scale = torch.ones(128)
    token = torch.ones(8)
    expected = torch.zeros(8, 64, dtype=torch.bfloat16)
    calls = []

    def fused_qbmm(**kw):
        calls.append(kw)
        return expected

    monkeypatch.setitem(sys.modules, "mindiesd.fused_ops", SimpleNamespace(eagle_quant_batch_matmul_swiglu=fused_qbmm))
    assert module.fused_fc1(qx, weight, scale, token, torch.bfloat16, None) is expected
    assert calls[0]["x"] is qx and calls[0]["weight"] is weight
    assert calls[0]["pertoken_scale"] is token and calls[0]["scale"] is scale
    assert calls[0]["group_size"] == 0 and not calls[0]["transpose_weight"]
    with pytest.raises(RuntimeError, match="no bias"):
        module.fused_fc1(qx, weight, scale, token, torch.bfloat16, torch.zeros(128))


def test_denoise_timing_counts_actual_sampler_evaluations(monkeypatch, capsys):
    import json

    from tests.diffusion.models.minimax_h3.test_minimax_h3_step_execution import _make_branch, _SegmentMeanModel
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(features, "DIAGNOSTICS", True)
    monkeypatch.setattr(features, "FEATURES", frozenset())
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_ACCELERATION_TIMING", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_HEAD_MAJOR_QKV", "0")
    monkeypatch.setattr(current_omni_platform, "synchronize", lambda: None)
    branch, video, audio = _make_branch(text_len=2, latent_t=1, latent_h=2, latent_w=2, audio_t=2, seed=42)
    model = _SegmentMeanModel()
    minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=video,
        initial_audio_rows=audio,
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
    )
    text = capsys.readouterr().out
    line = next(row for row in text.splitlines() if row.startswith("[MINIMAX_H3_DIT_TIMING] "))
    timing = json.loads(line.split("] ", 1)[1])
    assert len(model.calls) == timing["nfe"] == 2
    assert timing["end_monotonic"] >= timing["begin_monotonic"]
