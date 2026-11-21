# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact storage, admission and lifecycle tests for optional AdaLN caching."""

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.offloader import adaln_cache
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    platform = adaln_cache.current_omni_platform
    patch_offload_runtime(monkeypatch, platform, synchronize=True)
    monkeypatch.setattr(platform, "get_free_memory", lambda device: 64 * 2**30)


def raw(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.adaLN_modulation = nn.Module()
        self.adaLN_modulation.weight = nn.Parameter(torch.arange(6.0).reshape(2, 3).t(), requires_grad=False)
        self.adaLN_modulation.register_buffer("scale", torch.tensor([1.0, 2.0], dtype=torch.float16))
        self.weight = nn.Parameter(torch.arange(4.0), requires_grad=False)


class DiT(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def test_exact_strided_storage_and_restore():
    block = Block()
    saved = [
        (t, t.detach(), t.data_ptr(), t.stride(), raw(t).clone()) for _, t in adaln_cache.select_adaln_tensors([block])
    ]
    cache = adaln_cache.AdaLNWeightCache(torch.device("cpu"))
    ids = cache.prepare([block], max_gib=1, reserve_gib=1)
    assert ids == frozenset(id(t) for t, *_ in saved)
    assert cache.status.enabled and cache.status.tensor_count == 2
    for target, _, pointer, stride, expected in saved:
        assert target.data_ptr() != pointer
        assert target.stride() == stride
        assert torch.equal(raw(target), expected)
    cache.restore()
    assert not cache.resident_ids
    for target, _, pointer, stride, expected in saved:
        assert target.data_ptr() == pointer and target.stride() == stride
        assert torch.equal(raw(target), expected)


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_quantized_weight_and_scale_dtypes_are_preserved(dtype):
    module = nn.Module()
    module.adaln = nn.Module()
    module.adaln.weight = nn.Parameter(torch.arange(8.0).to(dtype), requires_grad=False)
    module.adaln.register_buffer("weight_scale", torch.tensor([0.5], dtype=torch.float32))
    if hasattr(torch, "float8_e8m0fnu"):
        module.adaln.register_buffer("mx_scale", torch.tensor([127, 128], dtype=torch.uint8).view(torch.float8_e8m0fnu))
    expected = {name: (t.dtype, t.stride(), raw(t).clone()) for name, t in adaln_cache.select_adaln_tensors([module])}
    cache = adaln_cache.AdaLNWeightCache(torch.device("cpu"))
    cache.prepare([module], max_gib=1, reserve_gib=1)
    assert cache.status.enabled
    for name, tensor in adaln_cache.select_adaln_tensors([module]):
        old_dtype, old_stride, old_bytes = expected[name]
        assert tensor.dtype == old_dtype and tensor.stride() == old_stride
        assert torch.equal(raw(tensor), old_bytes)


def test_shared_tensors_are_cached_once():
    block = Block()
    assert len(adaln_cache.select_adaln_tensors([block, block])) == 2


def test_activation_buffers_are_not_selected_as_static_weights():
    block = Block()
    block.adaLN_modulation.register_buffer("activation_cache", torch.ones(16))
    selected = {id(tensor) for _, tensor in adaln_cache.select_adaln_tensors([block])}
    assert id(block.adaLN_modulation.activation_cache) not in selected
    assert id(block.adaLN_modulation.scale) in selected


@pytest.mark.parametrize(
    "case,reason",
    [
        ("budget", "cache_budget_exceeded"),
        ("free", "insufficient_free_memory"),
        ("query", "memory_query_unavailable"),
        ("post_alloc", "post_allocation_reserve_exceeded"),
    ],
)
def test_admission_failure_keeps_original_model_storage(monkeypatch, case, reason):
    block = Block()
    originals = [(t, t.data_ptr(), raw(t).clone()) for _, t in adaln_cache.select_adaln_tensors([block])]
    platform = adaln_cache.current_omni_platform
    if case == "free":
        monkeypatch.setattr(platform, "get_free_memory", lambda device: 0)
    elif case == "query":

        def unavailable(device):
            raise NotImplementedError

        monkeypatch.setattr(platform, "get_free_memory", unavailable)
    elif case == "post_alloc":
        values = iter([64 * 2**30, 0])
        monkeypatch.setattr(platform, "get_free_memory", lambda device: next(values))
    cache = adaln_cache.AdaLNWeightCache(torch.device("cpu"))
    assert not cache.prepare([block], max_gib=0 if case == "budget" else 1, reserve_gib=1)
    assert not cache.status.enabled and cache.status.reason == reason
    for target, pointer, expected in originals:
        assert target.data_ptr() == pointer and torch.equal(raw(target), expected)


@pytest.mark.parametrize("first_contiguous", [False, True])
def test_second_allocation_failure_rolls_back_without_changing_values(monkeypatch, first_contiguous):
    block = Block()
    if first_contiguous:
        block.adaLN_modulation.weight = nn.Parameter(block.adaLN_modulation.weight.contiguous(), requires_grad=False)
    originals = [(t, t.data_ptr(), raw(t).clone()) for _, t in adaln_cache.select_adaln_tensors([block])]
    original_empty = torch.empty_strided
    original_to = torch.Tensor.to
    calls = 0

    def fail_second(operation, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("injected allocation failure")
        return operation(*args, **kwargs)

    def injected_empty(*args, **kwargs):
        return fail_second(original_empty, *args, **kwargs)

    def injected_to(tensor, *args, **kwargs):
        if kwargs.get("copy"):
            return fail_second(original_to, tensor, *args, **kwargs)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "empty_strided", injected_empty)
    monkeypatch.setattr(torch.Tensor, "to", injected_to)
    cache = adaln_cache.AdaLNWeightCache(torch.device("cpu"))
    assert not cache.prepare([block], max_gib=1, reserve_gib=1)
    assert cache.status.reason == "cache_allocation_or_copy_failed"
    for target, pointer, expected in originals:
        assert target.data_ptr() == pointer and torch.equal(raw(target), expected)


def test_contiguous_singleton_layout_preserves_noncanonical_stride():
    module = nn.Module()
    module.adaln = nn.Module()
    value = torch.arange(2.0).as_strided((2, 1), (1, 100))
    module.adaln.weight = nn.Parameter(value, requires_grad=False)
    original = module.adaln.weight.detach()
    assert original.is_contiguous()
    cache = adaln_cache.AdaLNWeightCache(torch.device("cpu"))
    cache.prepare([module], max_gib=1, reserve_gib=1)
    assert cache.status.enabled
    assert module.adaln.weight.stride() == original.stride()
    assert torch.equal(raw(module.adaln.weight), raw(original))
    cache.restore()
    assert module.adaln.weight.data_ptr() == original.data_ptr()


@pytest.mark.parametrize("after_commit", [False, True])
def test_partial_storage_commit_failure_restores_every_tensor(monkeypatch, after_commit):
    block = Block()
    saved = [(t, t.data_ptr(), raw(t).clone()) for _, t in adaln_cache.select_adaln_tensors([block])]
    original = adaln_cache.set_tensor_storage
    calls = 0

    def injected(target, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            if after_commit:
                original(target, value)
            raise RuntimeError("injected storage commit failure")
        original(target, value)

    monkeypatch.setattr(adaln_cache, "set_tensor_storage", injected)
    cache = adaln_cache.AdaLNWeightCache(torch.device("cpu"))
    assert not cache.prepare([block], max_gib=1, reserve_gib=1)
    for tensor, pointer, expected in saved:
        assert tensor.data_ptr() == pointer and torch.equal(raw(tensor), expected)


def test_invalid_post_allocation_telemetry_falls_back(monkeypatch):
    values = iter([64 * 2**30, None])
    monkeypatch.setattr(adaln_cache.current_omni_platform, "get_free_memory", lambda device: next(values))
    cache = adaln_cache.AdaLNWeightCache(torch.device("cpu"))
    assert not cache.prepare([Block()], max_gib=1, reserve_gib=1)
    assert cache.status.reason == "cache_allocation_or_copy_failed"


def test_budget_counts_strided_storage_span():
    block = Block()
    block.adaLN_modulation.weight = nn.Parameter(torch.ones(16).as_strided((2, 3), (8, 2)), requires_grad=False)
    cache = adaln_cache.AdaLNWeightCache(torch.device("cpu"))
    assert not cache.prepare([block], max_gib=32 / 2**30, reserve_gib=0)
    assert cache.status.reason == "cache_budget_exceeded" and cache.status.bytes > 32


@pytest.mark.parametrize("enabled,budget", [(True, 1), (True, 0), (False, 1)])
def test_backend_installs_cache_or_falls_back_and_preserves_repeated_execution(enabled, budget):
    pipeline = nn.Module()
    pipeline.transformer = DiT()
    blocks = pipeline.transformer.blocks
    expected = {id(t): raw(t).clone() for t in pipeline.parameters()}
    backend = LayerWiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.LAYER_WISE,
            pin_cpu_memory=False,
            cache_adaln=enabled,
            adaln_cache_max_gib=budget,
            adaln_cache_reserve_gib=1,
        ),
        torch.device("cpu"),
    )
    backend.enable(pipeline)
    assert backend.enabled and len(backend._dit_hooks) == 2
    assert backend.adaln_cache_status.enabled == (enabled and budget > 0)
    for _ in range(2):
        for block, hook in zip(blocks, backend._dit_hooks, strict=True):
            hook.pre_forward(block)
            for tensor in block.parameters():
                assert torch.equal(raw(tensor), expected[id(tensor)])
            hook.post_forward(block, None)
    backend.disable()
    assert not backend.enabled and not backend.resident_tensor_ids
    for tensor in pipeline.parameters():
        assert torch.equal(raw(tensor), expected[id(tensor)])


def test_backend_shutdown_releases_cache_to_original_storage():
    pipeline = nn.Module()
    pipeline.transformer = DiT()
    cached = adaln_cache.select_adaln_tensors([pipeline.transformer])
    pointers = {id(t): t.data_ptr() for _, t in cached}
    backend = LayerWiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.LAYER_WISE,
            pin_cpu_memory=False,
            cache_adaln=True,
            adaln_cache_max_gib=1,
            adaln_cache_reserve_gib=1,
        ),
        torch.device("cpu"),
    )
    backend.enable(pipeline)
    backend.shutdown()
    assert not backend.enabled and not backend._adaln_cache.resident_ids
    for _, tensor in cached:
        assert tensor.data_ptr() == pointers[id(tensor)]


def test_backend_disable_restores_original_cache_storage():
    pipeline = nn.Module()
    pipeline.transformer = DiT()
    cached = adaln_cache.select_adaln_tensors([pipeline.transformer])
    pointers = {id(t): t.data_ptr() for _, t in cached}
    backend = LayerWiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.LAYER_WISE,
            pin_cpu_memory=False,
            cache_adaln=True,
            adaln_cache_max_gib=1,
            adaln_cache_reserve_gib=1,
        ),
        torch.device("cpu"),
    )
    backend.enable(pipeline)
    backend.disable()
    for _, tensor in cached:
        assert tensor.data_ptr() == pointers[id(tensor)]


def test_disabled_cache_never_queries_device_memory(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled cache consulted memory")

    monkeypatch.setattr(adaln_cache.current_omni_platform, "get_free_memory", unexpected)
    pipeline = nn.Module()
    pipeline.transformer = DiT()
    backend = LayerWiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.LAYER_WISE, pin_cpu_memory=False), torch.device("cpu")
    )
    backend.enable(pipeline)
    backend.disable()
