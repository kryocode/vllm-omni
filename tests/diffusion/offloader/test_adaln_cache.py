# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AdaLN storage rollback, resident tensor exclusion and A5 cache rejection."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.layerwise_backend as backend_module
from tests.diffusion.offloader.helpers import DummyStream, patch_offload_runtime
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader import adaln_cache
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend, LayerwiseOffloadHook

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


class MixedBlock(nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        # Register the cache first to exercise materialization-probe selection.
        self.cached_weight = nn.Parameter(torch.arange(4.0) + offset, requires_grad=False)
        self.weight = nn.Parameter(torch.arange(6.0) + offset, requires_grad=False)
        self.register_buffer("cached_state", torch.arange(3.0) + offset)
        self.register_buffer("state", torch.arange(2, dtype=torch.int64) + offset)


def resident_ids(blocks):
    return frozenset(id(tensor) for block in blocks for tensor in (block.cached_weight, block.cached_state))


def snapshots(blocks):
    return [
        [
            (tensor, tensor.detach().clone(), tensor.data_ptr(), tensor.stride())
            for tensor in (block.cached_weight, block.cached_state)
        ]
        for block in blocks
    ]


def assert_resident_untouched(saved):
    for tensors in saved:
        for tensor, expected, pointer, stride in tensors:
            assert tensor.data_ptr() == pointer
            assert tensor.stride() == stride
            assert torch.equal(tensor, expected)


@pytest.mark.parametrize("block_count", [2, 3])
def test_ring_preserves_resident_parameters_and_buffers(block_count):
    blocks = nn.ModuleList([MixedBlock(index * 10) for index in range(block_count)])
    saved = snapshots(blocks)
    expected = [(block.weight.detach().clone(), block.state.clone()) for block in blocks]
    hooks = backend_module._install_layerwise_hook_group(
        blocks, torch.device("cpu"), DummyStream(), pin_memory=False, resident_tensor_ids=resident_ids(blocks)
    )

    assert_resident_untouched(saved)
    for hook in hooks:
        assert set(hook.block_parameters) == {"weight"}
        assert set(hook.block_buffers) == {"state"}
        assert set(hook.next_block_parameters) == {"weight"}
        assert set(hook.next_block_buffers) == {"state"}
        assert hook._materialization_probe is hook.block_parameters["weight"]
        assert not hook.is_materialized
        # Only the six FP32 weight values and two INT64 state values cross H2D.
        assert sum(t.numel() * t.element_size() for t in hook.dtype_cpu_flattened_weights.values()) == 40

    for _ in range(3):
        for block, (weight, state) in zip(blocks, expected, strict=True):
            hook = block._hook_registry.get_hook(LayerwiseOffloadHook._HOOK_NAME)
            hook.pre_forward(block)
            assert hook.is_materialized
            assert torch.equal(block.weight, weight)
            assert torch.equal(block.state, state)
            hook.post_forward(block, None)
            hook.offload_layer()
            assert not hook.is_materialized
            assert_resident_untouched(saved)

    for hook in hooks:
        hook.prefetch_layer(non_blocking=False)
        hook.prefetch_layer(non_blocking=False)
        hook.restore_next_block()
        hook.restore_next_block()
        assert_resident_untouched(saved)
    for block, (weight, state) in zip(blocks, expected, strict=True):
        assert torch.equal(block.weight, weight)
        assert torch.equal(block.state, state)


def test_all_resident_block_needs_no_host_master():
    blocks = nn.ModuleList([MixedBlock(), MixedBlock(10)])
    ids = frozenset(id(t) for block in blocks for t in [*block.parameters(), *block.buffers()])
    pointers = [t.data_ptr() for block in blocks for t in [*block.parameters(), *block.buffers()]]
    hooks = backend_module._install_layerwise_hook_group(
        blocks, torch.device("cpu"), DummyStream(), pin_memory=False, resident_tensor_ids=ids
    )
    for hook in hooks:
        assert hook._materialization_probe is None
        assert hook.is_materialized
        assert not hook.dtype_cpu_flattened_weights
        hook.prefetch_layer()
        hook.offload_layer()
        hook.restore_next_block()
    assert [t.data_ptr() for block in blocks for t in [*block.parameters(), *block.buffers()]] == pointers


def test_a5_rejects_cache_before_tensor_selection_or_allocation(monkeypatch):
    from vllm_omni.platforms import npu

    monkeypatch.setattr(npu, "is_a5", lambda device: True)

    def unexpected(*args, **kwargs):
        raise AssertionError("A5 must not select or allocate cache tensors")

    monkeypatch.setattr(adaln_cache, "select_adaln_tensors", unexpected)
    cache = adaln_cache.AdaLNWeightCache(SimpleNamespace(type="npu", index=0))
    assert cache.prepare([], max_gib=25, reserve_gib=8) == frozenset()
    assert not cache.status.enabled and cache.status.reason == "unsupported_on_a5"
    assert not cache.resident_ids


@pytest.fixture(autouse=True)
def local_model_paths(monkeypatch):
    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache_adaln": "true"},
        {"adaln_cache_max_gib": -1},
        {"adaln_cache_max_gib": float("nan")},
        {"adaln_cache_reserve_gib": float("inf")},
        {"adaln_cache_reserve_gib": True},
    ],
)
def test_invalid_cache_options_are_rejected_before_loading(kwargs):
    with pytest.raises((TypeError, ValueError)):
        OmniDiffusionConfig(model="fake-model", enable_layerwise_offload=True, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"enable_cpu_offload": True},
        {"enable_distributed_layerwise_offload": True},
        {"diffusion_offload_config": {"mode": "layer", "components": ["text_encoder"]}},
    ],
)
def test_cache_requires_ordinary_dit_layer_offload(kwargs):
    with pytest.raises(ValueError, match="cache_adaln requires"):
        OmniDiffusionConfig(model="fake-model", cache_adaln=True, **kwargs)
