# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU regressions for caller-owned resident weights in ordinary layerwise offload."""

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.layerwise_backend as backend_module
from tests.diffusion.offloader.helpers import DummyStream, patch_offload_runtime
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend, LayerwiseOffloadHook

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    patch_offload_runtime(monkeypatch, backend_module.current_omni_platform)


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


@pytest.mark.parametrize("probe_name", ["cached_weight", "cached_state"])
def test_explicit_resident_probe_is_replaced(probe_name):
    current, following = MixedBlock(), MixedBlock(10)
    hook = LayerwiseOffloadHook(
        following,
        torch.device("cpu"),
        DummyStream(),
        pin_memory=False,
        materialization_probe_tensor=getattr(current, probe_name),
        resident_tensor_ids=resident_ids([current, following]),
    )
    hook.initialize_hook(current)
    assert hook._materialization_probe is current.weight
    hook.offload_layer()
    assert not hook.is_materialized
    assert current.cached_weight.numel() == 4
    assert current.cached_state.numel() == 3


def test_ring_uses_streamed_buffer_probe_when_all_parameters_are_resident():
    blocks = nn.ModuleList([MixedBlock(), MixedBlock(10)])
    ids = resident_ids(blocks) | frozenset(id(block.weight) for block in blocks)
    hooks = backend_module._install_layerwise_hook_group(
        blocks, torch.device("cpu"), DummyStream(), pin_memory=False, resident_tensor_ids=ids
    )
    assert all(not hook.is_materialized for hook in hooks)
    for block in blocks:
        hook = block._hook_registry.get_hook(LayerwiseOffloadHook._HOOK_NAME)
        assert hook._materialization_probe is block.state
        hook.pre_forward(block)
        assert hook.is_materialized
        assert block.weight.numel() == 6
        hook.offload_layer()
        assert not hook.is_materialized
    for hook in hooks:
        hook.restore_next_block()


@pytest.mark.parametrize("ids", [None, frozenset()])
def test_default_ring_offloads_all_tensors(ids):
    blocks = nn.ModuleList([MixedBlock(), MixedBlock(10)])
    expected = [
        {name: tensor.detach().clone() for name, tensor in [*block.named_parameters(), *block.named_buffers()]}
        for block in blocks
    ]
    hooks = backend_module._install_layerwise_hook_group(
        blocks, torch.device("cpu"), DummyStream(), pin_memory=False, resident_tensor_ids=ids
    )
    assert all(tensor.numel() == 0 for block in blocks for tensor in [*block.parameters(), *block.buffers()])
    for hook in hooks:
        hook.restore_next_block()
    for block, original in zip(blocks, expected, strict=True):
        for name, tensor in [*block.named_parameters(), *block.named_buffers()]:
            assert torch.equal(tensor, original[name])


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


def test_backend_passes_resident_ids_and_preserves_enable_disable_cycles():
    class MixedDiT(nn.Module):
        _layerwise_offload_blocks_attrs = ["blocks"]

    pipeline = nn.Module()
    pipeline.transformer = MixedDiT()
    blocks = pipeline.transformer.blocks = nn.ModuleList([MixedBlock(), MixedBlock(10)])
    saved = snapshots(blocks)
    backend = LayerWiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.LAYER_WISE, pin_cpu_memory=False, components=frozenset({"dit"})),
        torch.device("cpu"),
        resident_tensor_ids=resident_ids(blocks),
    )
    for _ in range(2):
        backend.enable(pipeline)
        assert backend.enabled
        assert len(backend._dit_hooks) == 2
        assert all(set(hook.block_parameters) == {"weight"} for hook in backend._dit_hooks)
        assert_resident_untouched(saved)
        backend.disable()
        assert not backend.enabled
        assert_resident_untouched(saved)
