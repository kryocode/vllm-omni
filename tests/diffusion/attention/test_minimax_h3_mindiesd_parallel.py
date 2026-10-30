# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU admission tests; mocked MindIE calls do not claim NPU numerical coverage."""

import ast
import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "vllm_omni/diffusion/models/minimax_h3"
HELPER_NAME = "vllm_omni.diffusion.models.minimax_h3.attention.mindiesd_parallel"


def make_metadata(**kwargs):
    return SimpleNamespace(joint_query=None, joint_key=None, joint_value=None, joint_attn_mask=None, **kwargs)


@pytest.fixture
def adapter(monkeypatch):
    forward_context = ModuleType("vllm_omni.diffusion.forward_context")
    setattr(forward_context, "get_ulysses_mode", lambda **_: "strict")
    platforms = ModuleType("vllm_omni.platforms")
    setattr(platforms, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))
    monkeypatch.setitem(sys.modules, forward_context.__name__, forward_context)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    spec = importlib.util.spec_from_file_location(HELPER_NAME, MODEL_DIR / "attention/mindiesd_parallel.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, HELPER_NAME, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)
    group = object()
    coordinator = SimpleNamespace(ring_world_size=1, allgather_world_size=1)
    strategy = SimpleNamespace(name="ulysses", _ulysses_pg=group, _sp_group=coordinator, _scatter_idx=2, _gather_idx=1)
    calls = []
    states = []

    def new_state(**kwargs):
        state = SimpleNamespace(**kwargs)
        states.append(state)
        return state

    def distributed_attention(q, k, v, spans, **kwargs):
        calls.append((q, k, v, spans, kwargs))
        return q + 1

    library = ModuleType("mindiesd.parallel")
    setattr(library, "ParallelState", new_state)
    setattr(library, "distributed_sparse_attention", distributed_attention)
    monkeypatch.setitem(sys.modules, "mindiesd.parallel", library)
    plan = SimpleNamespace(used_len=14, video_spans=None, prefix_len=6, latent_shape=(2, 2, 2))
    impl = SimpleNamespace(
        rainfusion=SimpleNamespace(precision="bf16", sparsity=0.8),
        softmax_scale=0.125,
        _resolve_plan=lambda metadata: plan,
    )

    def original_attention(q, k, v, metadata):
        raise AssertionError("Omni must not execute its outer all-to-all on an admitted call")

    attention = original_attention
    attention.__dict__.update(
        attn_backend=SimpleNamespace(
            get_name=lambda: "RAINFUSION_ATTN", supports_packed_mask_free=lambda: False, supports_prefix_kv_slicing=True
        ),
        causal=False,
        qkv_layout="BSND",
        _has_custom_attention=False,
        is_paged_kv_active=lambda: False,
        paged_kv_cache_role=None,
        _get_active_parallel_strategy=lambda: strategy,
        _with_kv_cache_dtype=lambda metadata: metadata,
        attention=impl,
    )
    metadata = make_metadata(extra={})
    q = torch.zeros((1, 8, 4, 8), dtype=torch.bfloat16)
    return SimpleNamespace(
        module=module,
        attention=attention,
        metadata=metadata,
        query=q,
        calls=calls,
        states=states,
        strategy=strategy,
        coordinator=coordinator,
        group=group,
        plan=plan,
        impl=impl,
        library=library,
    )


def run(adapter, **kwargs):
    q = kwargs.get("query", adapter.query)
    return adapter.module.try_distributed_sparse_attention(
        kwargs.get("attention", adapter.attention), q, kwargs.get("key", q), kwargs.get("value", q), adapter.metadata
    )


@pytest.mark.parametrize("precision", ["bf16", "mix"])
def test_precision_and_global_geometry_are_explicit(adapter, precision):
    adapter.impl.rainfusion.precision = precision
    out = run(adapter)
    torch.testing.assert_close(out, adapter.query + 1)
    q, k, v, spans, kwargs = adapter.calls[0]
    assert q is k is v is adapter.query
    assert spans == [{"start": 6, "latent_shape": [2, 2, 2]}]
    assert kwargs == dict(
        group=adapter.group,
        used_len=14,
        state=adapter.states[0],
        precision=precision,
        scale=0.125,
        sparsity=0.8,
    )


@pytest.mark.parametrize("effective,expected", [("mix", "mix"), ("bf16", "bf16"), ("float", "bf16")])
def test_effective_attention_quantization_policy_wins(adapter, effective, expected):
    adapter.impl.rainfusion.precision = "mix"
    seen = []

    def apply_policy(metadata):
        seen.append(metadata)
        return make_metadata(extra={"kv_cache_dtype": effective})

    adapter.attention._with_kv_cache_dtype = apply_policy
    run(adapter)
    assert seen == [adapter.metadata]
    assert adapter.calls[0][-1]["precision"] == expected


def test_state_reused_across_layers_on_the_same_communicator(adapter):
    run(adapter)
    second_layer = SimpleNamespace(**adapter.attention.__dict__)
    run(adapter, attention=second_layer)
    assert len(adapter.states) == 1
    assert adapter.calls[0][-1]["state"] is adapter.calls[1][-1]["state"]
    adapter.strategy._ulysses_pg = object()
    run(adapter)
    assert len(adapter.states) == 2


def test_overlap_is_explicit_and_cached_state_cannot_change_mode(adapter):
    adapter.impl.rainfusion.overlap_comm = True
    run(adapter)
    run(adapter)
    assert len(adapter.states) == 1
    assert adapter.states[0].overlap_comm is True
    adapter.impl.rainfusion.overlap_comm = False
    with pytest.raises(RuntimeError, match="overlap_comm"):
        run(adapter)
    assert len(adapter.calls) == 2


@pytest.mark.parametrize("value", [False, True, "false", 1, None])
def test_backend_overlap_config_preserves_booleans_and_rejects_ambiguous_values(value):
    # Isolate the actual configuration class from backend imports requiring NPU.
    path = ROOT / "vllm_omni/diffusion/attention/backends/rainfusion_attn.py"
    config = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "RainFusionConfig"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, config], type_ignores=[]))
    namespace = {"dataclass": dataclasses.dataclass, "__name__": __name__}
    exec(compile(module, str(path), "exec"), namespace)
    from_backend_kwargs = getattr(namespace["RainFusionConfig"], "from_backend_kwargs")
    assert from_backend_kwargs({}).overlap_comm is False
    if isinstance(value, bool):
        assert from_backend_kwargs({"overlap_comm": value}).overlap_comm is value
    else:
        with pytest.raises(ValueError, match="boolean"):
            from_backend_kwargs({"overlap_comm": value})


def test_multi_video_spans_are_forwarded_without_reconstructing_a_tail(adapter):
    adapter.plan.video_spans = [{"start": 1, "latent_shape": [1, 2, 2]}, {"start": 6, "latent_shape": [2, 2, 2]}]
    run(adapter)
    assert adapter.calls[0][3] is adapter.plan.video_spans


@pytest.mark.parametrize(
    "change",
    [
        lambda a: setattr(a.strategy, "name", "none"),
        lambda a: setattr(a.coordinator, "ring_world_size", 2),
        lambda a: setattr(a.coordinator, "allgather_world_size", 2),
        lambda a: setattr(a.strategy, "_scatter_idx", 1),
        lambda a: setattr(a.strategy, "_ulysses_pg", None),
        lambda a: setattr(a.module, "get_ulysses_mode", lambda **_: "advanced_uaa"),
        lambda a: setattr(a.attention, "causal", True),
        lambda a: setattr(a.attention, "_has_custom_attention", True),
        lambda a: setattr(a.attention, "is_paged_kv_active", lambda: True),
        lambda a: setattr(a.attention, "paged_kv_cache_role", "read"),
        lambda a: setattr(a.impl, "_resolve_plan", lambda metadata: None),
        lambda a: setattr(a.plan, "used_len", 17),
        lambda a: setattr(a.impl.rainfusion, "precision", "fp8"),
        lambda a: setattr(a.metadata, "joint_key", a.query),
        lambda a: a.metadata.extra.update(minimax_h3_reference_kv_post_parallel=(object(), 1, False)),
        lambda a: a.metadata.extra.update(gate_compress=a.query),
        lambda a: a.metadata.extra.update(vdn_window=object()),
    ],
)
def test_unsupported_calls_decline_before_library_or_state_creation(adapter, change):
    change(adapter)
    assert run(adapter) is None
    assert not adapter.calls and not adapter.states


@pytest.mark.parametrize(
    "shape,dtype", [((2, 8, 4, 8), torch.bfloat16), ((1, 8, 3, 8), torch.bfloat16), ((1, 8, 4, 8), torch.float16)]
)
def test_unsupported_tensor_contract_declines(adapter, shape, dtype):
    assert run(adapter, query=torch.zeros(shape, dtype=dtype)) is None
    assert not adapter.calls


def test_unequal_qkv_declines(adapter):
    assert run(adapter, key=adapter.query[:, :4]) is None
    assert not adapter.calls


def test_collective_error_propagates_without_retry(adapter):
    def failed_collective(*args, **kwargs):
        raise RuntimeError("collective failed")

    adapter.library.distributed_sparse_attention = failed_collective
    with pytest.raises(RuntimeError, match="collective failed"):
        run(adapter)


def packed_method():
    """Execute the production eager island without importing the model runner."""
    tree = ast.parse((MODEL_DIR / "minimax_h3_transformer.py").read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run_packed_attention"
    )
    method.decorator_list = []
    source = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = dict(
        __package__="vllm_omni.diffusion.models.minimax_h3",
        torch=torch,
        AttentionMetadata=make_metadata,
        _ring_sequence_parallel_is_active=lambda attention: False,
        MINIMAX_H3_LASER_INPUT_SCALE=256.0,
    )
    exec(compile(ast.fix_missing_locations(source), str(MODEL_DIR / "minimax_h3_transformer.py"), "exec"), namespace)
    return namespace["_run_packed_attention"]


def test_model_admitted_call_bypasses_original_outer_a2a(adapter):
    holder = SimpleNamespace(
        attention=adapter.attention, vsa_sparsity=None, vsa_tile_size=None, vsa_ref_policy=None, vsa_ref_keep_rate=None
    )
    q = adapter.query.squeeze(0)
    out = packed_method()(holder, q, q, q, cu_seqlens=torch.tensor([0, 14]), max_seqlen=14, packed_total=16)
    torch.testing.assert_close(out, q + 1)
    assert len(adapter.calls) == 1


def test_model_declined_call_uses_original_attention(adapter):
    calls = []

    def original(q, k, v, metadata):
        calls.append(metadata)
        return q + 2

    original.__dict__.update(adapter.attention.__dict__)
    adapter.strategy.name = "none"
    holder = SimpleNamespace(
        attention=original, vsa_sparsity=None, vsa_tile_size=None, vsa_ref_policy=None, vsa_ref_keep_rate=None
    )
    q = adapter.query.squeeze(0)
    out = packed_method()(holder, q, q, q, cu_seqlens=torch.tensor([0, 14]), max_seqlen=14, packed_total=16)
    torch.testing.assert_close(out, q + 2)
    assert len(calls) == 1 and not adapter.calls
