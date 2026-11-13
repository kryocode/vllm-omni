# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU admission tests; mocked MindIE calls do not claim NPU numerical coverage."""

import ast
import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "vllm_omni/diffusion/models/minimax_h3"
HELPER_NAME = "vllm_omni.diffusion.models.minimax_h3.attention.mindiesd_parallel"


def make_metadata(**kwargs):
    defaults = dict(
        joint_query=None,
        joint_key=None,
        joint_value=None,
        joint_attn_mask=None,
        full_attn_spans=None,
        attn_mask=None,
        video_layout=None,
    )
    return SimpleNamespace(**(defaults | kwargs))


@pytest.fixture
def adapter(monkeypatch):
    layers = ModuleType("vllm_omni.diffusion.layers")
    feature_name = "vllm_omni.diffusion.layers.minimax_h3_acceleration"
    feature_spec = importlib.util.spec_from_file_location(
        feature_name, ROOT / "vllm_omni/diffusion/layers/minimax_h3_acceleration.py"
    )
    assert feature_spec is not None and feature_spec.loader is not None
    feature_module = importlib.util.module_from_spec(feature_spec)
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_ACCELERATION_FEATURES", "")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_ACCELERATION_DIAGNOSTICS", "0")
    monkeypatch.setitem(sys.modules, feature_name, feature_module)
    feature_spec.loader.exec_module(feature_module)
    setattr(layers, "minimax_h3_acceleration", feature_module)
    monkeypatch.setitem(sys.modules, layers.__name__, layers)
    config_path = ROOT / "vllm_omni/diffusion/distributed/mindiesd_config.py"
    config_name = "vllm_omni.diffusion.distributed.mindiesd_config"
    config_spec = importlib.util.spec_from_file_location(config_name, config_path)
    assert config_spec is not None and config_spec.loader is not None
    config_module = importlib.util.module_from_spec(config_spec)
    monkeypatch.setitem(sys.modules, config_name, config_module)
    config_spec.loader.exec_module(config_module)
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
    resources_path = ROOT / "vllm_omni/diffusion/distributed/resources.py"
    spec_resources = importlib.util.spec_from_file_location("h3_test_resources", resources_path)
    assert spec_resources is not None and spec_resources.loader is not None
    resources_module = importlib.util.module_from_spec(spec_resources)
    spec_resources.loader.exec_module(resources_module)
    coordinator = SimpleNamespace(
        ring_world_size=1, allgather_world_size=1, resources=resources_module.SequenceParallelResources()
    )
    strategy = SimpleNamespace(
        name="ulysses",
        _ulysses_pg=group,
        _sp_group=coordinator,
        _scatter_idx=2,
        _gather_idx=1,
        _ulysses_a2a_permute=False,
    )
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
    setattr(library, "plan_qkv_layout", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "mindiesd.parallel", library)
    plan = SimpleNamespace(used_len=14, video_spans=None, prefix_len=6, latent_shape=(2, 2, 2))
    impl = SimpleNamespace(
        rainfusion=SimpleNamespace(
            precision="bf16", sparsity=0.8, parallel=module.MindIEParallelConfig(), overlap_comm=False
        ),
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
        effective_attention=lambda: (impl, SimpleNamespace(get_name=lambda: "RAINFUSION_ATTN"), None),
        paged_kv_cache_role=None,
        _get_active_parallel_strategy=lambda: strategy,
        _with_kv_cache_dtype=lambda metadata: metadata,
        attention=impl,
    )
    metadata = make_metadata(extra={}, full_attn_spans=None, attn_mask=None, video_layout=None)
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
        sparse_type="rf_v3",
        pre_rearranged=False,
        scale=0.125,
        sparsity=0.8,
    )


@pytest.mark.parametrize("effective,expected", [("mix", "mix"), ("bf16", "bf16"), ("float", "bf16")])
def test_effective_attention_quantization_policy_wins(adapter, effective, expected):
    adapter.impl.rainfusion.precision = "mix"
    seen = []

    def apply_policy(metadata):
        seen.append(metadata)
        return make_metadata(
            extra={"kv_cache_dtype": effective}, video_layout=None, full_attn_spans=None, attn_mask=None
        )

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
    adapter.impl.rainfusion.parallel = adapter.module.MindIEParallelConfig(overlap_comm=True)
    run(adapter)
    run(adapter)
    assert len(adapter.states) == 1
    assert adapter.states[0].overlap_comm is True
    adapter.impl.rainfusion.parallel = adapter.module.MindIEParallelConfig(overlap_comm=False)
    with pytest.raises(RuntimeError, match="configuration"):
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
    namespace = {
        "dataclass": dataclasses.dataclass,
        "__name__": __name__,
        "MindIEParallelConfig": adapter_config_class(),
    }
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
        lambda a: setattr(a.strategy, "_ulysses_a2a_permute", True),
        lambda a: setattr(a.metadata, "attn_mask", a.query),
        lambda a: setattr(a.metadata, "full_attn_spans", ((0, 1),)),
        lambda a: setattr(a.impl.rainfusion, "parallel", None),
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


def adapter_config_class():
    path = ROOT / "vllm_omni/diffusion/distributed/mindiesd_config.py"
    spec = importlib.util.spec_from_file_location("h3_config_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MindIEParallelConfig


def packed_method():
    """Execute the production eager island without importing the model runner."""
    tree = ast.parse((MODEL_DIR / "minimax_h3_blocks.py").read_text(encoding="utf-8"))
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
    exec(compile(ast.fix_missing_locations(source), str(MODEL_DIR / "minimax_h3_blocks.py"), "exec"), namespace)
    return namespace["_run_packed_attention"]


def test_model_admitted_call_bypasses_original_outer_a2a(adapter):
    holder = SimpleNamespace(
        attention=adapter.attention, vsa_sparsity=None, vsa_tile_size=None, vsa_ref_policy=None, vsa_ref_keep_rate=None
    )
    adapter.attention.effective_attention = lambda: (
        adapter.impl,
        SimpleNamespace(
            get_name=lambda: "RAINFUSION_ATTN", supports_prefix_kv_slicing=True, supports_packed_mask_free=lambda: False
        ),
        None,
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
    original.__dict__["effective_attention"] = lambda: (
        adapter.impl,
        SimpleNamespace(
            get_name=lambda: "RAINFUSION_ATTN", supports_prefix_kv_slicing=True, supports_packed_mask_free=lambda: False
        ),
        None,
    )
    holder = SimpleNamespace(
        attention=original, vsa_sparsity=None, vsa_tile_size=None, vsa_ref_policy=None, vsa_ref_keep_rate=None
    )
    adapter.attention.effective_attention = lambda: (
        adapter.impl,
        SimpleNamespace(
            get_name=lambda: "RAINFUSION_ATTN", supports_prefix_kv_slicing=True, supports_packed_mask_free=lambda: False
        ),
        None,
    )
    q = adapter.query.squeeze(0)
    out = packed_method()(holder, q, q, q, cu_seqlens=torch.tensor([0, 14]), max_seqlen=14, packed_total=16)
    torch.testing.assert_close(out, q + 2)
    assert len(calls) == 1 and not adapter.calls


def test_resource_poison_prevents_replay(adapter):
    def fail(*args, **kwargs):
        raise RuntimeError("submitted collective failed")

    adapter.library.distributed_sparse_attention = fail
    with pytest.raises(RuntimeError, match="collective failed"):
        run(adapter)
    adapter.library.distributed_sparse_attention = lambda *args, **kwargs: pytest.fail("poisoned resource replayed")
    with pytest.raises(RuntimeError, match="closed or failed"):
        run(adapter)


def test_teardown_drains_before_releasing_state_and_is_idempotent(adapter, monkeypatch):
    from contextlib import nullcontext

    run(adapter)
    resource = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, adapter.query.device), lambda: None
    )
    state = resource.state
    drained = []
    monkeypatch.setattr(
        adapter.module.torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False
    )
    monkeypatch.setattr(
        adapter.module.current_omni_platform, "synchronize", lambda: drained.append(resource.state), raising=False
    )
    adapter.coordinator.resources.close()
    assert drained == [state] and resource.state is None
    adapter.coordinator.resources.close()
    assert drained == [state]
    with pytest.raises(RuntimeError, match="closed"):
        run(adapter)


def test_failed_drain_keeps_resource_and_prevents_release(adapter, monkeypatch):
    from contextlib import nullcontext

    run(adapter)
    resource = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, adapter.query.device), lambda: None
    )
    state = resource.state
    monkeypatch.setattr(
        adapter.module.torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False
    )

    def failed_drain():
        raise RuntimeError("drain failed")

    monkeypatch.setattr(adapter.module.current_omni_platform, "synchronize", failed_drain, raising=False)
    with pytest.raises(RuntimeError, match="drain failed"):
        adapter.coordinator.resources.close()
    assert resource.state is state
    monkeypatch.setattr(adapter.module.current_omni_platform, "synchronize", lambda: None)
    adapter.coordinator.resources.close()
    assert resource.state is None


@pytest.mark.parametrize(
    "changed",
    [
        {"overlap_comm": 1},
        {"head_major_qkv": True},
        {"sparse_type": "ada_bsa"},
        {"bsa_backend": "eagle", "sparse_type": "rf_v2"},
        {"bsa_backend": "unknown"},
    ],
)
def test_parallel_configuration_rejects_ambiguous_and_incompatible_options(adapter, changed):
    with pytest.raises(ValueError):
        adapter.module.MindIEParallelConfig(**changed)


def test_effective_schedule_backend_wins_over_baseline(adapter):
    adapter.attention.effective_attention = lambda: (object(), SimpleNamespace(get_name=lambda: "FLASH_ATTN"), None)
    assert run(adapter) is None
    assert not adapter.calls and not adapter.states


def test_sdk_receives_explicit_optimizations_backend_and_layout_plan(adapter):
    adapter.impl.rainfusion.parallel = adapter.module.MindIEParallelConfig(
        overlap_comm=True,
        optimize_attention_pipeline=True,
        bsa_backend="eagle",
        reorder_hoist=True,
        head_major_qkv=True,
    )
    adapter.metadata.video_layout = SimpleNamespace(pre_rearranged=True)
    run(adapter)
    state = adapter.states[0]
    assert state.overlap_comm and state.optimize_attention_pipeline
    assert state.bsa_backend == "eagle"
    assert state.qkv_layout_plan.requested and state.qkv_layout_plan.pre_rearranged
    assert adapter.calls[0][-1]["pre_rearranged"]


def test_serial_pipeline_explicitly_disables_overlap_and_buffer_optimization(adapter):
    adapter.impl.rainfusion.parallel = adapter.module.MindIEParallelConfig(
        overlap_comm=False, optimize_attention_pipeline=False, sparse_type="rf_v2"
    )
    run(adapter)
    assert not adapter.states[0].overlap_comm and not adapter.states[0].optimize_attention_pipeline
    assert adapter.calls[0][-1]["sparse_type"] == "rf_v2"


def test_resource_identity_changes_on_new_device(adapter):
    run(adapter)
    # The registry key also distinguishes devices, without needing NPU allocation.
    original = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, adapter.query.device), lambda: None
    )
    other = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, torch.device("cpu:1")), lambda: object()
    )
    assert original is not other


def test_rf_relabel_preserves_each_modality_and_padding(adapter, monkeypatch):
    abstract = ModuleType("vllm_omni.diffusion.attention.backends.abstract")
    setattr(abstract, "AttentionMetadata", make_metadata)
    monkeypatch.setitem(sys.modules, abstract.__name__, abstract)
    name = "vllm_omni.diffusion.models.minimax_h3.attention.rf_reorder_hoist"
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / "attention/rf_reorder_hoist.py")
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    branch = SimpleNamespace(
        seq_len=10,
        device=torch.device("cpu"),
        img_pos=torch.tensor([2, 3, 6, 7]),
        audio_pos=torch.tensor([4, 5]),
        text_pos_dev=torch.tensor([0, 1]),
        token_tags_dev=torch.arange(10),
        img_position_ids_dev=torch.arange(30).reshape(10, 3),
        static_kwargs={},
    )
    perm = torch.tensor([0, 1, 4, 5, 6, 7, 2, 3])
    inv = torch.argsort(perm)
    helper.relabel_branch(branch, perm, inv)
    padded = torch.cat((perm, torch.tensor([8, 9])))
    original_video_rows = padded.index_select(0, branch.img_pos_dev)
    original_audio_rows = padded.index_select(0, branch.audio_pos_dev)
    assert original_video_rows.tolist() == [2, 3, 6, 7]
    assert original_audio_rows.tolist() == [4, 5]
    assert branch.token_tags_dev.tolist() == [0, 1, 4, 5, 6, 7, 2, 3, 8, 9]
    assert torch.equal(branch.static_kwargs["img_position_ids"][0], torch.arange(30).reshape(10, 3)[padded])
    assert branch.static_kwargs["text_pos_info"]["position_ids"].tolist() == [0, 1]


def test_coordinator_drains_resources_before_destroying_process_groups(adapter, monkeypatch):
    path = ROOT / "vllm_omni/diffusion/distributed/group_coordinator.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SequenceParallelGroupCoordinator")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "destroy")
    events = []

    class Parent:
        def destroy(self):
            events.append("base")

    node = ast.ClassDef(
        name="Probe", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    namespace: dict[str, Any] = {
        "Parent": Parent,
        "torch": SimpleNamespace(distributed=SimpleNamespace(destroy_process_group=lambda group: events.append(group))),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    instance = namespace["Probe"]()
    instance.resources = SimpleNamespace(close=lambda: events.append("drain"))
    instance.device_group, instance.cpu_group = object(), object()
    instance.ulysses_group, instance.ring_group, instance.allgather_group = object(), object(), object()
    groups = [instance.ulysses_group, instance.ring_group, instance.allgather_group]
    instance.destroy()
    assert events == ["drain", *groups, "base"]
    assert instance.ulysses_group is instance.ring_group is instance.allgather_group is None


def test_typed_producer_consumer_share_one_request_plan(adapter):
    features = sys.modules["vllm_omni.diffusion.layers.minimax_h3_acceleration"]
    adapter.impl.rainfusion.parallel = adapter.module.MindIEParallelConfig(
        overlap_comm=True, reorder_hoist=True, head_major_qkv=True
    )
    adapter.metadata.video_layout = SimpleNamespace(pre_rearranged=True)
    with features.request_layout_scope():
        producer = features.qkv_layout_plan(
            head_count=4,
            head_dim=8,
            batch_size=1,
            world=2,
            precision="bf16",
            sparse_type="rf_v3",
            pre_rearranged=True,
            overlap_comm=True,
            optimize_attention_pipeline=True,
            requested=True,
        )
        run(adapter)
        assert adapter.states[0].qkv_layout_plan is producer


def test_head_major_request_switch_preserves_communicator_state(adapter):
    config = adapter.module.MindIEParallelConfig
    features = sys.modules["vllm_omni.diffusion.layers.minimax_h3_acceleration"]
    adapter.impl.rainfusion.parallel = config(overlap_comm=True, reorder_hoist=True, head_major_qkv=False)
    adapter.metadata.video_layout = SimpleNamespace(pre_rearranged=True)
    with features.request_layout_scope():
        run(adapter)
        first = adapter.states[0]
        assert not first.qkv_layout_plan.requested
    adapter.impl.rainfusion.parallel = config(overlap_comm=True, reorder_hoist=True, head_major_qkv=True)
    with features.request_layout_scope():
        run(adapter)
        assert first.qkv_layout_plan.requested
    assert adapter.states == [first]
    assert adapter.calls[0][-1]["state"] is adapter.calls[1][-1]["state"]
