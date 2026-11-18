# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contracts for the narrow MiniMax-H3 MindIE-SD pipeline adapter."""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
from vllm_omni.diffusion.attention.parallel.ulysses_allgather import UlyssesAllGatherKVParallelAttention
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelResources

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "vllm_omni/diffusion/models/minimax_h3"
HELPER_NAME = "vllm_omni.diffusion.models.minimax_h3.attention.mindiesd_parallel"


@dataclass
class HoistAttention:
    attention: object


@dataclass
class HoistBlock:
    attn: HoistAttention


@dataclass
class HoistModel:
    blocks: list[HoistBlock]


@dataclass
class HoistBranch:
    seq_len: int
    used_len: int
    device: torch.device
    img_pos: torch.Tensor
    audio_pos: torch.Tensor
    text_pos_dev: torch.Tensor
    token_tags_dev: torch.Tensor
    img_position_ids_dev: torch.Tensor
    static_kwargs: dict[str, object]
    img_pos_dev: torch.Tensor = field(init=False)
    audio_pos_dev: torch.Tensor = field(init=False)


@dataclass
class HoistPlan:
    permutation: torch.Tensor
    inverse: torch.Tensor


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    calls, states = [], []

    def new_state(**kwargs):
        state = SimpleNamespace(**kwargs)
        states.append(state)
        return state

    def distributed_attention(q, k, v, spans, **kwargs):
        calls.append((q, k, v, spans, kwargs))
        return q + 1

    platform = SimpleNamespace(is_npu=lambda: True)
    for name, exports in (
        ("vllm_omni.diffusion.forward_context", {"get_ulysses_mode": lambda **_: "strict"}),
        ("vllm_omni.platforms", {"current_omni_platform": platform}),
        ("mindiesd.parallel", {"ParallelState": new_state, "distributed_sparse_attention": distributed_attention}),
    ):
        stub = ModuleType(name)
        stub.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, stub)
    module = load_module(monkeypatch, HELPER_NAME, MODEL_DIR / "attention/mindiesd_parallel.py")
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)

    group = object()
    coordinator = SimpleNamespace(
        ring_world_size=1,
        allgather_world_size=1,
        resources=SequenceParallelResources(),
    )
    strategy = SimpleNamespace(
        name="ulysses",
        _ulysses_pg=group,
        _sp_group=coordinator,
        _scatter_idx=2,
        _gather_idx=1,
        _ulysses_a2a_permute=False,
    )
    plan = SimpleNamespace(used_len=14, video_spans=None, prefix_len=6, latent_shape=(2, 2, 2))
    impl = SimpleNamespace(
        rainfusion=SimpleNamespace(
            precision="bf16",
            sparse_type="rf_v3",
            sparsity=0.8,
            parallel=module.MindIEParallelConfig(),
        ),
        softmax_scale=0.125,
        _resolve_plan=lambda metadata: plan,
    )

    attention = SimpleNamespace(
        causal=False,
        qkv_layout="BSND",
        _has_custom_attention=False,
        is_paged_kv_active=lambda: False,
        effective_attention=lambda: (impl, SimpleNamespace(get_name=lambda: "RAINFUSION_ATTN"), None),
        paged_kv_cache_role=None,
        _get_active_parallel_strategy=lambda: strategy,
        _with_kv_cache_dtype=lambda metadata: metadata,
    )
    query = torch.zeros((1, 8, 4, 8), dtype=torch.bfloat16)
    return SimpleNamespace(
        module=module,
        attention=attention,
        metadata=AttentionMetadata(),
        query=query,
        calls=calls,
        states=states,
        strategy=strategy,
        coordinator=coordinator,
        group=group,
        plan=plan,
        impl=impl,
        library=sys.modules["mindiesd.parallel"],
        platform=platform,
    )


def run(adapter, *, query=None):
    q = adapter.query if query is None else query
    return adapter.module.try_distributed_sparse_attention(adapter.attention, q, q, q, adapter.metadata)


def test_pipeline_call_is_explicit_and_state_is_reused(adapter):
    assert torch.equal(run(adapter), adapter.query + 1)
    run(adapter)
    assert len(adapter.states) == 1
    resource = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, None, adapter.query.device, "rf_v3", "bf16", "eagle"), lambda: None
    )
    assert resource.completed_calls == 2
    state = adapter.states[0]
    assert state.overlap_comm is True
    assert state.optimize_attention_pipeline is True
    assert state.bsa_backend == "eagle"
    q, k, v, spans, kwargs = adapter.calls[0]
    assert q is k is v is adapter.query
    assert spans == [{"start": 6, "latent_shape": [2, 2, 2]}]
    assert kwargs == dict(
        group=adapter.group,
        cp_group=None,
        used_len=14,
        state=state,
        precision="bf16",
        sparse_type="rf_v3",
        pre_rearranged=False,
        scale=0.125,
        sparsity=0.8,
    )


def test_pre_rearranged_is_forwarded(adapter):
    adapter.metadata.video_layout = SimpleNamespace(pre_rearranged=True)
    run(adapter)
    assert adapter.calls[0][-1]["pre_rearranged"] is True


@pytest.mark.parametrize(
    "sparse_type,precision,backend,dtype",
    [
        ("rf_v2", "bf16", "cann", torch.bfloat16),
        ("rf_v2", "fp16", "cann", torch.float16),
        ("rf_v3", "bf16", "eagle", torch.bfloat16),
        ("rf_v3", "fp16", "eagle", torch.float16),
        ("rf_v3", "mix", "eagle", torch.bfloat16),
        ("rf_v3", "mxfp8", "cann", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("pre_rearranged", [False, True])
def test_all_four_sparse_paths_reach_the_public_sdk(adapter, sparse_type, precision, backend, dtype, pre_rearranged):
    adapter.impl.rainfusion.sparse_type = sparse_type
    adapter.impl.rainfusion.precision = precision
    adapter.impl.rainfusion.bsa_backend = backend
    adapter.metadata.video_layout = VideoTokenLayout(pre_rearranged=pre_rearranged)
    q = torch.zeros((1, 8, 4, 128), dtype=dtype)
    assert torch.equal(run(adapter, query=q), q + 1)
    assert adapter.calls[0][-1]["precision"] == precision
    assert adapter.calls[0][-1]["sparse_type"] == sparse_type
    assert adapter.calls[0][-1]["pre_rearranged"] is pre_rearranged
    assert adapter.states[0].bsa_backend == backend


@pytest.mark.parametrize(
    "sparse_type,precision,backend",
    [
        ("rf_v2", "bf16", "cann"),
        ("rf_v2", "fp16", "cann"),
        ("rf_v3", "bf16", "eagle"),
        ("rf_v3", "fp16", "eagle"),
        ("rf_v3", "mix", "eagle"),
        ("rf_v3", "mxfp8", "cann"),
    ],
)
@pytest.mark.parametrize("ulysses_size", [4, 8])
def test_cp_uses_the_existing_orthogonal_group_and_exact_sdk_keywords(
    adapter, sparse_type, precision, backend, ulysses_size
):
    adapter.impl.rainfusion.sparse_type = sparse_type
    adapter.impl.rainfusion.precision = precision
    adapter.impl.rainfusion.bsa_backend = backend
    adapter.metadata.video_layout = VideoTokenLayout(pre_rearranged=True)
    adapter.coordinator.allgather_world_size = 2
    adapter.coordinator.allgather_group = object()
    adapter.coordinator.ulysses_world_size = ulysses_size
    adapter.coordinator.ulysses_group = adapter.group
    combined = object.__new__(UlyssesAllGatherKVParallelAttention)
    combined._sp_group = adapter.coordinator
    combined._ulysses = adapter.strategy
    adapter.strategy = combined
    adapter.attention._get_active_parallel_strategy = lambda: adapter.strategy
    adapter.module.dist.get_world_size = (
        lambda group=None: ulysses_size * 2
        if group is None
        else (2 if group is adapter.coordinator.allgather_group else ulysses_size)
    )
    adapter.plan.used_len = 251
    adapter.plan.video_spans = [{"start": 123, "latent_shape": [2, 8, 8]}]
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    q = torch.zeros((1, 32, 56, 128), dtype=dtype)
    assert torch.equal(run(adapter, query=q), q + 1)
    kwargs = adapter.calls[0][-1]
    assert kwargs["group"] is adapter.group
    assert kwargs["cp_group"] is adapter.coordinator.allgather_group
    assert adapter.states[0].cp_group is adapter.coordinator.allgather_group
    assert not {"kv_gather_group", "comm_dtype", "comm_quant_scope"}.intersection(kwargs)


def test_resource_caches_do_not_mix_precision_or_backend(adapter):
    adapter.impl.rainfusion.bsa_backend = "eagle"
    run(adapter)
    adapter.impl.rainfusion.sparse_type = "rf_v2"
    adapter.impl.rainfusion.bsa_backend = "cann"
    run(adapter)
    assert len(adapter.states) == 2


def test_explicit_fp16_attention_restores_the_bf16_producer_dtype(adapter):
    adapter.impl.rainfusion.precision = "fp16"
    adapter.impl.rainfusion.bsa_backend = "eagle"
    output = run(adapter)
    assert output.dtype == torch.bfloat16
    assert all(value.dtype == torch.float16 for value in adapter.calls[0][:3])


def test_pipeline_optimization_switches_are_passed_without_overriding(adapter):
    adapter.impl.rainfusion.parallel = adapter.module.MindIEParallelConfig(
        overlap_comm=False,
        reorder_hoist=False,
        optimize_attention_pipeline=False,
        chunk_ready_pipeline=False,
    )
    run(adapter)
    state = adapter.states[0]
    assert state.overlap_comm is state.optimize_attention_pipeline is state.chunk_ready_pipeline is False


@pytest.mark.parametrize(
    "sparse_type,precision,backend",
    [
        ("rf_v2", "bf16", "cann"),
        ("rf_v2", "fp16", "cann"),
        ("rf_v3", "bf16", "eagle"),
        ("rf_v3", "fp16", "eagle"),
        ("rf_v3", "mix", "eagle"),
        ("rf_v3", "mxfp8", "cann"),
    ],
)
def test_native_comparison_selects_the_same_sparse_provider(monkeypatch, sparse_type, precision, backend):
    from vllm_omni.diffusion.attention.backends import rainfusion_attn

    calls = []

    def sparse_attention(q, k, v, *, precision, video_spans, **kwargs):
        calls.append((q, k, v, precision, video_spans, kwargs))
        return q

    library = ModuleType("mindiesd")
    setattr(library, "sparse_attention", sparse_attention)
    monkeypatch.setitem(sys.modules, "mindiesd", library)
    monkeypatch.setattr(rainfusion_attn, "_mindiesd_supports_precision", lambda: True)
    impl = rainfusion_attn.RainFusionAttentionImpl(
        num_heads=4,
        head_size=128,
        softmax_scale=128**-0.5,
        qkv_layout="BSND",
        backend_kwargs={"sparsity": 0.8, "sparse_type": sparse_type, "precision": precision, "bsa_backend": backend},
    )
    q = torch.arange(32 * 4 * 128).reshape(1, 32, 4, 128).to(torch.bfloat16)
    spans = [{"start": 0, "latent_shape": [2, 2, 3]}]
    output = impl._forward_sparse_npu(q, q, q, rainfusion_attn.RainFusionPlan(used_len=12, video_spans=spans))
    assert torch.equal(output, q) and output.dtype == q.dtype
    operands = calls[0][:3]
    assert all(t.shape[1] == 12 for t in operands)
    assert all(t.dtype == (torch.float16 if precision == "fp16" else q.dtype) for t in operands)
    assert calls[0][3] == ("bf16" if precision == "fp16" else precision)
    assert calls[0][-1]["sparse_type"] == sparse_type
    if sparse_type == "rf_v3":
        assert calls[0][-1]["bsa_backend"] == backend


@pytest.mark.parametrize(
    "sparse_type,precision,backend",
    [
        ("rf_v2", "bf16", "cann"),
        ("rf_v2", "fp16", "cann"),
        ("rf_v3", "bf16", "eagle"),
        ("rf_v3", "fp16", "eagle"),
        ("rf_v3", "mix", "eagle"),
        ("rf_v3", "mxfp8", "cann"),
    ],
)
def test_all_four_paths_can_hoist_the_same_public_rf_plan(adapter, monkeypatch, sparse_type, precision, backend):
    adapter.impl.rainfusion.sparse_type = sparse_type
    adapter.impl.rainfusion.precision = precision
    adapter.impl.rainfusion.bsa_backend = backend
    helper = load_module(
        monkeypatch,
        "vllm_omni.diffusion.models.minimax_h3.attention.rf_reorder_hoist",
        MODEL_DIR / "attention/rf_reorder_hoist.py",
    )
    plan = torch.tensor([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 6, 7, 8, 9])
    root = ModuleType("mindiesd")
    setattr(root, "build_rf_attention_plan", lambda **kwargs: HoistPlan(plan, torch.argsort(plan)))
    monkeypatch.setitem(sys.modules, "mindiesd", root)
    branch = HoistBranch(
        seq_len=16,
        used_len=14,
        device=torch.device("cpu"),
        img_pos=torch.arange(6, 14),
        audio_pos=torch.tensor([4, 5]),
        text_pos_dev=torch.arange(4),
        token_tags_dev=torch.arange(16),
        img_position_ids_dev=torch.arange(48).reshape(16, 3),
        static_kwargs={
            "video_token_layout": VideoTokenLayout(prefix_len=6, latent_grid=(2, 2, 2)),
            "packed_seq_params": {"vsa_prefix_segments": ()},
        },
    )
    model = HoistModel([HoistBlock(HoistAttention(adapter.attention))])
    assert helper.maybe_hoist_branch(branch, model)
    layout = branch.static_kwargs["video_token_layout"]
    packed = branch.static_kwargs["packed_seq_params"]
    assert isinstance(layout, VideoTokenLayout) and layout.pre_rearranged
    assert isinstance(packed, dict) and "vsa_prefix_segments" not in packed
    assert torch.cat((plan, torch.tensor([14, 15]))).index_select(0, branch.img_pos_dev).tolist() == list(range(6, 14))


def test_live_resource_rejects_overlap_mode_change(adapter):
    run(adapter)
    adapter.impl.rainfusion.parallel = adapter.module.MindIEParallelConfig(overlap_comm=False)
    with pytest.raises(RuntimeError, match="configuration"):
        run(adapter)
    assert len(adapter.calls) == 1


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
        lambda a: setattr(a.metadata, "joint_key", a.query),
        lambda a: a.metadata.extra.update(minimax_h3_reference_kv_post_parallel=(object(), 1, False)),
        lambda a: a.metadata.extra.update(gate_compress=a.query),
        lambda a: a.metadata.extra.update(vdn_window=object()),
    ],
)
def test_unsupported_calls_decline_before_state_or_collective(adapter, change):
    change(adapter)
    assert run(adapter) is None
    assert not adapter.calls and not adapter.states


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((2, 8, 4, 8), torch.bfloat16),
        ((1, 8, 3, 8), torch.bfloat16),
        ((1, 8, 4, 8), torch.float16),
    ],
)
def test_unsupported_tensor_contract_declines(adapter, shape, dtype):
    assert run(adapter, query=torch.zeros(shape, dtype=dtype)) is None
    assert not adapter.calls


@pytest.mark.parametrize("precision,sparse_type", [("mix", "rf_v2"), ("fp8", "rf_v3"), ("mxfp4", "rf_v3")])
def test_enabled_pipeline_rejects_non_target_attention_mode(adapter, precision, sparse_type):
    adapter.impl.rainfusion.precision = precision
    adapter.impl.rainfusion.sparse_type = sparse_type
    with pytest.raises(ValueError, match="Unsupported MindIE parallel"):
        run(adapter)
    assert not adapter.calls and not adapter.states


def test_collective_failure_poisons_resource_without_replay(adapter):
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
        ("mindiesd", adapter.group, None, adapter.query.device, "rf_v3", "bf16", "eagle"), lambda: None
    )
    state, drained = resource.state, []
    monkeypatch.setattr(
        adapter.module.torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False
    )
    monkeypatch.setattr(adapter.platform, "synchronize", lambda: drained.append(resource.state), raising=False)
    adapter.coordinator.resources.close()
    assert drained == [state] and resource.state is None
    adapter.coordinator.resources.close()
    assert drained == [state]


@pytest.mark.parametrize(
    "target,name,value",
    [
        ("strategy", "_ulysses_pg", None),
        ("strategy", "_scatter_idx", 1),
        ("strategy", "_gather_idx", 2),
        ("rainfusion", "sparse_type", "ada_bsa"),
    ],
)
def test_reorder_preflight_rejects_incompatible_executor(adapter, monkeypatch, target, name, value):
    setattr(adapter.strategy if target == "strategy" else adapter.impl.rainfusion, name, value)
    helper = load_module(
        monkeypatch,
        "vllm_omni.diffusion.models.minimax_h3.attention.rf_reorder_hoist",
        MODEL_DIR / "attention/rf_reorder_hoist.py",
    )
    layout = SimpleNamespace(pre_rearranged=False)
    branch = SimpleNamespace(static_kwargs={"video_token_layout": layout})
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(attention=adapter.attention))])
    with pytest.raises(ValueError, match="MindIE|strict Ulysses"):
        helper.maybe_hoist_branch(branch, model)
    assert not layout.pre_rearranged and not adapter.calls and not adapter.states


def test_rf_relabel_preserves_modalities_and_padding(adapter, monkeypatch):
    name = "vllm_omni.diffusion.models.minimax_h3.attention.rf_reorder_hoist"
    helper = load_module(monkeypatch, name, MODEL_DIR / "attention/rf_reorder_hoist.py")
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
    permutation = torch.tensor([0, 1, 4, 5, 6, 7, 2, 3])
    helper.relabel_branch(branch, permutation, torch.argsort(permutation))
    padded = torch.cat((permutation, torch.tensor([8, 9])))
    assert padded.index_select(0, branch.img_pos_dev).tolist() == [2, 3, 6, 7]
    assert padded.index_select(0, branch.audio_pos_dev).tolist() == [4, 5]
    assert branch.token_tags_dev.tolist() == padded.tolist()
    assert torch.equal(branch.static_kwargs["img_position_ids"][0], torch.arange(30).reshape(10, 3)[padded])
