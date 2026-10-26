# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU control-flow contracts. Fake NPU tensors do not validate HCCL or kernels."""

import ast
import importlib
import json
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass(frozen=True)
class ControlDevice:
    type: str = "npu"
    index: int = 0

    def __str__(self):
        return "npu:0"


_CONTROL_DEVICE = ControlDevice()


class ControlTensor(torch.Tensor):
    @property
    def device(self):
        return _CONTROL_DEVICE


@dataclass
class Metadata:
    extra: dict = field(default_factory=dict)
    video_layout: object = None
    packed_padding: object = "padding"
    attn_mask: object = None
    joint_attn_mask: object = None
    joint_query: object = None
    joint_key: object = None
    joint_value: object = None
    full_attn_spans: object = None
    query_ranges: object = None


@pytest.fixture
def env(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    state = types.SimpleNamespace(world=2, rank=1, agreements=0, executions=0, legacy=0, fail=False, peer=None)
    state.control_streams = []
    state.dependencies = []
    state.receipts = []

    class Stream:
        def wait_stream(self, other):
            state.dependencies.append((self, other))

    state.current_stream = state.default_stream = Stream()

    @contextmanager
    def stream_context(stream):
        old = state.current_stream
        state.current_stream = stream
        try:
            yield
        finally:
            state.current_stream = old

    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(
            current_stream=lambda device=None: state.current_stream,
            stream=stream_context,
        ),
        raising=False,
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)

    def package(name, path=None):
        value = types.ModuleType(name)
        value.__path__ = [] if path is None else [str(path)]
        monkeypatch.setitem(sys.modules, name, value)
        return value

    for name in list(sys.modules):
        if name.startswith("vllm_omni.") or name.startswith("mindiesd."):
            monkeypatch.delitem(sys.modules, name)
    package("vllm_omni", root / "vllm_omni")
    package("vllm_omni.diffusion", root / "vllm_omni/diffusion")
    package("vllm_omni.diffusion.attention", root / "vllm_omni/diffusion/attention")
    package("vllm_omni.diffusion.offloader", root / "vllm_omni/diffusion/offloader")
    package("vllm_omni.diffusion.models", root / "vllm_omni/diffusion/models")
    package("vllm_omni.diffusion.models.minimax_h3", root / "vllm_omni/diffusion/models/minimax_h3")
    package("vllm_omni.diffusion.models.minimax_h3.attention", root / "vllm_omni/diffusion/models/minimax_h3/attention")
    package("vllm")
    logger = package("vllm.logger")
    logger.init_logger = lambda _: types.SimpleNamespace(
        # vLLM caches info_once arguments; a dict must fail here as in production.
        info_once=lru_cache(maxsize=128)(lambda *a, **k: None),
        info=lambda message, payload: state.receipts.append(json.loads(payload)),
    )
    package("vllm_omni.diffusion.distributed")
    ps = package("vllm_omni.diffusion.distributed.parallel_state")
    state.group = object()
    ps.get_sp_group = lambda: types.SimpleNamespace(
        ulysses_group=state.group, world_size=state.world, ring_world_size=1
    )
    fc = package("vllm_omni.diffusion.forward_context")
    fc.get_ulysses_mode = lambda **_: "strict"
    package("mindiesd")
    parallel = package("mindiesd.parallel")

    class State:
        def __init__(self, **kwargs):
            self.stream = kwargs.get("stream")
            self.geom_cache = {}
            self.calls = 0
            self.lookahead = None
            self.graph_cache = None

        def stats(self):
            return {
                "calls": self.calls,
                "engaged": self.calls,
                "single_stream_calls": 0,
                "overlapped_calls": self.calls,
            }

        def make_stream(self, device):
            if self.stream is None:
                self.stream = Stream()

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(
                implementation="head_pipeline", head_chunk_size=1, pipeline=True, prepack=True, prepare_all=True
            )
            self.__dict__.update(kwargs)

    class Executor:
        def __init__(self, *, config, state):
            self.state = state
            self.config = config

        def execution_signature(self, *, graph_requested):
            import os

            return (
                tuple(sorted(vars(self.config).items())),
                self.state.lookahead,
                graph_requested,
                os.environ.get("MINDIESD_ATTENTION_GRAPH_PREFIX", "0"),
            )

        def __call__(self, q, k, v, *, backend, **kwargs):
            state.executions += 1
            if state.fail:
                raise RuntimeError("collective already submitted")
            self.state.calls += 1
            # Backend contract is exercised independently below. This stub only
            # validates bridge ownership and local padding restoration.
            return torch.zeros_like(q) + 7, {"single_stream": self.state.stream is None}

    def legacy(q, k, v, **kwargs):
        state.legacy += 1
        return torch.zeros_like(q) + 3, {}

    parallel.TransportState = parallel.ParallelState = State
    parallel.UlyssesConfig = Config
    parallel.AttentionExecutor = Executor
    parallel.distributed_attention = legacy
    parallel.input_gate_supported = lambda: True
    parallel.rank_window = lambda rank, world, local, used: (
        rank * local,
        max(rank * local, min((rank + 1) * local, used)),
        min(local, max(0, used - rank * local)),
    )
    package("mindiesd.layers")
    package("mindiesd.layers.flash_attn")
    geometry = package("mindiesd.layers.flash_attn.rf_geometry")

    class Cache:
        def __init__(self):
            self.storage = {}

    geometry.RFGeometryCache = Cache
    geometry.rf_pool_split_enabled = lambda: False
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: state.world)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: state.rank)

    def gather(output, value, group=None):
        state.agreements += 1
        state.control_streams.append(state.current_stream)
        output[: value.numel()].copy_(value)
        peer = value.clone() if state.peer is None else state.peer(value.clone())
        output[value.numel() :].copy_(peer)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    # Control tensors have CPU storage. Production agreement allocation is NPU;
    # patch only that small boundary allocation for these control-flow tests.
    real_tensor, real_empty = torch.tensor, torch.empty
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **k: real_tensor(*a, **{x: y for x, y in k.items() if x != "device"})
    )
    monkeypatch.setattr(torch, "empty", lambda *a, **k: real_empty(*a, **{x: y for x, y in k.items() if x != "device"}))
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD_BRIDGE", "1")
    monkeypatch.setenv("MINDIESD_ATTENTION_D", "1")
    state.bridge = importlib.import_module("vllm_omni.diffusion.attention.mindiesd_bridge")
    state.h3 = importlib.import_module("vllm_omni.diffusion.models.minimax_h3.attention.mindiesd_h3")
    state.backend = importlib.import_module("vllm_omni.diffusion.attention.backends.mindiesd_native")
    return state


def tensor():
    return torch.ones(1, 5, 8, 4, dtype=torch.bfloat16).as_subclass(ControlTensor)


def attention(*, sparse=False):
    native = types.SimpleNamespace(causal=False)
    native.rainfusion = types.SimpleNamespace(
        precision="bf16", sparsity=0.8, start_step=1, end_step=0, skip_layers=frozenset()
    )
    native._resolve_plan = lambda metadata: (
        types.SimpleNamespace(video_spans=({"start": 0, "latent_shape": (1, 1, 7)},), used_len=7) if sparse else None
    )
    value = types.SimpleNamespace(
        skip_sequence_parallel=False,
        attention=native,
        attn_backend=types.SimpleNamespace(get_name=lambda: "RAINFUSION_ATTN" if sparse else "FLASH_ATTN"),
        _get_active_parallel_strategy=lambda: types.SimpleNamespace(enabled=True, name="ulysses"),
        _active_paged_kv_adapter=lambda: None,
        _with_kv_cache_dtype=lambda metadata: metadata,
        _kv_cache_dtype=None,
        _disable_kv_quant=False,
    )

    def run(q, k, v, metadata):
        value.last_metadata = metadata
        out = metadata.extra.get("mindiesd_rf_output")
        if out is not None:
            out.copy_(q)
            return out
        return q.clone()

    value._run_local_attention = run
    return value


def metadata():
    return Metadata(
        extra={
            "h3_num_requests": 1,
            "valid_kv_length": 7,
            "cu_seqlens_q": torch.tensor([0, 7, 10]),
            "cu_seqlens_k": torch.tensor([0, 7, 10]),
            "max_seqlen_q": 7,
            "max_seqlen_k": 7,
            "npu_attn_varlen": True,
        }
    )


def prepare(env, owner, layer, value):
    return owner.prepare_request(
        plan_id=1,
        shape=tuple(value.shape),
        used_len=7,
        device=value.device,
        attention_layers=(layer,),
        video_layout=None,
    )


def observation_model(owner, layer):
    model = types.SimpleNamespace(
        _mindiesd_parallel=owner,
        _rope_local_span=lambda seq: (5, 5),
        modules=lambda: (types.SimpleNamespace(attention=layer, num_heads=8, head_dim=4),),
    )
    branch = types.SimpleNamespace(seq_len=10, used_len=7, device=tensor().device, static_kwargs={})
    return model, branch


def test_request_receipt_selected_without_calls_is_not_executed(env):
    owner, layer = env.h3.MindIESDH3Parallel(), attention()
    model, branch = observation_model(owner, layer)
    with env.h3.prepare_h3_request(model, branch):
        pass
    receipt = owner.last_execution_receipt
    assert receipt["cumulative"]["attention_owner"]["route"]["selected"] == "prepared_ulysses"
    assert not receipt["executed"] and receipt["execution_status"] == "no_calls"
    assert receipt["request_delta"]["attention"]["successful_host_returns"] == 0
    assert receipt["device_completion"] == "not_observed" and len(env.receipts) == 1


def test_request_receipts_separate_delta_from_cumulative_returns(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    model, branch = observation_model(owner, layer)
    for calls in (1, 2):
        with env.h3.prepare_h3_request(model, branch):
            for _ in range(calls):
                owner.try_attention(layer, value, value, value, metadata())
        receipt = owner.last_execution_receipt
        assert receipt["executed"] and receipt["execution_status"] == "host_returns_observed"
        assert receipt["request_delta"]["attention"]["prepared_host_returns"] == calls
    assert receipt["cumulative"]["attention_owner"]["attention"]["prepared_host_returns"] == 3
    assert receipt["request_sequence"] == 2 and len(env.receipts) == 2
    assert env.agreements == 2  # telemetry adds no collective


def test_failed_request_receipt_preserves_partial_work(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    model, branch = observation_model(owner, layer)
    failure = ValueError("body failed")
    with pytest.raises(ValueError) as caught:
        with env.h3.prepare_h3_request(model, branch):
            owner.try_attention(layer, value, value, value, metadata())
            raise failure
    receipt = owner.last_execution_receipt
    assert caught.value is failure
    assert not receipt["executed"] and receipt["execution_status"] == "partial_failed"
    assert receipt["error_type"] == "ValueError"
    assert receipt["request_delta"]["attention"]["prepared_host_returns"] == 1
    assert receipt["cumulative"]["attention_owner"]["failed"] and len(env.receipts) == 1


def test_preparation_failure_still_emits_one_receipt(env, monkeypatch):
    owner, layer = env.h3.MindIESDH3Parallel(), attention()
    model, branch = observation_model(owner, layer)

    def reject(**kwargs):
        raise ValueError("invalid prepare")

    monkeypatch.setattr(owner, "prepare_request", reject)
    with pytest.raises(ValueError, match="invalid prepare"):
        with env.h3.prepare_h3_request(model, branch):
            pytest.fail("body must not execute")
    assert len(env.receipts) == 1 and not owner.last_execution_receipt["executed"]
    assert owner.last_execution_receipt["execution_status"] == "failed_before_return"


def test_disabled_optimization_receipt_has_no_transport_execution(env):
    owner, layer = env.h3.MindIESDH3Parallel(), attention()
    owner.enabled = False
    model, branch = observation_model(owner, layer)
    with env.h3.prepare_h3_request(model, branch):
        pass
    assert owner.last_execution_receipt["cumulative"]["attention_owner"]["route"]["selected"] == "disabled"
    assert not owner.last_execution_receipt["executed"] and env.agreements == 0


def test_graph_receipt_counts_replays_separately_from_capture(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    model, branch = observation_model(owner, layer)
    # First prepare creates transport state; the following request observes
    # existing host counters without invoking a Graph/device operation.
    with env.h3.prepare_h3_request(model, branch):
        pass
    graph_counts = {"graph_captures": 1, "graph_replays": 4, "graph_status": "replay", "graph_failed": False}
    owner.state.graph_cache = types.SimpleNamespace(snapshot=lambda: dict(graph_counts))
    with env.h3.prepare_h3_request(model, branch):
        owner.try_attention(layer, value, value, value, metadata())
        graph_counts["graph_replays"] += 1
    receipt = owner.last_execution_receipt
    assert receipt["request_delta"]["graph"] == {"graph_captures": 0, "graph_replays": 1}
    assert receipt["cumulative"]["attention_owner"]["graph"]["graph_replays"] == 5


@pytest.mark.parametrize("change", ["lookahead", "prefix", "config"])
def test_actual_execution_policy_drift_is_rejected_before_submission(env, monkeypatch, change):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    assert prepare(env, owner, layer, value)
    if change == "lookahead":
        owner._bridge.state.lookahead = 1
    elif change == "prefix":
        monkeypatch.setenv("MINDIESD_ATTENTION_GRAPH_PREFIX", "1")
    else:
        owner._bridge.executor.config.pipeline = False
    with pytest.raises(RuntimeError, match="execution policy changed"):
        owner.try_attention(layer, value, value, value, metadata())
    assert owner._bridge.failed and env.executions == 0


def test_weight_layout_owner_validates_active_and_historical_proofs(env):
    owner = env.h3.MindIESDH3Parallel()
    contract = ("qbmm_nz_v1", "1", (("weight", 0, (32, 32), (32, 1)),))
    group = object()
    owner.validate_weight_layout_setup()
    with pytest.raises(RuntimeError):
        owner.assert_weight_layout_live(contract, group)
    owner._bridge.group = group
    owner._bridge.prepared = True
    owner._prepared_weight_contracts = frozenset((contract,))
    owner.assert_weight_layout_live(contract, group)
    owner._qbmm_nz_agreed_group = group
    owner._qbmm_nz_agreed_contracts = frozenset((contract,))
    owner.finish_request()
    with pytest.raises(RuntimeError, match="active shared request|group changed"):
        owner.assert_weight_layout_live(contract, group)
    owner.assert_weight_layout_live(contract, group, historical=True)
    with pytest.raises(RuntimeError, match="group changed"):
        owner.assert_weight_layout_live(contract, object(), historical=True)
    with pytest.raises(RuntimeError, match="historical layout agreement"):
        owner.assert_weight_layout_live(("foreign",), group, historical=True)
    owner._bridge.failed = True
    with pytest.raises(RuntimeError, match="disabled or failed"):
        owner.assert_weight_layout_live(contract, group, historical=True)


@pytest.fixture
def grouped_environment():
    # Group activation intentionally publishes multiple startup variables.
    # Restore all of them, including keys not assigned through monkeypatch.
    with patch.dict(os.environ):
        yield


@pytest.mark.usefixtures("grouped_environment")
def test_nz_request_agreement_does_not_enable_attention_pipeline(env, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD", "0")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD_BRIDGE", "0")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_COMM_MODE", "legacy")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_WEIGHT_LAYOUT", "nz")
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    assert not owner.enabled and not owner.bridge_enabled
    contract = ("qbmm_nz_v1", "1", (("weight", 0, (32, 32), (32, 1)),))
    model, branch = observation_model(owner, layer)
    model.named_modules = lambda: (("block", types.SimpleNamespace(_mindiesd_qbmm_nz_contract=contract)),)
    owner.validate_weight_layout_setup()
    with env.h3.prepare_h3_request(model, branch):
        assert owner.request_prepared and not owner._bridge.accepted
        owner.assert_weight_layout_live(contract, env.group)
        assert owner.try_attention(layer, value, value, value, metadata()) is None
    assert env.agreements == 1 and env.executions == 0
    assert not owner.request_prepared


@pytest.mark.usefixtures("grouped_environment")
def test_explicit_pipeline_group_rejects_unsupported_route(env, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_COMM_MODE", "pipeline")
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    layer._get_active_parallel_strategy = lambda: types.SimpleNamespace(enabled=False, name="none")
    with pytest.raises(RuntimeError, match="COMM_MODE=pipeline"):
        prepare(env, owner, layer, value)
    assert owner._bridge.failed and env.executions == 0


def test_post_embed_request_declares_block_layout_without_mutating_branch(env, monkeypatch):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Layout:
        video_spans: tuple = (("video",),)
        pre_rearranged: bool = False

    monkeypatch.setenv("H3_SAFE_RF_POST_EMBED", "1")
    owner, layer = env.h3.MindIESDH3Parallel(), attention()
    model = types.SimpleNamespace(
        _mindiesd_parallel=owner,
        _rope_local_span=lambda seq: (5, 5),
        modules=lambda: (types.SimpleNamespace(attention=layer, num_heads=8, head_dim=4),),
    )
    layout = Layout()
    branch = types.SimpleNamespace(
        seq_len=10, used_len=7, device=tensor().device, static_kwargs={"video_token_layout": layout}
    )
    seen = []
    original = owner.prepare_request

    def prepare_and_capture(**kwargs):
        seen.append(kwargs["video_layout"])
        return original(**kwargs)

    monkeypatch.setattr(owner, "prepare_request", prepare_and_capture)
    with env.h3.prepare_h3_request(model, branch):
        assert branch.static_kwargs["video_token_layout"] is layout
    assert not layout.pre_rearranged and seen[0].pre_rearranged


@pytest.mark.parametrize("unsupported", [None, "dense", "start", "end", "skip", "geometry"])
def test_prefix_graph_coverage_is_agreed_before_data_submission(env, monkeypatch, unsupported):
    monkeypatch.setenv("MINDIESD_ATTENTION_GRAPH", "1")
    monkeypatch.setenv("MINDIESD_ATTENTION_GRAPH_PREFIX", "1")
    owner = env.h3.MindIESDH3Parallel()
    layer, value = attention(sparse=unsupported != "dense"), tensor()
    layer.attention.rainfusion.start_step = 1 if unsupported == "start" else 0
    layer.attention.rainfusion.end_step = 1 if unsupported == "end" else 0
    layer.attention.rainfusion.skip_layers = frozenset((0,)) if unsupported == "skip" else frozenset()
    layer.attention.layer_idx = 0
    layer.attention.qkv_layout = "BSND"
    layer.attention._resolve_multi_span_plan = lambda layout, used: None if unsupported == "geometry" else object()
    layout = types.SimpleNamespace(video_spans=("video",), used_len=7)

    def invoke():
        return owner.prepare_request(
            plan_id=1,
            shape=tuple(value.shape),
            used_len=7,
            device=value.device,
            attention_layers=(layer,),
            video_layout=layout,
        )

    if unsupported is None:
        assert invoke()
        assert owner.route_summary["selected"] == "prepared_ulysses"
    else:
        with pytest.raises(RuntimeError, match="prefix Graph is unsupported"):
            invoke()
        assert owner._bridge.failed and owner.route_summary["reason"] != "eligible"
    assert env.agreements == 1 and env.executions == 0


def test_request_agreement_once_and_h3_padding_state_reuse(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    assert prepare(env, owner, layer, value)
    for _ in range(2):
        result = owner.try_attention(layer, value, value, value, metadata())
        assert torch.equal(result[:, :2], torch.zeros_like(value[:, :2]) + 7)
        assert torch.equal(result[:, 2:], value[:, 2:])
    assert env.agreements == 1
    assert env.executions == 2 and env.legacy == 0
    assert owner.calls == owner.dense_calls == 2 and owner.sparse_calls == 0
    assert owner.state.calls == 2 and owner.state.stream is not None
    owner.finish_request()


def test_missing_request_boundary_retains_legacy_route(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    result = owner.try_attention(layer, value, value, value, metadata())
    assert result is not None
    assert env.agreements == 0 and env.executions == 0 and env.legacy == 1


def test_disabled_owner_still_falls_back(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    owner.enabled = False
    assert owner.try_attention(layer, value, value, value, metadata()) is None
    assert env.executions == env.legacy == 0


def test_rank_contract_mismatch_refused_at_boundary(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    env.peer = lambda payload: payload + 1
    with pytest.raises(RuntimeError, match="rank|agreement|contract"):
        prepare(env, owner, layer, value)
    assert env.executions == env.legacy == 0


def test_prepared_contract_change_is_fatal_before_collectives(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    prepare(env, owner, layer, value)
    changed = metadata()
    changed.extra["unknown_head_tensor"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="contract|prepared"):
        owner.try_attention(layer, value, value, value, changed)
    assert env.executions == env.legacy == 0


def test_collective_failure_never_retries_native(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    prepare(env, owner, layer, value)
    env.fail = True
    with pytest.raises(RuntimeError, match="collective already submitted"):
        owner.try_attention(layer, value, value, value, metadata())
    with pytest.raises(RuntimeError, match="failed|Failed|terminate"):
        owner.try_attention(layer, value, value, value, metadata())
    assert env.executions == 1 and env.legacy == 0
    owner.finish_request()
    with pytest.raises(RuntimeError, match="failed|Failed|terminate"):
        owner.try_attention(layer, value, value, value, metadata())
    assert env.executions == 1 and env.legacy == 0


@pytest.mark.parametrize("field", ["enabled", "bridge_enabled", "overlap", "verify_first", "head_chunk_size"])
def test_prepared_owner_switch_changes_are_fatal(env, field):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    prepare(env, owner, layer, value)
    setattr(owner, field, 2 if field == "head_chunk_size" else not getattr(owner, field))
    with pytest.raises(RuntimeError, match="contract|prepared"):
        owner.try_attention(layer, value, value, value, metadata())
    assert env.executions == env.legacy == 0


@pytest.mark.parametrize("change", ["skip", "custom", "strategy", "backend"])
@pytest.mark.parametrize("new_enabled", [False, True])
def test_prepared_layer_route_changes_are_fatal(env, change, new_enabled):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    owner.bridge_enabled = new_enabled
    prepare(env, owner, layer, value)
    if change == "skip":
        layer.skip_sequence_parallel = True
    elif change == "custom":
        layer._has_custom_attention = True
    elif change == "strategy":
        layer._get_active_parallel_strategy = lambda: types.SimpleNamespace(enabled=True, name="ring")
    else:
        layer.attn_backend.get_name = lambda: "UNKNOWN_ATTN"
    with pytest.raises(RuntimeError, match="contract|prepared"):
        owner.try_attention(layer, value, value, value, metadata())
    assert env.executions == env.legacy == 0
    assert owner._bridge.failed


@pytest.mark.parametrize("new_enabled", [False, True])
def test_excluded_native_refiner_retains_native_route(env, new_enabled):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    owner.bridge_enabled = new_enabled
    prepare(env, owner, layer, value)
    refiner = attention()
    refiner.skip_sequence_parallel = True
    assert owner.try_attention(refiner, value, value, value, metadata()) is None
    assert env.executions == env.legacy == 0
    assert not owner._bridge.failed


@pytest.mark.parametrize("new_enabled", [False, True])
def test_unknown_parallel_layer_is_fatal_in_prepared_request(env, new_enabled):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    owner.bridge_enabled = new_enabled
    prepare(env, owner, layer, value)
    unknown_layer = attention()
    with pytest.raises(RuntimeError, match="contract|prepared"):
        owner.try_attention(unknown_layer, value, value, value, metadata())
    assert env.executions == env.legacy == 0
    assert owner._bridge.failed


def test_all_heads_chunk_is_reconfigured_at_request_boundary(env):
    bridge = env.bridge.AttentionBridge(head_chunk_size=None)
    contract = env.bridge.RouteContract((1, 5, 8, 4), str(torch.bfloat16), 7)
    bridge.prepare_request(1, contract, group=env.group, device=tensor().device)
    first_state = bridge.state
    assert bridge.executor.config.head_chunk_size == 4
    bridge.finish_request()
    contract = env.bridge.RouteContract((1, 5, 12, 4), str(torch.bfloat16), 7)
    bridge.prepare_request(2, contract, group=env.group, device=tensor().device)
    assert bridge.executor.config.head_chunk_size == 6
    assert bridge.state is first_state


@pytest.mark.parametrize("legacy_first", [False, True])
def test_legacy_and_new_routes_share_one_transport_stream(env, legacy_first):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    original_stream = None
    if legacy_first:
        owner.try_attention(layer, value, value, value, metadata())
        original_stream = owner.state.stream
    prepare(env, owner, layer, value)
    new_state = owner.state
    if original_stream is not None:
        assert new_state.stream is original_stream
    owner.finish_request()
    owner.try_attention(layer, value, value, value, metadata())
    assert owner.state.stream is new_state.stream
    prepare(env, owner, layer, value)
    assert owner.state is new_state


@pytest.mark.parametrize("new_enabled", [False, True])
def test_request_metadata_and_attention_share_persistent_issuer(env, new_enabled):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    owner.bridge_enabled = new_enabled
    prepare(env, owner, layer, value)
    owner.try_attention(layer, value, value, value, metadata())
    issuer = owner.state.stream
    assert env.control_streams == [issuer]
    assert (issuer, env.default_stream) in env.dependencies
    assert (env.default_stream, issuer) in env.dependencies
    owner.finish_request()
    prepare(env, owner, layer, value)
    assert env.control_streams == [issuer, issuer]


@pytest.mark.parametrize("activate_before_call", [False, True])
def test_boundary_uses_declared_strategy_before_sp_region(env, activate_before_call):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    layer.parallel_strategy = types.SimpleNamespace(enabled=True, name="ulysses")
    layer._get_active_parallel_strategy = lambda: types.SimpleNamespace(enabled=False, name="none")
    assert prepare(env, owner, layer, value)
    if activate_before_call:
        layer._get_active_parallel_strategy = lambda: layer.parallel_strategy
        assert owner.try_attention(layer, value, value, value, metadata()) is not None
    else:
        with pytest.raises(RuntimeError, match="contract|prepared|active"):
            owner.try_attention(layer, value, value, value, metadata())
        assert env.executions == env.legacy == 0


@pytest.mark.parametrize("accepted", [False, True])
def test_transport_group_cannot_change_between_requests(env, accepted):
    bridge = env.bridge.AttentionBridge()
    contract = env.bridge.RouteContract((1, 5, 8, 4), str(torch.bfloat16), 7)
    bridge.prepare_request(1, contract, group=env.group, device=tensor().device, candidate=accepted)
    bridge.finish_request()
    with pytest.raises(RuntimeError, match="group|context|transport"):
        bridge.prepare_request(2, contract, group=object(), device=tensor().device, candidate=accepted)
    assert env.agreements == 1
    assert bridge.failed


@pytest.mark.parametrize("field", ["enabled", "bridge_enabled", "overlap", "verify_first", "head_chunk_size"])
def test_rejected_route_still_locks_owner_switches(env, field):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    owner.bridge_enabled = False
    assert not prepare(env, owner, layer, value)
    setattr(owner, field, 2 if field == "head_chunk_size" else not getattr(owner, field))
    with pytest.raises(RuntimeError, match="contract|prepared"):
        owner.try_attention(layer, value, value, value, metadata())
    assert env.executions == env.legacy == 0
    assert owner._bridge.failed


def test_rejected_legacy_collective_exception_poisons_owner(env):
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    owner.bridge_enabled = False
    prepare(env, owner, layer, value)

    def fail(*args, **kwargs):
        raise RuntimeError("legacy collective already submitted")

    owner._legacy.try_attention = fail
    with pytest.raises(RuntimeError, match="legacy collective already submitted"):
        owner.try_attention(layer, value, value, value, metadata())
    owner.finish_request()
    with pytest.raises(RuntimeError, match="failed|Failed|terminate"):
        prepare(env, owner, layer, value)
    assert env.agreements == 1


def test_native_backend_owns_shared_rf_cache_and_preserves_metadata(env):
    factory = env.backend.NativeBackendFactory()
    layer, original = attention(sparse=True), metadata()
    backend = factory.build(layer, original)
    q = torch.ones(1, 7, 1, 4, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    assert backend.forward(q, q, q, chunk=None, out=out) is out
    assert torch.equal(out, q)
    assert layer.last_metadata.extra["mindiesd_rf_geometry_cache"] is factory.geometry_cache
    assert original.extra.keys().isdisjoint({"mindiesd_rf_geometry_cache", "mindiesd_rf_output"})
    assert "mindiesd_rf_output" not in backend.metadata.extra


def test_request_scope_clears_plan_after_exception(env):
    owner, layer = env.h3.MindIESDH3Parallel(), attention()
    model = types.SimpleNamespace(
        _mindiesd_parallel=owner,
        _rope_local_span=lambda seq: (5, 5),
        modules=lambda: (types.SimpleNamespace(attention=layer, num_heads=8, head_dim=4),),
    )
    branch = types.SimpleNamespace(seq_len=10, used_len=7, device=tensor().device, static_kwargs={})
    with pytest.raises(ValueError, match="body failed"):
        with env.h3.prepare_h3_request(model, branch):
            raise ValueError("body failed")
    assert not owner.request_prepared
    assert env.agreements == 1


@pytest.mark.parametrize("loop_fails", [False, True])
def test_real_diffuse_wrapper_prepares_before_residency_and_always_finishes(env, loop_fails):
    """Execute the actual wrapper without importing model/tokenizer dependencies."""
    path = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "diffuse")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    owner, layer, events = env.h3.MindIESDH3Parallel(), attention(), []
    model = types.SimpleNamespace(
        _mindiesd_parallel=owner,
        _rope_local_span=lambda seq: (5, 5),
        modules=lambda: (types.SimpleNamespace(attention=layer, num_heads=8, head_dim=4),),
    )
    branch = types.SimpleNamespace(seq_len=10, used_len=7, device=tensor().device, static_kwargs={})

    @contextmanager
    def prepare(model, branch):
        try:
            with env.h3.prepare_h3_request(model, branch):
                events.append("prepared")
                yield
        finally:
            events.append("finished")

    @contextmanager
    def resident(**kwargs):
        events.append("resident")
        yield

    @contextmanager
    def progress(**kwargs):
        yield types.SimpleNamespace(update=lambda: None)

    def loop(**kwargs):
        events.append("loop")
        assert owner.request_prepared
        if loop_fails:
            raise RuntimeError("loop failed")
        return 11, 22

    namespace = {
        "prepare_h3_request": prepare,
        "minimax_h3_denoise_loop": loop,
        "MINIMAX_H3_IMGVID_COND_TIMESTEP": 0.2,
        "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP": 0.2,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    pipe = types.SimpleNamespace(
        _build_denoise_inputs=lambda **kwargs: {
            "branch": branch,
            "sigmas_video": [1, 0],
            "sigmas_audio": [1, 0],
            "video_rows": 1,
            "audio_rows": 2,
            "cond_anchor": None,
            "audio_anchor": None,
            "video_edit": None,
            "audio_edit": None,
        },
        _transformer_for_task=lambda task: model,
        transformer=model,
        _resident_dit_layers_on_device=resident,
        progress_bar=progress,
        _unpack_denoised_rows=lambda branch, video, audio, **kwargs: (video, audio),
        device=tensor().device,
    )
    kwargs = {argument.arg: None for argument in function.args.kwonlyargs}
    if loop_fails:
        with pytest.raises(RuntimeError, match="loop failed"):
            namespace["diffuse"](pipe, **kwargs)
    else:
        assert namespace["diffuse"](pipe, **kwargs) == (11, 22)
    assert events == ["prepared", "resident", "loop", "finished"]
    assert env.agreements == 1
    assert not owner.request_prepared
