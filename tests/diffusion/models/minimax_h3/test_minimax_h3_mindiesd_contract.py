# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the optional communication-only H3 adapter."""

import ast
import functools
import os
import sys
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
ADAPTER = ROOT / "vllm_omni/diffusion/models/minimax_h3/attention/mindiesd_h3.py"
TRANSFORMER = ROOT / "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"
RAINFUSION = ROOT / "vllm_omni/diffusion/attention/backends/rainfusion_attn.py"


@dataclass
class Metadata:
    extra: dict = field(default_factory=dict)
    attn_mask: object = None
    joint_attn_mask: object = None
    joint_query: object = None
    joint_key: object = None
    joint_value: object = None
    full_attn_spans: object = None
    query_ranges: object = None
    packed_padding: object = None
    video_layout: object = None


class NPUTensor(torch.Tensor):
    @property
    def device(self):
        return SimpleNamespace(type="npu")


def load_adapter(monkeypatch, *, overlap="1", verify="0"):
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD", "1")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD_OVERLAP", overlap)
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD_VERIFY_FIRST", verify)
    tree = ast.parse(ADAPTER.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    scope = dict(
        torch=torch,
        os=os,
        replace=replace,
        functools=functools,
        import_module=import_module,
        logger=SimpleNamespace(info_once=lambda *a: None),
        _REQUIRES_MINDIESD="missing",
    )
    scope["logs"] = []
    scope["logger"] = SimpleNamespace(info_once=lambda *args: scope["logs"].append(args))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ADAPTER), "exec"), scope)
    group = object()
    scope["get_ulysses_mode"] = lambda **kw: "strict"
    scope["get_sp_group"] = lambda: SimpleNamespace(ulysses_group=group, world_size=2, ring_world_size=1)
    scope["dist"] = SimpleNamespace(get_world_size=lambda _: 2, get_rank=lambda _: 1)
    calls, streams, local = [], [], []

    class State:
        def __init__(self, **kwargs):
            self.group = kwargs["group"]

        def make_stream(self, device):
            streams.append(device)

    def distributed(q, k, v, **kwargs):
        calls.append(kwargs)
        # Simulate full valid sequence delivered to the exact framework local consumer.
        full = torch.cat([q, q], dim=1)[:, : kwargs["used_len"]]
        out = kwargs["attention_fn"](full, full, full)
        # Stand-in transport restores a local shard; padding must be restored by adapter.
        result = q + 7
        result[:, : max(0, kwargs["used_len"] - q.shape[1])] = out[:, q.shape[1] :]
        return result, {"a2a_calls": 8}

    scope["_mindiesd_module"] = lambda _: SimpleNamespace(ParallelState=State, distributed_attention=distributed)
    q = torch.arange(1 * 4 * 56 * 8).reshape(1, 4, 56, 8).bfloat16().as_subclass(NPUTensor)
    metadata = Metadata(
        extra={
            "h3_num_requests": 1,
            "valid_kv_length": 6,
            "cu_seqlens_q": torch.tensor([0, 6, 8], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 6, 8], dtype=torch.int32),
            "max_seqlen_q": 6,
            "max_seqlen_k": 6,
            "npu_attn_varlen": True,
            "laser_input_scale": 2,
        }
    )

    class Attention:
        skip_sequence_parallel = False
        attn_backend = SimpleNamespace(get_name=lambda: "FLASH_ATTN")
        attention = SimpleNamespace(causal=False)

        def _get_active_parallel_strategy(self):
            return SimpleNamespace(enabled=True, name="ulysses")

        def _active_paged_kv_adapter(self):
            return None

        def _with_kv_cache_dtype(self, md):
            return replace(md, extra={**md.extra, "kv_cache_dtype": "float"})

        def _run_local_attention(self, q, k, v, md):
            local.append((q, md))
            assert q.shape[1] == 6
            assert md.extra["max_seqlen_q"] == 6
            assert md.extra["kv_cache_dtype"] == "float"
            assert md.extra["laser_input_scale"] == 2
            assert md.attn_mask is None and md.packed_padding is None
            return q + 1

        def __call__(self, q, k, v, md):
            return q + 1

    return scope, Attention(), q, metadata, calls, streams, local


def test_callback_reuses_original_kernel_metadata_and_persistent_state(monkeypatch):
    s, attn, q, md, calls, streams, local = load_adapter(monkeypatch)
    adapter = s["MindIESDH3Parallel"]()
    out = adapter.try_attention(attn, q, q, q, md)
    adapter.try_attention(attn, q, q, q, md)
    assert len(local) == len(calls) == adapter.calls == 2
    assert calls[0]["state"] is calls[1]["state"] is adapter.state
    assert calls[0]["head_chunk_size"] == 1
    assert len(streams) == 1
    assert torch.equal(out[:, :2], (q + 1)[:, :2])
    assert torch.equal(out[:, 2:], q[:, 2:])
    assert md.extra["cu_seqlens_q"].shape == (3,)


def test_sequential_and_full_head_options(monkeypatch):
    s, attn, q, md, calls, streams, _ = load_adapter(monkeypatch, overlap="0")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD_HEAD_CHUNK_SIZE", "0")
    adapter = s["MindIESDH3Parallel"]()
    adapter.try_attention(attn, q, q, q, md)
    assert calls[0]["head_chunk_size"] is None
    assert not streams


@pytest.mark.parametrize(
    "unsupported",
    [
        "ring",
        "advanced",
        "batch",
        "multiple_requests",
        "gate",
        "mask",
        "joint",
        "paged",
        "kv_quant",
        "unknown",
        "heads",
        "causal",
        "sparse_fp8",
        "sparse_mxfp4",
        "sparse_mix",
        "other_backend",
    ],
)
def test_unsupported_routes_never_import_parallel(monkeypatch, unsupported):
    s, attn, q, md, calls, streams, _ = load_adapter(monkeypatch)
    if unsupported == "ring":
        attn._get_active_parallel_strategy = lambda: SimpleNamespace(enabled=True, name="ring")
    elif unsupported == "advanced":
        s["get_ulysses_mode"] = lambda **kw: "advanced_uaa"
    elif unsupported == "batch":
        q = q.repeat(2, 1, 1, 1)
    elif unsupported == "multiple_requests":
        md.extra["h3_num_requests"] = 2
    elif unsupported == "gate":
        md.extra["gate_compress"] = q
    elif unsupported == "mask":
        md.attn_mask = torch.ones(1, 56, 8, 8, dtype=torch.bool)
    elif unsupported == "joint":
        md.joint_query = q
    elif unsupported == "paged":
        attn._active_paged_kv_adapter = lambda: object()
    elif unsupported == "kv_quant":
        attn._with_kv_cache_dtype = lambda md: replace(md, extra={**md.extra, "kv_cache_dtype": "fp8"})
    elif unsupported == "unknown":
        md.extra["head_bias"] = q
    elif unsupported == "heads":
        q = q[:, :, :55]
    elif unsupported == "causal":
        attn.attention.causal = True
    elif unsupported.startswith("sparse_"):
        attn.attn_backend = SimpleNamespace(get_name=lambda: "RAINFUSION_ATTN")
        attn.attention._resolve_plan = lambda md: object()
        attn.attention.rainfusion = SimpleNamespace(precision=unsupported.removeprefix("sparse_"))
        attn._with_kv_cache_dtype = lambda md: md
    elif unsupported == "other_backend":
        attn.attn_backend = SimpleNamespace(get_name=lambda: "FASTVIDEO_VSA")
    s["_mindiesd_module"] = lambda _: pytest.fail("unsupported path imported MindIE")
    assert s["MindIESDH3Parallel"]().try_attention(attn, q, q, q, md) is None
    assert not calls and not streams


def test_master_defaults_off(monkeypatch):
    s, attn, q, md, *_ = load_adapter(monkeypatch)
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_MINDIESD")
    s["_mindiesd_module"] = lambda _: pytest.fail("disabled imported MindIE")
    assert s["MindIESDH3Parallel"]().try_attention(attn, q, q, q, md) is None


def test_rainfusion_dense_uses_same_local_backend(monkeypatch):
    s, attn, q, md, calls, _, local = load_adapter(monkeypatch)
    attn.attn_backend = SimpleNamespace(get_name=lambda: "RAINFUSION_ATTN")
    attn.attention._resolve_plan = lambda md: None
    assert s["MindIESDH3Parallel"]().try_attention(attn, q, q, q, md) is not None
    assert calls and local


def test_verify_first_is_bit_exact_on_valid_rows(monkeypatch):
    s, attn, q, md, *_ = load_adapter(monkeypatch, verify="1")
    adapter = s["MindIESDH3Parallel"]()
    adapter.try_attention(attn, q, q, q, md)
    attn.__class__.__call__ = lambda self, q, k, v, md: q + 2
    adapter._verified_attention.clear()
    with pytest.raises(AssertionError, match="bit-exact"):
        adapter.try_attention(attn, q, q, q, md)


def native_rainfusion(monkeypatch, attention, *, multi_video=False):
    """Use the frozen native consumer; mock only its hardware kernel boundary."""
    tree = ast.parse(RAINFUSION.read_text())
    plan_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RainFusionPlan")
    impl_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RainFusionAttentionImpl")
    methods = [
        n for n in impl_node.body if isinstance(n, ast.FunctionDef) and n.name in ("forward_npu", "_forward_sparse_npu")
    ]
    scope = dict(
        torch=torch,
        dataclass=dataclass,
        RainFusionPlan=object,
        AttentionMetadata=Metadata,
        _INPUT_LAYOUT="BSND",
        _INNER_PRECISE=0,
        _BLOCK_SIZE=128,
        _supports_video_spans=lambda fn: True,
        logger=SimpleNamespace(info_once=lambda *args: None),
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    native_class = ast.ClassDef(name="NativeRainFusion", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[future, plan_node, native_class], type_ignores=[]))
    exec(compile(module, str(RAINFUSION), "exec"), scope)
    native = scope["NativeRainFusion"]()
    native.causal = False
    native.softmax_scale = 0.125
    native.rainfusion = SimpleNamespace(precision="bf16", sparsity=0.8)
    spans = [{"start": 1, "latent_shape": [1, 1, 2]}, {"start": 4, "latent_shape": [1, 1, 2]}]
    plan = scope["RainFusionPlan"](
        used_len=6,
        prefix_len=None if multi_video else 2,
        latent_shape=None if multi_video else [1, 2, 2],
        video_spans=spans if multi_video else None,
    )
    native._resolve_plan = lambda md: plan if md.video_layout is not None else None
    native.dense_fallback = SimpleNamespace(forward_npu=lambda q, k, v, md: q + 1)
    kernel_calls = []

    def sparse_attention(q, k, v, **kwargs):
        kernel_calls.append((q, kwargs))
        return q + 1

    module = ModuleType("mindiesd")
    module.sparse_attention = sparse_attention
    monkeypatch.setitem(sys.modules, "mindiesd", module)
    attention.attn_backend = SimpleNamespace(get_name=lambda: "RAINFUSION_ATTN")
    attention.attention = native
    attention._run_local_attention = native.forward_npu
    return plan, kernel_calls


@pytest.mark.parametrize("multi_video", [False, True])
def test_sparse_plan_uses_native_consumer_and_preserves_video_geometry(monkeypatch, multi_video):
    s, attn, q, md, calls, streams, _ = load_adapter(monkeypatch)
    plan, kernels = native_rainfusion(monkeypatch, attn, multi_video=multi_video)
    layout = SimpleNamespace(used_len=6, video_spans=plan.video_spans)
    md.video_layout = layout
    adapter = s["MindIESDH3Parallel"]()
    out = adapter.try_attention(attn, q, q, q, md)
    assert out is not None
    assert len(calls) == len(kernels) == adapter.calls == adapter.sparse_calls == 1
    assert adapter.dense_calls == 0
    assert len(streams) == 1
    received_q, kwargs = kernels[0]
    assert received_q.shape[1] == plan.used_len
    assert kwargs["head_num"] == received_q.shape[2]
    assert kwargs["precision"] == "bf16" and kwargs["sparse_type"] == "rf_v2"
    assert kwargs["sparsity"] == 0.8 and kwargs["scale"] == 0.125
    if multi_video:
        assert kwargs["video_spans"] is plan.video_spans
    else:
        assert kwargs["txt_len"] == 2 and kwargs["latent_shape_q"] == [1, 2, 2]
    assert md.video_layout is layout and md.extra["cu_seqlens_q"].shape == (3,)
    assert torch.equal(out[:, :2], (q + 1)[:, :2])
    assert torch.equal(out[:, 2:], q[:, 2:])
    assert any("sparse" in str(log) for log in s["logs"])


def test_sparse_callback_preserves_layout_identity_and_head_chunk(monkeypatch):
    s, attn, q, md, calls, _, _ = load_adapter(monkeypatch)
    plan, kernels = native_rainfusion(monkeypatch, attn, multi_video=True)
    md.video_layout = SimpleNamespace(used_len=6, video_spans=plan.video_spans)
    seen = []
    native_local = attn._run_local_attention

    def local(q, k, v, local_md):
        seen.append(local_md)
        return native_local(q, k, v, local_md)

    attn._run_local_attention = local
    adapter = s["MindIESDH3Parallel"]()
    assert adapter.try_attention(attn, q, q, q, md) is not None
    full_chunk = torch.cat([q, q], dim=1)[:, :6, :1]
    calls[0]["attention_fn"](full_chunk, full_chunk, full_chunk)
    assert seen[-1].video_layout is md.video_layout
    assert seen[-1].packed_padding is None
    assert seen[-1].extra["cu_seqlens_q"].tolist() == [0, 6]
    assert kernels[-1][1]["head_num"] == 1


def test_dense_sparse_transition_reuses_state_and_counts_success(monkeypatch):
    s, attn, q, md, calls, streams, _ = load_adapter(monkeypatch)
    plan, kernels = native_rainfusion(monkeypatch, attn)
    adapter = s["MindIESDH3Parallel"]()
    for layout in (None, SimpleNamespace(used_len=plan.used_len), None):
        md.video_layout = layout
        assert adapter.try_attention(attn, q, q, q, md) is not None
    assert adapter.calls == 3 and adapter.dense_calls == 2 and adapter.sparse_calls == 1
    assert len(kernels) == 1 and len(streams) == 1
    assert all(call["state"] is adapter.state for call in calls)


def test_verify_first_distinguishes_layer_and_dense_sparse_state(monkeypatch):
    s, attn, q, md, *_ = load_adapter(monkeypatch, verify="1")
    plan, _ = native_rainfusion(monkeypatch, attn)
    verified = []
    attn.__class__.__call__ = lambda self, q, k, v, metadata: (verified.append((self, metadata.video_layout)), q + 1)[1]
    adapter = s["MindIESDH3Parallel"]()
    adapter.try_attention(attn, q, q, q, md)
    md.video_layout = SimpleNamespace(used_len=plan.used_len)
    adapter.try_attention(attn, q, q, q, md)
    adapter.try_attention(attn, q, q, q, md)
    other = attn.__class__()
    native_rainfusion(monkeypatch, other)
    adapter.try_attention(other, q, q, q, md)
    assert len(verified) == 3
    assert verified[0][1] is None and verified[1][1] is md.video_layout
    assert verified[2][0] is other


def test_no_verify_does_not_construct_geometry_key(monkeypatch):
    s, attn, q, md, *_ = load_adapter(monkeypatch)
    plan, _ = native_rainfusion(monkeypatch, attn)

    class Layout:
        def __repr__(self):
            pytest.fail("verification geometry was serialized with verification disabled")

        def __hash__(self):
            pytest.fail("verification geometry was hashed with verification disabled")

    md.video_layout = Layout()
    adapter = s["MindIESDH3Parallel"]()
    assert adapter.try_attention(attn, q, q, q, md) is not None
    assert plan.used_len == 6
    assert not adapter._verified_attention


def test_sparse_consumer_failure_propagates_and_is_not_counted(monkeypatch):
    s, attn, q, md, *_ = load_adapter(monkeypatch)
    plan, _ = native_rainfusion(monkeypatch, attn)
    md.video_layout = SimpleNamespace(used_len=plan.used_len)

    def broken(*args):
        raise RuntimeError("native sparse failure")

    attn._run_local_attention = broken
    adapter = s["MindIESDH3Parallel"]()
    with pytest.raises(RuntimeError, match="native sparse failure"):
        adapter.try_attention(attn, q, q, q, md)
    assert adapter.calls == adapter.sparse_calls == adapter.dense_calls == 0


@pytest.mark.parametrize("constraint", ["mask", "head_extra", "joint", "paged", "ring", "kv_quant"])
def test_sparse_plan_keeps_unsupported_constraint_fallback(monkeypatch, constraint):
    s, attn, q, md, calls, streams, _ = load_adapter(monkeypatch)
    plan, _ = native_rainfusion(monkeypatch, attn)
    md.video_layout = SimpleNamespace(used_len=plan.used_len)
    if constraint == "mask":
        md.attn_mask = torch.ones(1, 56, 8, 8, dtype=torch.bool)
    elif constraint == "head_extra":
        md.extra["head_bias"] = q
    elif constraint == "joint":
        md.joint_key = q
    elif constraint == "paged":
        attn._active_paged_kv_adapter = lambda: object()
    elif constraint == "ring":
        s["get_sp_group"] = lambda: SimpleNamespace(ulysses_group=object(), world_size=2, ring_world_size=2)
    elif constraint == "kv_quant":
        attn._with_kv_cache_dtype = lambda md: replace(md, extra={**md.extra, "kv_cache_dtype": "fp8"})
    s["_mindiesd_module"] = lambda _: pytest.fail("unsupported sparse route imported MindIE")
    assert s["MindIESDH3Parallel"]().try_attention(attn, q, q, q, md) is None
    assert not calls and not streams


def test_transformer_retains_original_fusions_and_ffn():
    source = TRANSFORMER.read_text()
    assert "from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope" in source
    assert "hidden, _ = self.fc1(x)" in source
    assert "out, _ = self.fc2(hidden)" in source
    for forbidden in ("try_mlp", "try_mx_projection", "gate_norm_mx_for_mlp", "_mindiesd_h3_mlp_fc1"):
        assert forbidden not in source


# D preserves the native BF16 consumer and supplies only geometry/output plumbing.
def _d_setup(monkeypatch):
    s, attn, q, md, calls, streams, local = load_adapter(monkeypatch)
    plan, kernels = native_rainfusion(monkeypatch, attn, multi_video=True)
    md.video_layout = SimpleNamespace(used_len=6, video_spans=plan.video_spans)
    State = s["_mindiesd_module"]("parallel").ParallelState
    original_init = State.__init__
    def init(self, **kwargs):
        original_init(self, **kwargs)
        self.geom_cache = {}
    monkeypatch.setattr(State, "__init__", init)
    return s, attn, q, md, calls, kernels


def test_d_adapter_supplies_native_output_callback_and_shared_geometry(monkeypatch):
    s, attn, q, md, calls, kernels = _d_setup(monkeypatch)
    monkeypatch.setenv("MINDIESD_ATTENTION_D", "1")
    adapter = s["MindIESDH3Parallel"]()
    assert adapter.try_attention(attn, q, q, q, md) is not None
    callback = calls[-1].get("attention_out_fn")
    assert callable(callback), "D must supply a native output-writing callback"
    seen = []
    def sparse(a, b, c, **kwargs):
        seen.append(kwargs)
        target = kwargs.get("out")
        if target is not None:
            target.copy_(a + 1)
            return target
        return a + 1
    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", sparse)
    full = torch.cat([q, q], dim=1)[:, :6, :4]
    out = torch.empty_like(full)
    assert callback(full, full, full, out=out) is out
    assert torch.equal(out, full + 1)
    assert seen[-1]["geometry_cache"] is adapter.state.geom_cache
    assert seen[-1]["precision"] == "bf16" and seen[-1]["sparse_type"] == "rf_v2"
    assert seen[-1]["out"] is out
    assert "mindiesd_rf_geometry_cache" not in md.extra
    assert "mindiesd_rf_output" not in md.extra


def test_d_off_keeps_original_native_kwargs(monkeypatch):
    s, attn, q, md, calls, kernels = _d_setup(monkeypatch)
    monkeypatch.setenv("MINDIESD_ATTENTION_D", "0")
    adapter = s["MindIESDH3Parallel"]()
    adapter.try_attention(attn, q, q, q, md)
    assert "attention_out_fn" not in calls[-1]
    assert "geometry_cache" not in kernels[-1][1] and "out" not in kernels[-1][1]


def test_d_backend_forwards_cache_and_output_only_for_full_bf16_multispan(monkeypatch):
    s, attn, q, md, calls, kernels = _d_setup(monkeypatch)
    cache = {}
    full = torch.cat([q, q], dim=1)[:, :6]
    out = torch.empty_like(full)
    seen = []
    def sparse(a, b, c, **kwargs):
        seen.append(kwargs)
        if kwargs.get("out") is not None:
            kwargs["out"].copy_(a + 1)
            return kwargs["out"]
        return a + 1
    monkeypatch.setattr(sys.modules["mindiesd"], "sparse_attention", sparse)
    local_md = replace(md, extra={**md.extra, "mindiesd_rf_geometry_cache": cache, "mindiesd_rf_output": out})
    actual = attn.attention._forward_sparse_npu(full, full, full, attn.attention._resolve_plan(md), local_md)
    assert actual is out
    assert seen[-1]["geometry_cache"] is cache
    assert seen[-1]["out"] is out


@pytest.mark.parametrize("value", ["2", "yes", ""])
def test_d_adapter_rejects_invalid_flag(monkeypatch, value):
    s, attn, q, md, calls, kernels = _d_setup(monkeypatch)
    monkeypatch.setenv("MINDIESD_ATTENTION_D", value)
    with pytest.raises(ValueError, match="MINDIESD_ATTENTION_D"):
        s["MindIESDH3Parallel"]().try_attention(attn, q, q, q, md)
    assert not calls
