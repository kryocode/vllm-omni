# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapter admission and padding checks without importing the full model runtime."""

import ast
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def env(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion/models/minimax_h3/attention/pr793_nonquant.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module.startswith("vllm_omni."))
    ]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_PR793_NONQUANT", "1")
    calls = []
    group = object()
    namespace["get_sp_group"] = lambda: types.SimpleNamespace(ulysses_group=group, world_size=2)
    namespace["get_ulysses_mode"] = lambda **kw: "strict"
    fake_dist = types.SimpleNamespace(get_world_size=lambda pg: 2, get_rank=lambda pg: 1)
    fake_dist.all_gather_object = lambda dst, value, **kw: dst.__setitem__(slice(None), [value, value])
    namespace["dist"] = fake_dist
    parallel = types.ModuleType("mindiesd.parallel")
    parallel.ParallelState = lambda **kw: object()

    def run(q, k, v, *args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(q.shape, dtype=torch.bfloat16)

    parallel.distributed_sparse_attention = run
    parallel.rank_window = lambda rank, world, local, used: (
        rank * local,
        max(rank * local, used),
        max(0, used - rank * local),
    )
    platform = types.ModuleType("mindiesd.utils.get_platform")
    platform.is_a5_device = lambda: False
    monkeypatch.setitem(sys.modules, "mindiesd.parallel", parallel)
    monkeypatch.setitem(sys.modules, "mindiesd.utils.get_platform", platform)
    impl = type("RainFusionAttentionImpl", (), {})()
    impl.rainfusion = types.SimpleNamespace(precision="bf16", sparsity=0.8)
    impl.softmax_scale, impl.causal = 0.5, False
    plan = types.SimpleNamespace(used_len=8, video_spans=[{"start": 0, "latent_shape": [1, 2, 4]}])
    impl._resolve_plan = lambda metadata: plan
    attention = types.SimpleNamespace(attention=impl, use_ring=False, _with_kv_cache_dtype=lambda m: m)
    metadata = types.SimpleNamespace(extra={})

    class Input:
        shape = (1, 5, 4, 4)
        ndim = 4
        dtype = torch.bfloat16
        device = types.SimpleNamespace(type="npu")

        def __getitem__(self, key):
            return torch.ones(self.shape, dtype=self.dtype)[key]

    return types.SimpleNamespace(
        cls=namespace["MindIESDH3PR793"],
        namespace=namespace,
        calls=calls,
        q=Input(),
        attention=attention,
        metadata=metadata,
        impl=impl,
        plan=plan,
        dist=fake_dist,
    )


def test_padding_and_owner_reuse(env):
    owner = env.cls()
    result = owner.try_attention(env.attention, env.q, env.q, env.q, env.metadata)
    assert torch.count_nonzero(result[:, :3]) == 0
    assert torch.all(result[:, 3:] == 1)
    state = owner.state
    owner.try_attention(env.attention, env.q, env.q, env.q, env.metadata)
    assert env.calls[0]["state"] is state is env.calls[1]["state"]


def test_fully_padded_rank(env):
    env.plan.used_len = 3
    result = env.cls().try_attention(env.attention, env.q, env.q, env.q, env.metadata)
    assert torch.all(result == 1)


def test_dense_retains_original_route(env):
    env.impl._resolve_plan = lambda metadata: None
    assert env.cls().try_attention(env.attention, env.q, env.q, env.q, env.metadata) is None
    assert not env.calls


@pytest.mark.parametrize("extra", [{"kv_cache_dtype": "int8"}, {"kv_cache_dtype": "fp8"}])
def test_quantized_attention_refused_before_data(env, extra):
    env.metadata.extra = extra
    with pytest.raises(ValueError, match="unsupported"):
        env.cls().try_attention(env.attention, env.q, env.q, env.q, env.metadata)
    assert not env.calls


def test_disagreement_refused_before_data(env):
    env.dist.all_gather_object = lambda dst, value, **kw: dst.__setitem__(slice(None), [value, None])
    with pytest.raises(ValueError, match="disagree"):
        env.cls().try_attention(env.attention, env.q, env.q, env.q, env.metadata)
    assert not env.calls


def test_old_optimization_environment_does_not_enable_route(env, monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_PR793_NONQUANT")
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_MINDIESD", "1")
    assert not env.cls().enabled
    assert env.cls().try_attention(None, None, None, None, None) is None


def test_bad_enable_value(env, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_PR793_NONQUANT", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        env.cls()


def test_overlap_default_and_opt_out(env, monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MINIMAX_H3_PR793_OVERLAP", raising=False)
    assert env.cls().overlap is True
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_PR793_OVERLAP", "0")
    owner = env.cls()
    assert owner.overlap is False
    owner.try_attention(env.attention, env.q, env.q, env.q, env.metadata)
    assert env.calls


def test_bad_overlap_value(env, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_PR793_OVERLAP", "yes")
    with pytest.raises(ValueError, match="OVERLAP must be 0 or 1"):
        env.cls()
