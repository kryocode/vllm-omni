# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tensors through the actual RF consumer, without platform imports."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion/attention/backends"


@pytest.fixture
def consumer(monkeypatch):
    package = ModuleType("_group_rf_tensor")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    name = package.__name__ + ".mindiesd_native"
    spec = importlib.util.spec_from_file_location(name, ROOT / "mindiesd_native.py")
    native = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, native)
    spec.loader.exec_module(native)
    tree = ast.parse((ROOT / "rainfusion_attn.py").read_text(encoding="utf-8"))
    impl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RainFusionAttentionImpl")
    method = next(n for n in impl.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_sparse_npu")
    namespace = {
        "__package__": package.__name__,
        "torch": torch,
        "RainFusionPlan": object,
        "AttentionMetadata": object,
        "_supports_video_spans": lambda _: True,
        "_mindiesd_supports_pre_rearranged": lambda: True,
        "_INPUT_LAYOUT": "BSND",
        "_BLOCK_SIZE": 128,
        "_INNER_PRECISE": 0,
        "logger": SimpleNamespace(info_once=lambda *a, **k: None),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "rainfusion_attn.py"), "exec"), namespace)
    calls = []
    mindie = ModuleType("mindiesd")

    def sparse(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q + k + v

    mindie.sparse_attention = sparse
    monkeypatch.setitem(sys.modules, "mindiesd", mindie)
    geometry = ModuleType("mindiesd.layers.flash_attn.rf_geometry")
    geometry.RFGeometryCache = type("Cache", (), {})
    monkeypatch.setitem(sys.modules, geometry.__name__, geometry)
    owner = SimpleNamespace(rainfusion=SimpleNamespace(precision="bf16", sparsity=0.8), softmax_scale=0.5)
    return namespace["_forward_sparse_npu"], owner, calls


@pytest.mark.parametrize("cache", ["0", "1"])
def test_original_route_cache_policy_preserves_padded_tail(consumer, monkeypatch, cache):
    function, owner, calls = consumer
    monkeypatch.setenv("MINDIESD_RF_GEOMETRY_CACHE", cache)
    q, k, v = [torch.arange(32, dtype=torch.bfloat16).reshape(1, 8, 1, 4) for _ in range(3)]
    plan = SimpleNamespace(used_len=5, video_spans=[{"start": 0, "latent_shape": [1, 1, 5]}], pre_rearranged=False)
    result = function(owner, q, k, v, plan)
    assert result.shape == q.shape
    assert torch.equal(result[:, :5], (q + k + v)[:, :5])
    assert torch.equal(result[:, 5:], q[:, 5:])
    assert calls[0][0].shape[1] == 5
    assert ("geometry_cache" in calls[0][3]) == (cache == "1")
    if cache == "1":
        first_cache = calls[0][3]["geometry_cache"]
        function(owner, q, k, v, plan)
        assert calls[1][3]["geometry_cache"] is first_cache


def test_original_route_without_new_policy_keeps_no_cache(consumer, monkeypatch):
    function, owner, calls = consumer
    monkeypatch.delenv("MINDIESD_RF_GEOMETRY_CACHE", raising=False)
    q = torch.ones(1, 4, 1, 4)
    plan = SimpleNamespace(used_len=4, video_spans=[{"start": 0, "latent_shape": [1, 1, 4]}], pre_rearranged=False)
    function(owner, q, q, q, plan)
    assert "geometry_cache" not in calls[0][3]


@pytest.mark.parametrize("cache", ["0", "1"])
def test_pre_rearranged_consumer_honors_independent_explicit_cache(consumer, monkeypatch, cache):
    function, owner, calls = consumer
    owner._h3_optimization_config = SimpleNamespace(isolated=True, value=lambda name: cache)
    monkeypatch.setenv("MINDIESD_RF_GEOMETRY_CACHE", "bad")
    q = torch.arange(32, dtype=torch.bfloat16).reshape(1, 8, 1, 4)
    plan = SimpleNamespace(used_len=8, video_spans=[{"start": 0, "latent_shape": [1, 1, 8]}], pre_rearranged=True)
    metadata = SimpleNamespace(extra={"mindiesd_rf_geometry_cache": object()})
    result = function(owner, q, q, q, plan, attn_metadata=metadata)
    assert torch.equal(result, q + q + q)
    assert calls[0][3]["pre_rearranged"] is True
    assert ("geometry_cache" in calls[0][3]) == (cache == "1")
    assert calls[0][3]["pooling_policy"] == "legacy"
