# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: F811
"""CLI integration, environment isolation and legacy-policy equivalence."""

import argparse
import ast
import importlib.util
import os
import pickle
import sys
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_mindiesd_bridge import attention, metadata, prepare, tensor
from test_mindiesd_bridge import env as env

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def validation():
    return load("vllm_omni.diffusion.optimization_config", "vllm_omni/diffusion/optimization_config.py")


def test_actual_cli_registration_and_worker_field_transport(validation):
    tree = ast.parse((ROOT / "vllm_omni/entrypoints/cli/serve.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "execution_help" for t in node.targets)
    ]
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and ast.unparse(node.iter) == "H3_EXECUTION_OPTIONS.items()"
    ]
    parser = argparse.ArgumentParser()
    scope = {"H3_EXECUTION_OPTIONS": validation.H3_EXECUTION_OPTIONS, "omni_config_group": parser}
    exec(compile(ast.Module(body=nodes + loops, type_ignores=[]), "<production-cli-registration>", "exec"), scope)
    args = parser.parse_args(
        [
            "--diffusion-attn-head-overlap-mode",
            "pipeline",
            "--diffusion-attn-rf-cache-mode",
            "cached",
            "--minimax-h3-rf-reorder-mode",
            "post_embed",
            "--diffusion-attn-head-overlap-graph-mode",
            "graph2",
            "--dlo-allgather-gate-mode",
            "first_head",
            "--dlo-weight-format",
            "nz",
        ]
    )
    transported = pickle.loads(pickle.dumps(vars(args)))
    assert transported["diffusion_attn_head_overlap_graph_mode"] == "graph2"
    assert vars(parser.parse_args([])) == {
        name: default for name, (_, default) in validation.H3_EXECUTION_OPTIONS.items()
    }
    for filename, classname in [
        ("vllm_omni/engine/arg_utils.py", "OrchestratorArgs"),
        ("vllm_omni/config/stage_config.py", "StageDeployConfig"),
        ("vllm_omni/config/omni_config.py", "_DiffusionConfigProjection"),
        ("vllm_omni/diffusion/data.py", "OmniDiffusionConfig"),
    ]:
        cls = next(
            node
            for node in ast.parse((ROOT / filename).read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef) and node.name == classname
        )
        fields = {
            node.target.id for node in cls.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert set(transported) <= fields, filename
    with pytest.raises(SystemExit):
        parser.parse_args(["--diffusion-attn-head-overlap-graph-mode", "standard"])
    for removed_name in (
        "--diffusion-attn-comm-mode",
        "--diffusion-rf-mode",
        "--h3-rf-layout",
        "--diffusion-attn-graph-mode",
        "--dlo-weight-ag-gate",
        "--dlo-weight-layout",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([removed_name, "off"])


@pytest.mark.parametrize("graph,layout", [("off", "none"), ("graph1", "post_embed"), ("graph2", "pre_embed")])
def test_cli_matches_previous_resolved_execution_policy(validation, graph, layout):
    config = load(
        "cli_policy_equivalence", "vllm_omni/diffusion/models/minimax_h3/optimization.py"
    ).H3OptimizationConfig
    current = config.from_config(
        SimpleNamespace(
            diffusion_attn_head_overlap_mode="pipeline",
            diffusion_attn_rf_cache_mode="cached",
            minimax_h3_rf_reorder_mode=layout,
            diffusion_attn_head_overlap_graph_mode=graph,
            dlo_allgather_gate_mode="first_head",
            dlo_weight_format="nz",
            enable_distributed_layerwise_offload=True,
        )
    )
    prefix = "VLLM_OMNI_MINIMAX_H3_"
    old = config.from_env(
        {
            prefix + "COMM_MODE": "pipeline",
            prefix + "RF_MODE": "cached" if layout == "none" else "optimized",
            prefix + "RF_LAYOUT": layout,
            prefix + "GRAPH_MODE": graph,
            prefix + "WEIGHT_GATE": "1",
            prefix + "WEIGHT_LAYOUT": "nz",
        }
    )
    assert current.switches == old.switches
    assert current.rf_layout == old.rf_layout


@pytest.mark.parametrize("mode", ["off", "graph2"])
def test_owner_consumes_explicit_config_under_invalid_environment(env, monkeypatch, validation, mode):
    config = env.h3.H3OptimizationConfig.from_config(
        SimpleNamespace(
            diffusion_attn_head_overlap_mode="pipeline",
            diffusion_attn_rf_cache_mode="cached",
            diffusion_attn_head_overlap_graph_mode=mode,
        )
    )
    dirty = {
        "MINDIESD_ATTENTION_GRAPH_PREFIX": "bad",
        "MINDIESD_ATTENTION_GRAPH": "bad",
        "MINDIESD_QBMM_NZ": "bad",
        "MINDIESD_FIRST_HEAD_AG_GATE": "bad",
        "MINDIESD_RF_GEOMETRY_CACHE": "bad",
    }
    for name, value in dirty.items():
        monkeypatch.setenv(name, value)
    parallel = sys.modules["mindiesd.parallel"]
    monkeypatch.setattr(parallel, "single_graph_supported", lambda: True, raising=False)
    monkeypatch.setattr(
        parallel.AttentionExecutor,
        "execution_signature",
        lambda self, **kw: (
            tuple(sorted(vars(self.config).items())),
            ("graph_single", str(int(self.config.graph_mode == "graph2"))),
        ),
    )
    owner, layer, q = env.h3.MindIESDH3Parallel(config), attention(sparse=True), tensor()
    layer.attention._h3_optimization_config = config
    monkeypatch.setattr(env.h3, "_graph_signature", lambda *a, **kw: ("stable_sparse", False, ""))
    prepare(env, owner, layer, q)
    assert owner._bridge.executor.config.graph_mode == mode
    assert owner._backends.build(layer, metadata()).is_sparse
    assert dict(owner._request_policies[id(layer)][-1][1])["MINDIESD_RF_GEOMETRY_CACHE"] == "1"
    owner.try_attention(layer, q, q, q, metadata())
    assert all(os.environ[name] == value for name, value in dirty.items())


def test_explicit_pipeline_refuses_environment_driven_direct_fallback(env, validation):
    config = env.h3.H3OptimizationConfig.from_config(SimpleNamespace(diffusion_attn_head_overlap_mode="pipeline"))
    owner = env.h3.MindIESDH3Parallel(config)
    q = tensor()
    with pytest.raises(RuntimeError, match="prepared shared request"):
        owner.try_attention(attention(), q, q, q, metadata())


def test_nz_bind_and_restore_keep_explicit_request_without_environment(monkeypatch):
    module = load("cli_nz_plan", "vllm_omni/diffusion/offloader/qbmm_nz.py")
    calls = []
    library = SimpleNamespace(qbmm_nz_enabled=lambda requested=None: calls.append(requested) or requested is True)
    plan = module.QbmmNZOffloadPlan({}, set(), library=library)
    plan.explicit_policy = True
    monkeypatch.setenv("MINDIESD_QBMM_NZ", "bad")
    assert plan.prepare_bind(0, {}, {}, generation=1, compute_stream=None) == set()
    plan.request_owner = SimpleNamespace(assert_weight_layout_live=lambda *a, **kw: None)
    plan.contract = ("nz", "1", ())
    plan.block = SimpleNamespace(_mindiesd_qbmm_nz_contract=plan.contract)
    with plan.restore_scope(drained=True):
        assert plan.restoring
    assert calls == [True, True]


@pytest.mark.parametrize("has_owner,enabled", [(False, True), (True, True), (True, False)])
def test_offload_nz_contract_excludes_refiner_blocks(monkeypatch, has_owner, enabled):
    """Run actual hook setup with CPU stand-ins for storage/device allocation."""
    source = ROOT / "vllm_omni/diffusion/offloader/distributed_layerwise_backend.py"
    cls = next(
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DistributedLayerwiseOffloadHook"
    )
    cls.bases = [ast.Name(id="BaseHook", ctx=ast.Load())]
    cls.body = [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("__init__", "initialize_hook")
    ]

    class BaseHook:
        def initialize_hook(self, module):
            return module

    plan = SimpleNamespace(stats=lambda: {})
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "cli_hook_scope.qbmm_nz",
        SimpleNamespace(create_qbmm_nz_plan=lambda *args, **kwargs: calls.append(args[0]) or plan),
    )
    scope = dict(
        __name__="cli_hook_scope.hook",
        __package__="cli_hook_scope",
        BaseHook=BaseHook,
        nn=torch.nn,
        torch=torch,
        Any=Any,
        os=os,
        chain=chain,
        clear_tensor_storage=lambda tensors: None,
        logger=SimpleNamespace(info=lambda *args: None),
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    next_block = torch.nn.Module()
    if has_owner:
        config = SimpleNamespace(isolated=True, value=lambda name: "1")
        next_block.qbmm_nz_request_owner = lambda: SimpleNamespace(optimization_config=config)
    monkeypatch.delenv("MINDIESD_QBMM_NZ", raising=False)
    hook = scope["DistributedLayerwiseOffloadHook"](
        next_block,
        torch.device("cpu"),
        None,
        1,
        0,
        copy_stream=object(),
        comm_stream=object(),
        shared_buffers=[{}, {}],
        materialization_probe_tensor=object(),
        qbmm_nz=enabled,
    )
    hook._shard_and_pin = lambda *args, **kwargs: ({}, {})
    hook.initialize_hook(torch.nn.Module())
    active = has_owner and enabled
    assert hook._qbmm_nz_requested is active
    assert bool(calls) is active
    assert hasattr(next_block, "_mindiesd_qbmm_nz_contract") is active
    if active:
        assert next_block._mindiesd_qbmm_nz_contract == ("qbmm_nz_v1", "1", ())
        assert plan.contract == next_block._mindiesd_qbmm_nz_contract
