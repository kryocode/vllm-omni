# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of both RF layout choices and the actual input-index hoist."""

import ast
import functools
import importlib.util
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion"
P = "VLLM_OMNI_MINIMAX_H3_"


@pytest.fixture
def config(monkeypatch):
    spec = importlib.util.spec_from_file_location("_pre_embed_config", ROOT / "models/minimax_h3/optimization.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    for name in tuple(os.environ):
        if name.startswith((P, "MINDIESD_", "H3_RF_", "H3_SAFE_RF_")):
            monkeypatch.delenv(name)
    return module.H3OptimizationConfig


@pytest.mark.parametrize("grouped", [False, True])
def test_pre_embed_enum_activates_only_input_hoist_and_round_trips(config, grouped):
    env = {P + "RF_LAYOUT": "pre_embed"}
    if grouped:
        env.update({P + "RF_MODE": "optimized", P + "COMM_MODE": "pipeline"})
    selected = config.from_env(env)
    # Retain the historical internal name for existing consumers and receipts.
    assert selected.rf_layout == "legacy_input_hoist"
    selected.activate(env)
    assert env[P + "RF_HOIST_REORDER"] == "1"
    assert env["H3_SAFE_RF_POST_EMBED"] == "0"
    assert config.from_env(env).rf_layout == selected.rf_layout


def test_pre_embed_cannot_coexist_with_post_embed_flag(config):
    with pytest.raises(ValueError, match="conflict"):
        config.from_env({P + "RF_LAYOUT": "pre_embed", "H3_SAFE_RF_POST_EMBED": "1"})


def test_explicit_pre_embed_setting_replaces_post_embed_environment(config):
    env = {P + "RF_LAYOUT": "post_embed", "H3_SAFE_RF_POST_EMBED": "1"}
    selected = config.from_env(env, settings={P + "RF_LAYOUT": "pre_embed"})
    selected.activate(env)
    assert env["H3_SAFE_RF_POST_EMBED"] == "0"
    assert env[P + "RF_HOIST_REORDER"] == "1"


@pytest.mark.parametrize("initial,target", [("post_embed", "pre_embed"), ("pre_embed", "post_embed")])
def test_explicit_layout_switches_previously_activated_groups(config, initial, target):
    env = {P + "COMM_MODE": "pipeline", P + "RF_MODE": "optimized", P + "RF_LAYOUT": initial}
    config.from_env(env).activate(env)
    selected = config.from_env(env, settings={P + "RF_LAYOUT": target})
    selected.activate(env)
    assert env["H3_SAFE_RF_POST_EMBED"] == str(int(target == "post_embed"))
    assert env[P + "RF_HOIST_REORDER"] == str(int(target == "pre_embed"))
    assert config.from_env(env).rf_layout == selected.rf_layout


@pytest.mark.parametrize("grouped", [False, True])
def test_explicit_layout_still_rejects_contradictory_explicit_flags(config, grouped):
    with pytest.raises(ValueError, match="conflict"):
        config.from_env(
            {P + "RF_MODE": "optimized"} if grouped else {},
            settings={
                P + "RF_LAYOUT": "pre_embed",
                "H3_SAFE_RF_POST_EMBED": "1",
            },
        )


def _branch_class(config):
    namespace = {
        "dataclass": dataclass,
        "replace": replace,
        "torch": torch,
        "os": os,
        "functools": functools,
        "H3OptimizationConfig": config,
        "_RF_BLOCK_SIZE": 128,
        "MINIMAX_H3_VIDEO_ROW_WIDTH": 96,
        "MINIMAX_H3_AUDIO_ROW_WIDTH": 32,
        "logger": SimpleNamespace(info_once=lambda *a, **k: None, warning_once=lambda *a, **k: None),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    metadata = ast.parse((ROOT / "attention/backends/abstract.py").read_text(encoding="utf-8"))
    definitions = [
        node
        for node in metadata.body
        if isinstance(node, ast.ClassDef) and node.name in ("VideoTokenSpan", "VideoTokenLayout")
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[])),
            "<real RF metadata>",
            "exec",
        ),
        namespace,
    )
    path = ROOT / "models/minimax_h3/denoise_loop.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_rf_hoist_enabled", "_mindiesd_accepts_pre_rearranged")
    ]
    branch = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3DenoiseBranch"
    )
    branch.body = [
        node
        for node in branch.body
        if isinstance(node, ast.FunctionDef) and node.name in ("__init__", "_hoist_rf_row_reorder", "rf_layout_proof")
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *nodes, branch], type_ignores=[])), str(path), "exec"
        ),
        namespace,
    )
    return namespace["MiniMaxH3DenoiseBranch"]


@pytest.mark.parametrize(
    "layout,hoisted,accepts_flag",
    [
        ("pre_embed", True, True),
        ("post_embed", False, True),
        ("none", False, True),
        ("pre_embed", False, False),
    ],
)
def test_selected_layout_routes_real_branch_indices_once(config, monkeypatch, layout, hoisted, accepts_flag):
    monkeypatch.setenv(P + "RF_LAYOUT", layout)
    permutation = torch.tensor([0, 1, 3, 2, 4, 5, 8, 6, 9, 7])
    calls = []
    for name in ("mindiesd", "mindiesd.layers", "mindiesd.layers.flash_attn"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    def sparse(*args, pre_rearranged=False, **kwargs):
        pass

    def old_sparse(*args, **kwargs):
        pass

    sys.modules["mindiesd"].sparse_attention = sparse if accepts_flag else old_sparse
    geometry = ModuleType("mindiesd.layers.flash_attn.rf_geometry")

    def reorder(spans, length, pool, tensor_layout, device):
        calls.append((spans, length, pool, tensor_layout))
        return permutation, torch.argsort(permutation)

    geometry.rf_multi_span_permutation = reorder
    monkeypatch.setitem(sys.modules, geometry.__name__, geometry)
    packed = dict(
        seq_len=torch.tensor(12),
        img_pos=torch.tensor([2, 3, 6, 7, 8, 9]),
        audio_pos=torch.tensor([4, 5]),
        text_pos=torch.tensor([0, 1]),
        update_mask=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool),
        cu_seqlens=torch.tensor([0, 10, 12]),
        img_position_ids=torch.arange(36).reshape(12, 3),
        video_spans=[
            dict(start=2, latent_grid=(1, 1, 2), role="reference"),
            dict(start=6, latent_grid=(1, 1, 4), role="target"),
        ],
    )
    before = {key: value.clone() for key, value in packed.items() if torch.is_tensor(value)}
    tags = torch.tensor([1, 1, 0, 0, 2, 2, 0, 0, 0, 0, -1, -1])
    branch = _branch_class(config)(
        packed=packed, text_embeddings=torch.ones(2, 4), token_tags=tags, device=torch.device("cpu")
    )
    proof = branch.rf_layout_proof()
    assert proof["request_hoist_calls"] == int(hoisted)
    assert proof["pre_rearranged"] is hoisted
    assert len(calls) == int(hoisted)
    order = torch.cat((permutation, torch.tensor([10, 11]))) if hoisted else torch.arange(12)
    inverse = torch.argsort(order)
    for key, actual in [
        ("img_pos", branch.img_pos_dev),
        ("audio_pos", branch.audio_pos_dev),
        ("text_pos", branch.text_pos_dev),
    ]:
        assert torch.equal(actual, inverse[before[key]])
    assert torch.equal(branch.img_position_ids_dev, before["img_position_ids"][order])
    assert torch.equal(branch.token_tags_dev, tags[order])
    assert torch.equal(branch.update_mask_dev, before["update_mask"])
    assert torch.equal(branch.static_kwargs["packed_seq_params"]["cu_seqlens_q"], before["cu_seqlens"])
    assert ("vsa_prefix_segments" in branch.static_kwargs["packed_seq_params"]) is not hoisted
    for key, value in before.items():
        assert torch.equal(packed[key], value), key
    if hoisted:
        with pytest.raises(RuntimeError, match="once per branch"):
            branch._hoist_rf_row_reorder(branch.static_kwargs["video_token_layout"].video_spans)
