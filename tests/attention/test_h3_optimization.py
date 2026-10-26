# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for H3 layout ownership and project optimization settings."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_module():
    return load("contract_h3_optimization", "models/minimax_h3/optimization.py")


@pytest.fixture
def layout_module():
    return load("contract_minimax_h3_rf_reorder_mode", "models/minimax_h3/rf_layout.py")


def test_legacy_layout_precedence_and_preserved_defaults(config_module):
    config = config_module.H3OptimizationConfig
    assert config.from_env({}).rf_layout == "none"
    profile = {"VLLM_OMNI_MINIMAX_H3_OPTIMIZATION_PROFILE": "u8"}
    assert config.from_env(profile).rf_layout == "post_embed"
    assert config.from_env({**profile, "VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "1"}).rf_layout == "legacy_input_hoist"
    assert config.from_env({**profile, "H3_SAFE_RF_POST_EMBED": "0"}).rf_layout == "none"
    assert config.from_env({"VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "true"}).rf_layout == "legacy_input_hoist"
    assert config.from_env(profile, settings={"VLLM_OMNI_MINIMAX_H3_RF_LAYOUT": "none"}).rf_layout == "none"
    assert not config.from_env({"VLLM_OMNI_MINIMAX_H3_MINDIESD": "0", **profile}).enabled("MINDIESD")


@pytest.mark.parametrize(
    "env",
    [
        {"H3_SAFE_RF_POST_EMBED": "1", "VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "1"},
        {"H3_SAFE_RF_POST_EMBED": "1", "H3_RF_PRESERVE_ADALN": "0"},
        {"VLLM_OMNI_MINIMAX_H3_RF_LAYOUT": "post_embed", "H3_SAFE_RF_POST_EMBED": "0"},
        {"H3_RF_EXACT_POOL": "1"},
    ],
)
def test_conflicting_or_unvalidated_layout_is_rejected(config_module, env):
    with pytest.raises(ValueError):
        config_module.H3OptimizationConfig.from_env(env)


def test_activation_never_overrides_explicit_compatibility(config_module):
    env = {
        "VLLM_OMNI_MINIMAX_H3_OPTIMIZATION_PROFILE": "u8",
        "MINDIESD_ATTENTION_GRAPH": "0",
        "VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "1",
    }
    config_module.H3OptimizationConfig.from_env(env).activate(env)
    assert env["MINDIESD_ATTENTION_GRAPH"] == "0"
    assert env["VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER"] == "1"
    assert env["MINDIESD_QBMM_NZ"] == "1"


def test_explicit_settings_are_published_to_legacy_consumers(config_module):
    graph = "MINDIESD_ATTENTION_GRAPH"
    env = {"VLLM_OMNI_MINIMAX_H3_OPTIMIZATION_PROFILE": "u8", graph: "0"}
    config = config_module.H3OptimizationConfig.from_env(env, settings={graph: "1"})
    config.activate(env)
    assert dict(config.switches)[graph] == env[graph] == "1"
    env = {"VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "1"}
    config = config_module.H3OptimizationConfig.from_env(env, settings={"VLLM_OMNI_MINIMAX_H3_RF_LAYOUT": "post_embed"})
    config.activate(env)
    assert env["VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER"] == "0"
    assert env["H3_SAFE_RF_POST_EMBED"] == "1" and config.rf_layout == "post_embed"
    assert config_module.H3OptimizationConfig.from_env(env).rf_layout == "post_embed"


@pytest.mark.parametrize(
    "layout,legacy",
    [
        ("post_embed", {"VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "0"}),
        ("legacy_input_hoist", {"H3_SAFE_RF_POST_EMBED": "0"}),
        ("none", {"H3_SAFE_RF_POST_EMBED": "0", "VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "0"}),
    ],
)
def test_enum_can_coexist_with_explicitly_disabled_other_layout(config_module, layout, legacy):
    env = {"VLLM_OMNI_MINIMAX_H3_RF_LAYOUT": layout, **legacy}
    config = config_module.H3OptimizationConfig.from_env(env)
    config.activate(env)
    assert config.rf_layout == config_module.H3OptimizationConfig.from_env(env).rf_layout == layout
    assert env["H3_SAFE_RF_POST_EMBED"] == str(int(layout == "post_embed"))
    assert env["VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER"] == str(int(layout == "legacy_input_hoist"))


@pytest.mark.parametrize(
    "layout,legacy",
    [
        ("post_embed", {"H3_SAFE_RF_POST_EMBED": "0"}),
        ("post_embed", {"VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "1"}),
        ("legacy_input_hoist", {"VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER": "0"}),
        ("none", {"H3_SAFE_RF_POST_EMBED": "1"}),
    ],
)
def test_enum_rejects_contradictory_old_layout_switches(config_module, layout, legacy):
    with pytest.raises(ValueError, match="conflicts"):
        config_module.H3OptimizationConfig.from_env({"VLLM_OMNI_MINIMAX_H3_RF_LAYOUT": layout, **legacy})


@pytest.mark.parametrize(
    "permutation,world",
    [
        ([0, 4, 8, 1, 5, 9, 2, 6, 3, 7, 10, 11], 3),
        ([7, 3, 0, 5, 1, 6, 4, 2], 2),
    ],
)
def test_variable_split_routes_match_global_permutation_and_inverse(layout_module, monkeypatch, permutation, world):
    p = torch.tensor(permutation)
    canonical = torch.arange(len(p) * 3).reshape(len(p), 3)
    local = len(p) // world

    def apply(rows, ordering):
        routes = [
            layout_module._DeviceRoute(layout_module.build_row_route(ordering, r, world), "cpu") for r in range(world)
        ]
        shards = list(rows.chunk(world))
        sends = [shards[r].index_select(0, route.send) for r, route in enumerate(routes)]
        segments = [list(buf.split(list(route.plan.send_counts))) for buf, route in zip(sends, routes)]
        outputs = []
        for rank, route in enumerate(routes):

            def exchange(output, value, *, output_split_sizes, input_split_sizes, group):
                assert input_split_sizes == list(route.plan.send_counts)
                assert output_split_sizes == [len(segments[src][rank]) for src in range(world)]
                output.copy_(torch.cat([segments[src][rank] for src in range(world)]))

            monkeypatch.setattr(layout_module.dist, "all_to_all_single", exchange)
            outputs.append(route.run(shards[rank], object()))
        return torch.cat(outputs)

    rf_rows = apply(canonical, p.numpy())
    assert torch.equal(rf_rows, canonical[p])
    assert torch.equal(apply(rf_rows, torch.argsort(p).numpy()), canonical)
    assert rf_rows.shape[0] == world * local


def test_original_owner_adaln_and_hidden_rope_pairing(layout_module, monkeypatch):
    # Rank-local projection results deliberately differ; using the receiving
    # rank's table instead of the original owner would fail this comparison.
    indices = torch.tensor([0, 5, 2, 7])
    hidden = torch.arange(8).reshape(4, 2).to(torch.bfloat16)
    rope = hidden + 50
    inverse = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    tags = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1])
    ordering = torch.tensor([3, 0, 2, 1])
    route = SimpleNamespace(run=lambda rows, group: rows[ordering])
    plan = layout_module.H3RFBlockLayout(route, route, indices, object(), 2, 4)
    h, r, combined = plan.enter(hidden, rope, inverse, tags, modality_count=2, adaln_rows=4)
    assert torch.equal(h, hidden[ordering]) and torch.equal(r, rope[ordering])
    assert torch.equal(combined, inverse[indices] * 2 + tags[indices] + (indices // 4) * 4)
    parameters = tuple(torch.arange(4).reshape(4, 1).to(torch.bfloat16) + c * 10 for c in range(6))

    def gather(out, packed, group):
        out.copy_(torch.cat((packed, packed + 100)))

    monkeypatch.setattr(layout_module.dist, "all_gather_into_tensor", gather)
    gathered = plan.adaln_parameters(parameters)
    selected = gathered[0][combined]
    expected = (inverse[indices] * 2 + tags[indices]).to(torch.bfloat16).reshape(-1, 1)
    expected += (indices // 4).reshape(-1, 1) * 100
    assert torch.equal(selected, expected)
    assert not torch.equal(selected, parameters[0][combined % 4])


def test_local_context_reentrancy_and_exception_cleanup(layout_module):
    adapter = layout_module.H3RFLayoutAdapter()
    with pytest.raises(ValueError):
        with adapter.forward_scope(enabled=True):
            with pytest.raises(RuntimeError, match="reentrant"):
                with adapter.forward_scope(enabled=True):
                    pass
            raise ValueError("model failed")
    with adapter.forward_scope(enabled=True):
        pass
    assert adapter._key is None and adapter._plan is None and not adapter._active


def test_offload_aggregation_deduplicates_hooks_and_retains_live_gauge():
    stats = load("contract_offload_execution_stats", "offloader/execution_stats.py")
    hook = SimpleNamespace(
        execution_snapshot=lambda: {
            "prefetch": {"released_after_input": 4, "live": 2, "failed": 1},
            "failed": True,
            "nz": {"selected_weights": 3, "packed_weights": 3, "view_count": 6, "bind_count": 12},
        }
    )
    registry = SimpleNamespace(get_hook=lambda name: hook)
    model = SimpleNamespace(
        modules=lambda: (SimpleNamespace(_hook_registry=registry), SimpleNamespace(_hook_registry=registry))
    )
    snapshot = stats.collect_offload_execution_snapshot(model)
    assert snapshot["hooks"] == snapshot["nz"]["plans"] == 1
    assert snapshot["prefetch"]["live"] == 2 and snapshot["failed"]
    assert snapshot["nz"]["bind_count"] == 12
    delta = stats.counter_delta({"released_after_input": 1, "live": 9}, snapshot["prefetch"])
    assert delta["released_after_input"] == 3 and delta["failed"] == 1 and "live" not in delta
