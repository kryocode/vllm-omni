# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: F811 -- imported pytest fixtures are reused as argument names
"""Graph variant selection and explicit block-end prefetch release contracts."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_mindiesd_bridge import attention, metadata, prepare, tensor
from test_mindiesd_bridge import env as env

ROOT = Path(__file__).resolve().parents[2] / "vllm_omni/diffusion"
P = "VLLM_OMNI_MINIMAX_H3_"


@pytest.fixture(autouse=True)
def isolate_activated_environment():
    # Config.activate publishes derived flags beyond monkeypatch.setenv keys.
    with patch.dict(os.environ, {}, clear=False):
        yield


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "choice,resolved,prefix,single",
    [
        ("standard", "standard", "1", "0"),
        ("graph1", "standard", "1", "0"),
        ("graph2", "graph2", "0", "1"),
        ("off", "off", "0", "0"),
    ],
)
def test_graph_modes_publish_distinct_compatible_routes(choice, resolved, prefix, single):
    config = load("graph_mode_config", "models/minimax_h3/optimization.py").H3OptimizationConfig
    values = {P + "COMM_MODE": "pipeline", P + "RF_MODE": "cached", P + "GRAPH_MODE": choice}
    selected = config.from_env(values)
    selected.activate(values)
    assert selected.graph_mode == resolved
    assert values["MINDIESD_ATTENTION_GRAPH_PREFIX"] == prefix
    assert values["MINDIESD_ATTENTION_GRAPH_SINGLE"] == single
    assert config.from_env(values).group_snapshot() == selected.group_snapshot()


def test_explicit_group_switch_can_replace_activated_graph1_flags():
    config = load("graph_switch_config", "models/minimax_h3/optimization.py").H3OptimizationConfig
    values = {P + "COMM_MODE": "pipeline", P + "RF_MODE": "cached", P + "GRAPH_MODE": "standard"}
    config.from_env(values).activate(values)
    selected = config.from_env(values, settings={P + "GRAPH_MODE": "graph2"})
    selected.activate(values)
    assert values["MINDIESD_ATTENTION_GRAPH_PREFIX"] == "0"
    assert values["MINDIESD_ATTENTION_GRAPH_SINGLE"] == "1"


def test_graph2_rejects_conflicting_prefix_flag():
    config = load("graph_conflict_config", "models/minimax_h3/optimization.py").H3OptimizationConfig
    with pytest.raises(ValueError, match="conflict"):
        config.from_env(
            {
                P + "COMM_MODE": "pipeline",
                P + "RF_MODE": "cached",
                P + "GRAPH_MODE": "graph2",
                "MINDIESD_ATTENTION_GRAPH_PREFIX": "1",
            }
        )


def ticket_for(module, *, request=1, consumer=2):
    calls, counters = [], {}
    ticket = module.PrefetchTicket(
        module.PrefetchGatePolicy(True, request, consumer, "test"),
        owner_hook=object(),
        slot=0,
        slot_generation=1,
        launch_ag=lambda event: calls.append(event),
        ready_event=object(),
        counters=counters,
    )
    ticket.h2d_submitted()
    return ticket, calls, counters


def test_graph2_block_end_release_is_explicit_and_exact_once():
    module = load("graph_ticket", "offloader/prefetch_ticket.py")
    ticket, calls, counters = ticket_for(module)
    with ticket.scope():
        ticket.defer_to_block_end()
        assert not calls
    ticket.finish()  # DLO post_forward also checks the same ticket.
    assert calls == [None] and ticket.launch_attempts == 1
    assert counters["released_at_block_end"] == 1
    assert counters.get("released_after_input", 0) == 0 and counters["live"] == 0
    with pytest.raises(RuntimeError):
        ticket.defer_to_block_end()


def test_graph2_must_not_mix_first_head_and_block_end_release():
    module = load("graph_ticket_mixed", "offloader/prefetch_ticket.py")
    ticket, calls, _ = ticket_for(module)
    with pytest.raises(RuntimeError):
        with ticket.scope():
            ticket.defer_to_block_end()
            ticket.release_after_event(object())
    assert not calls and ticket.state == "FAILED"


def test_failed_graph2_block_never_flushes_deferred_allgather():
    module = load("graph_ticket_failure", "offloader/prefetch_ticket.py")
    ticket, calls, _ = ticket_for(module)
    with pytest.raises(ValueError, match="failed body"):
        with ticket.scope():
            ticket.defer_to_block_end()
            raise ValueError("failed body")
    assert not calls and ticket.state == "FAILED"


def test_graph1_still_refuses_missing_first_head_release():
    module = load("graph_ticket_original", "offloader/prefetch_ticket.py")
    ticket, calls, _ = ticket_for(module)
    with pytest.raises(RuntimeError, match="promised prefetch release"):
        with ticket.scope():
            pass
    assert not calls and ticket.state == "FAILED"


@pytest.mark.parametrize("sparse", [True, False])
def test_graph2_route_defers_sparse_ticket_and_keeps_dense_input_gate(env, monkeypatch, sparse):
    monkeypatch.setenv(P + "COMM_MODE", "pipeline")
    monkeypatch.setenv(P + "RF_MODE", "cached")
    monkeypatch.setenv(P + "GRAPH_MODE", "graph2")
    monkeypatch.setenv(P + "WEIGHT_GATE", "1")
    monkeypatch.setattr(sys.modules["mindiesd.parallel"], "single_graph_supported", lambda: True, raising=False)
    executor = sys.modules["mindiesd.parallel"].AttentionExecutor
    previous = executor.execution_signature
    monkeypatch.setattr(
        executor, "execution_signature", lambda self, **kwargs: previous(self, **kwargs) + (("graph_single", "1"),)
    )
    owner, layer, q = env.h3.MindIESDH3Parallel(), attention(sparse=sparse), tensor()
    prepare(env, owner, layer, q)
    owner._bridge.executor.supports_input_chunk_events = lambda **kwargs: True
    monkeypatch.setattr(env.h3, "_graph_signature", lambda *args: ("stable_sparse", False, ""))
    module = load("graph_route_ticket", "offloader/prefetch_ticket.py")
    ticket, calls, _ = ticket_for(module, request=(id(owner), owner._request_id), consumer=id(layer))
    seen = []
    event = object()

    def execute(*args, **kwargs):
        seen.append(kwargs)
        assert not calls
        if sparse:
            assert kwargs["on_input_chunk_ready"] is None
            assert kwargs["graph_signature"] is not None
        else:
            assert kwargs["graph_signature"] is None
            kwargs["on_input_chunk_ready"](SimpleNamespace(index=0), event)
        return q, {}

    monkeypatch.setattr(owner._bridge, "run", execute)
    with ticket.scope():
        owner.try_attention(layer, q, q, q, metadata(), prefetch_ticket=ticket)
        assert bool(calls) is not sparse
    assert len(seen) == 1 and calls == ([None] if sparse else [event])


def test_graph2_refuses_older_mindiesd_without_whole_graph_api(env, monkeypatch):
    monkeypatch.setenv(P + "COMM_MODE", "pipeline")
    monkeypatch.setenv(P + "RF_MODE", "cached")
    monkeypatch.setenv(P + "GRAPH_MODE", "graph2")
    owner, layer, q = env.h3.MindIESDH3Parallel(), attention(sparse=True), tensor()
    with pytest.raises(RuntimeError, match="unsupported"):
        prepare(env, owner, layer, q)


def test_graph2_refuses_partial_upgrade_missing_executor_protocol(env, monkeypatch):
    monkeypatch.setenv(P + "COMM_MODE", "pipeline")
    monkeypatch.setenv(P + "RF_MODE", "cached")
    monkeypatch.setenv(P + "GRAPH_MODE", "graph2")
    monkeypatch.setattr(sys.modules["mindiesd.parallel"], "single_graph_supported", lambda: True, raising=False)
    owner, layer, q = env.h3.MindIESDH3Parallel(), attention(sparse=True), tensor()
    with pytest.raises(RuntimeError, match="unsupported"):
        prepare(env, owner, layer, q)


@pytest.mark.parametrize("layout", ["post_embed", "pre_embed"])
@pytest.mark.parametrize("graph", ["graph1", "graph2"])
def test_rf_and_graph_choices_are_independent(layout, graph):
    config = load("graph_rf_config", "models/minimax_h3/optimization.py").H3OptimizationConfig
    selected = config.from_env(
        {P + "COMM_MODE": "pipeline", P + "RF_MODE": "optimized", P + "RF_LAYOUT": layout, P + "GRAPH_MODE": graph}
    )
    assert selected.rf_layout == ("post_embed" if layout == "post_embed" else "legacy_input_hoist")
    assert selected.graph_mode == ("standard" if graph == "graph1" else "graph2")
