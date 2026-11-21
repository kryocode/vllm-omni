# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-free regression for matching Ulysses and CP HCCL execution modes."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model, pytest.mark.cpu]
SOURCE = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/distributed/parallel_state.py"


def load_group_builders(monkeypatch, mode, *, npu=True, entry="_set_ulysses_allgather_pg"):
    created = []

    def new_group(ranks, **kwargs):
        group = SimpleNamespace(ranks=list(ranks), kwargs=kwargs)
        created.append(group)
        return group

    class Options:
        def __init__(self):
            self.hccl_config = {}

    extension = ModuleType("torch_npu")
    extension._C = SimpleNamespace(_distributed_c10d=SimpleNamespace(ProcessGroupHCCL=SimpleNamespace(Options=Options)))
    monkeypatch.setitem(sys.modules, "torch_npu", extension)
    tree = ast.parse(SOURCE.read_text())
    names = {"build_ulysses_allgather_rank_groups", "_set_ulysses_allgather_pg", "set_seq_parallel_pg"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and (node.name in names or (node.name.startswith("_new_") and node.name.endswith("_group")))
    ]
    namespace = {
        "torch": SimpleNamespace(distributed=SimpleNamespace(new_group=new_group)),
        "envs": SimpleNamespace(VLLM_OMNI_ULYSSES_HCCL_EXPANSION_MODE=mode),
        "current_omni_platform": SimpleNamespace(is_npu=lambda: npu),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[entry], created


@pytest.mark.parametrize("mode", [2, 3])
@pytest.mark.parametrize("rank", range(16))
def test_cp_uses_same_explicit_hccl_mode_as_ulysses(monkeypatch, mode, rank):
    build, created = load_group_builders(monkeypatch, mode)
    ulysses, ring, cp = build(8, 2, rank, 16, None)
    assert ulysses.ranks == list(range((rank // 8) * 8, (rank // 8 + 1) * 8))
    assert cp.ranks == [rank % 8, rank % 8 + 8]
    assert ring.ranks == [rank]
    communication_groups = [group for group in created if len(group.ranks) > 1]
    assert len(communication_groups) == 10
    for group in communication_groups:
        assert group.kwargs.get("backend") == "hccl"
        options = group.kwargs.get("pg_options")
        assert options is not None, f"missing explicit HCCL options for {group.ranks}"
        assert options.hccl_config == {"hccl_op_expansion_mode": mode}
    assert len({id(group.kwargs["pg_options"]) for group in communication_groups}) == 10
    assert [group.ranks for group in created] == [
        list(range(8)),
        list(range(8, 16)),
        *[[rank, rank + 8] for rank in range(8)],
        *[[rank] for rank in range(16)],
    ]


@pytest.mark.parametrize("mode", [None, 2, 3])
@pytest.mark.parametrize("rank", range(4))
def test_pure_cp_uses_the_selected_hccl_mode(monkeypatch, mode, rank):
    build, created = load_group_builders(monkeypatch, mode, entry="set_seq_parallel_pg")
    ulysses, ring, cp = build(1, 1, rank, 4, sp_allgather_degree=4)
    assert cp.ranks == [0, 1, 2, 3]
    assert ulysses.ranks == ring.ranks == [rank]
    assert [group.ranks for group in created] == [list(range(4)), *[[r] for r in range(4)], *[[r] for r in range(4)]]
    if mode is None:
        assert all(not group.kwargs for group in created)
    else:
        assert cp.kwargs.get("backend") == "hccl"
        options = cp.kwargs.get("pg_options")
        assert options is not None
        assert options.hccl_config == {"hccl_op_expansion_mode": mode}


def test_pure_cp_rejects_explicit_npu_mode_on_cpu(monkeypatch):
    build, created = load_group_builders(monkeypatch, 2, npu=False, entry="set_seq_parallel_pg")
    with pytest.raises(ValueError, match="requires NPU"):
        build(1, 1, 0, 4, sp_allgather_degree=4)
    assert created == []


def test_unset_mode_preserves_default_group_creation(monkeypatch):
    build, created = load_group_builders(monkeypatch, None)
    build(8, 2, 0, 16, None)
    assert all(not group.kwargs for group in created)


@pytest.mark.parametrize("mode,npu", [(1, True), (2, False)])
def test_invalid_explicit_mode_fails_before_creating_groups(monkeypatch, mode, npu):
    build, created = load_group_builders(monkeypatch, mode, npu=npu)
    with pytest.raises(ValueError, match="requires NPU"):
        build(8, 2, 0, 16, None)
    assert created == []
