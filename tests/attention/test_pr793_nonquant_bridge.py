# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: F811
"""The paired SD topology owns H3's local valid/padding boundary."""

import sys

import pytest
import torch
from test_mindiesd_bridge import attention, metadata, prepare, tensor
from test_mindiesd_bridge import env as env

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_request_resolves_sd_rank_window_once_and_h3_uses_it(env, monkeypatch):
    calls = []

    def rank_window(rank, world, s_local, used_len):
        calls.append((rank, world, s_local, used_len))
        return 5, 7, 2

    monkeypatch.setattr(sys.modules["mindiesd.parallel"], "rank_window", rank_window, raising=False)
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    assert prepare(env, owner, layer, value)
    assert owner._bridge.local_sequence_window == (5, 7, 2)
    for _ in range(2):
        result = owner.try_attention(layer, value, value, value, metadata())
        assert torch.equal(result[:, 2:], value[:, 2:])
        assert torch.all(result[:, :2] == 7)
    assert calls == [(1, 2, 5, 7)]
    assert env.agreements == 1
    owner.finish_request()
    assert owner._bridge.local_sequence_window is None


def test_missing_sd_window_capability_is_rejected_before_attention(env, monkeypatch):
    monkeypatch.delattr(sys.modules["mindiesd.parallel"], "rank_window", raising=False)
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    assert not prepare(env, owner, layer, value)
    assert env.executions == 0
    assert env.agreements == 1
