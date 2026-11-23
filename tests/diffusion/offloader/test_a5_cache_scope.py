# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unvalidated AdaLN weight residency remains disabled on A5."""
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.offloader import adaln_cache

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_a5_rejects_cache_before_tensor_selection_or_allocation(monkeypatch):
    from vllm_omni.platforms import npu

    monkeypatch.setattr(npu, "is_a5", lambda device: True)

    def unexpected(*args, **kwargs):
        raise AssertionError("A5 must not select or allocate cache tensors")

    monkeypatch.setattr(adaln_cache, "select_adaln_tensors", unexpected)
    cache = adaln_cache.AdaLNWeightCache(SimpleNamespace(type="npu", index=0))
    assert cache.prepare([], max_gib=25, reserve_gib=8) == frozenset()
    assert not cache.status.enabled and cache.status.reason == "unsupported_on_a5"
    assert not cache.resident_ids


@pytest.mark.parametrize("kind", ["cpu", "npu"])
def test_other_platform_cache_admission_is_preserved(monkeypatch, kind):
    from vllm_omni.platforms import npu

    monkeypatch.setattr(npu, "is_a5", lambda device: False)
    cache = adaln_cache.AdaLNWeightCache(SimpleNamespace(type=kind, index=0))
    assert cache.prepare([], max_gib=25, reserve_gib=8) == frozenset()
    assert cache.status.reason == "no_adaln_tensors"
