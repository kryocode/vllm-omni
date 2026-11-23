# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Connector normalization is shared by vLLM 0.33 and 0.34 cache APIs."""
import pytest
from vllm.config import ModelConfig

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.distributed.omni_connectors.utils.config import connector_hash_factors

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("method", ["compute_hash", "compile_factors"])
def test_cache_api_delegates_without_mutating_connector_configuration(monkeypatch, method):
    config = object.__new__(OmniModelConfig)
    config.stage_connector_config = {"name": "SharedMemoryConnector", "extra": {"shm_name": "runtime-id"}}
    config.omni_kv_config = {"connector_config": config.stage_connector_config.copy()}
    original = config.stage_connector_config.copy()
    seen = []

    def base(normalized):
        seen.append(normalized)
        return "cache-key" if method == "compute_hash" else {"cache": "factor"}

    monkeypatch.setattr(ModelConfig, method, base, raising=False)
    result = getattr(config, method)()
    assert result == ("cache-key" if method == "compute_hash" else {"cache": "factor"})
    assert seen[0] is not config
    assert seen[0].stage_connector_config == connector_hash_factors(original)
    assert seen[0].omni_kv_config["connector_config"] == connector_hash_factors(original)
    assert config.stage_connector_config == original
    assert config.omni_kv_config["connector_config"] == original
