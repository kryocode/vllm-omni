# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Legacy request construction and explicit frontend hash transport."""
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def core_request():
    return OmniEngineCoreRequest.from_request(EngineCoreRequest(
        request_id="compat-request", prompt_token_ids=[1, 2], mm_features=None,
        sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
        arrival_time=0.0, lora_request=None, cache_salt=None, data_parallel_rank=None,
    ))


def test_legacy_request_without_frontend_hashes_constructs():
    result = OmniRequest.from_engine_core_request(core_request(), block_hasher=None)
    assert result.request_id == "compat-request" and result.prompt_token_ids == [1, 2]
    assert result.block_hashes == []


@pytest.mark.parametrize("hashes", [None, [], ["frontend-block"]])
def test_only_explicit_frontend_hashes_are_forwarded(hashes):
    class RecordingRequest(OmniRequest):
        def __init__(self, **kwargs):
            self.received = kwargs

    result = RecordingRequest.from_engine_core_request(core_request(), None, hashes)
    if hashes is None:
        assert "block_hashes" not in result.received
    else:
        assert result.received["block_hashes"] is hashes
