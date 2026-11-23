# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Central output construction retains the old request index when needed."""
import pytest

from vllm_omni.outputs import ModelRunnerOutput, OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_connector_only_output_constructs_on_legacy_runtime():
    token = object()
    output = OmniModelRunnerOutput.with_kv_conn_output_only(token)
    assert output.req_ids == [] and output.kv_connector_output is token
    assert output.req_id_to_index == {}


def test_legacy_index_tracks_output_order():
    output = OmniModelRunnerOutput._from_components(req_ids=["second", "first"], sampled_token_ids=[[], []])
    assert output.req_id_to_index == {"second": 0, "first": 1}


def test_new_output_layout_does_not_receive_legacy_field(monkeypatch):
    fields = {k: v for k, v in ModelRunnerOutput.__dataclass_fields__.items() if k != "req_id_to_index"}
    monkeypatch.setattr(ModelRunnerOutput, "__dataclass_fields__", fields)

    class RecordingOutput(OmniModelRunnerOutput):
        def __init__(self, **kwargs):
            self.received = kwargs

    output = RecordingOutput._from_components(req_ids=["a"], sampled_token_ids=[[]])
    assert "req_id_to_index" not in output.received
