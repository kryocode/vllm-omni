# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 prefix hits must replay text conditioning and its token-role labels."""
import pytest
import torch

from vllm_omni.core.prefix_cache.interface import StageCacheOutputs
from vllm_omni.model_executor.models.minimax_h3.encoder import MiniMaxH3Encoder
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.prefix_cache import MRv2PrefixCacheMixin, PrefixCacheRewrite

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_h3_enables_multimodal_prefix_output_replay():
    runner = MRv2PrefixCacheMixin()
    runner._snapshot_prefix_cache_model_policy(MiniMaxH3Encoder)
    assert not runner._omni_cache_policy.native_kv_cache_only
    # The named hidden_states.output payload is cached; duplicating it as
    # generic "hidden" would waste storage and add the payload twice.
    assert not runner._omni_cache_policy.needs_full_hidden_states


def test_text_prefix_and_role_labels_replace_the_scheduled_tail():
    full = torch.arange(32).reshape(8, 4).to(torch.bfloat16)
    tags = torch.tensor([[1], [1], [0], [0], [1], [0], [1], [1]])
    tail = {"hidden_states.output": full[6:].clone(), "meta.token_role_ids": tags[6:].clone()}

    class Cache:
        def materialize(self, step_id, req_ids, *, include_uncached):
            assert (step_id, req_ids, include_uncached) == (1, ["request"], False)
            return StageCacheOutputs(hidden_states=None, mm_outputs={
                "hidden_states.output": {"request": full},
                "meta.token_role_ids": {"request": tags},
            })

    output = OmniModelRunnerOutput._from_components(
        req_ids=["request"], sampled_token_ids=[[]],
        inter_stage_outputs=[tail], pooler_output=[tail], multimodal_outputs=[tail],
    )
    PrefixCacheRewrite(Cache(), 1, ((0, "request"),)).apply(output)
    assert torch.equal(tail["hidden_states.output"], full)
    assert torch.equal(tail["meta.token_role_ids"], tags)
    assert output.full_payload_prefix_cache_keys["request"] == frozenset(tail)
