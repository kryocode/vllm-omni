# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public CLI, config transport and A2/A5 attention compatibility contracts."""

import pytest

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.omni_config import extract_diffusion_stage_config_kwargs
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def local_model_paths(monkeypatch):
    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)


@pytest.mark.parametrize(
    "flags,expected", [([], False), (["--cache-adaln"], True), (["--cache-adaln", "--no-cache-adaln"], False)]
)
def test_cli_enable_disable_and_budgets(flags, expected):
    parser = TrackingArgumentParser()
    OmniServeCommand().subparser_init(parser.add_subparsers(dest="subcommand"))
    args = parser.parse_args(
        [
            "serve",
            "fake-model",
            "--omni",
            "--enable-layerwise-offload",
            "--adaln-cache-max-gib",
            "12.5",
            "--adaln-cache-reserve-gib",
            "6",
            *flags,
        ]
    )
    assert args.cache_adaln is expected
    assert args.adaln_cache_max_gib == 12.5 and args.adaln_cache_reserve_gib == 6
    if flags:
        assert args.get_explicit_kwargs_dict()["cache_adaln"] is expected


@pytest.mark.parametrize(
    "sparse_type,precision,backend",
    [
        ("rf_v2", "bf16", "cann"),
        ("rf_v2", "fp16", "cann"),
        ("rf_v3", "bf16", "eagle"),
        ("rf_v3", "fp16", "eagle"),
        ("rf_v3", "mix", "eagle"),
        ("rf_v3", "mxfp8", "cann"),
    ],
)
def test_stage_config_and_a5_attention_modes_accept_the_same_cache_policy(sparse_type, precision, backend):
    kwargs = dict(
        model="fake-model",
        enable_layerwise_offload=True,
        cache_adaln=True,
        adaln_cache_max_gib=20.0,
        adaln_cache_reserve_gib=10.0,
        diffusion_attention_config={
            "default": {
                "backend": "RAINFUSION_ATTN",
                "block_sparse": {
                    "sparse_type": sparse_type,
                    "precision": precision,
                    "bsa_backend": backend,
                    "parallel": {},
                },
            }
        },
    )
    stages = StageConfigFactory.create_default_diffusion(kwargs)
    extracted = extract_diffusion_stage_config_kwargs(dict(stages[0]["engine_args"]), stage_id=0)
    config = OmniDiffusionConfig(**{k: v for k, v in extracted.items() if v is not None})
    assert config.cache_adaln and config.adaln_cache_max_gib == 20 and config.adaln_cache_reserve_gib == 10
    policy = OffloadConfig.from_od_config(config)
    assert policy.cache_adaln and policy.adaln_cache_max_gib == 20 and policy.adaln_cache_reserve_gib == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache_adaln": "true"},
        {"adaln_cache_max_gib": -1},
        {"adaln_cache_max_gib": float("nan")},
        {"adaln_cache_reserve_gib": float("inf")},
        {"adaln_cache_reserve_gib": True},
    ],
)
def test_invalid_cache_options_are_rejected_before_loading(kwargs):
    with pytest.raises((TypeError, ValueError)):
        OmniDiffusionConfig(model="fake-model", enable_layerwise_offload=True, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"enable_cpu_offload": True},
        {"enable_distributed_layerwise_offload": True},
        {"diffusion_offload_config": {"mode": "layer", "components": ["text_encoder"]}},
    ],
)
def test_cache_requires_ordinary_dit_layer_offload(kwargs):
    with pytest.raises(ValueError, match="cache_adaln requires"):
        OmniDiffusionConfig(model="fake-model", cache_adaln=True, **kwargs)


def test_structured_offload_config_accepts_cache():
    config = OmniDiffusionConfig(
        model="fake-model", cache_adaln=True, diffusion_offload_config={"mode": "layer", "components": ["dit"]}
    )
    assert OffloadConfig.from_od_config(config).cache_adaln
