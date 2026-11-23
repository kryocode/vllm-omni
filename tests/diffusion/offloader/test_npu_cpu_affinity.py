# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offload locality respects visible devices, explicit overrides and CPU limits."""

import importlib
import json
import multiprocessing
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.offloader import npu_cpu_affinity as affinity

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def config(**options):
    return SimpleNamespace(
        enable_layerwise_offload=True,
        cache_adaln=True,
        additional_config=options,
        parallel_config=SimpleNamespace(
            ulysses_degree=8, allgather_degree=2, ring_degree=1, tensor_parallel_size=1, world_size=16
        ),
        diffusion_attention_config=SimpleNamespace(
            default=SimpleNamespace(
                backend="RAINFUSION_ATTN", block_sparse=SimpleNamespace(parallel=SimpleNamespace())
            ),
            per_role={},
        ),
    )


@pytest.fixture
def runtime(monkeypatch):
    current = {"cpus": {0, 1, 2, 3}}
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda pid: set(current["cpus"]))
    monkeypatch.setattr(affinity.os, "sched_setaffinity", lambda pid, cpus: current.update(cpus=set(cpus)))
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {2, 3, 99}))
    monkeypatch.setattr(affinity, "_worker_threads", lambda: [0], raising=False)
    return current


def test_auto_binding_intersects_incoming_cpu_limits(runtime):
    assert affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {2, 3}


def test_worker_start_affinity_restores_parent_after_each_device(runtime, monkeypatch):
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (index, index, {index}))
    for index in (0, 1):
        with affinity.npu_worker_start_affinity(config(), index, is_npu=True):
            assert runtime["cpus"] == {index}
        assert runtime["cpus"] == {0, 1, 2, 3}


def test_worker_start_failure_restores_parent(runtime):
    with pytest.raises(RuntimeError, match="spawn failed"):
        with affinity.npu_worker_start_affinity(config(), 0, is_npu=True):
            assert runtime["cpus"] == {2, 3}
            raise RuntimeError("spawn failed")
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_executor_start_uses_local_cpus_and_restores_on_failure(runtime, monkeypatch):
    from unittest.mock import MagicMock

    from vllm_omni.diffusion.executor import multiproc_executor

    cfg = config()
    cfg.num_gpus = 16
    cfg.worker_extension_cls = None
    context = MagicMock()
    context.Pipe.return_value = (MagicMock(), MagicMock())
    process = context.Process.return_value

    def failed_start():
        assert runtime["cpus"] == {2, 3}
        raise RuntimeError("executor spawn failed")

    process.start.side_effect = failed_start
    monkeypatch.setattr(multiproc_executor.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(multiproc_executor, "set_multiprocessing_worker_envs", lambda **kwargs: None)
    monkeypatch.setattr(multiproc_executor.mp, "set_start_method", lambda *args, **kwargs: None)
    executor = object.__new__(multiproc_executor.MultiprocDiffusionExecutor)
    executor.od_config = cfg
    executor._mp_context = context
    with pytest.raises(RuntimeError, match="executor spawn failed"):
        executor._launch_workers(MagicMock(), [MagicMock()] * 16)
    assert runtime["cpus"] == {0, 1, 2, 3}


@pytest.mark.parametrize("npu,options", [(False, {}), (True, {"enable_cpu_binding": False})])
def test_worker_start_disabled_does_not_query_topology(runtime, monkeypatch, npu, options):
    def unexpected(index):
        raise AssertionError("disabled start queried topology")

    monkeypatch.setattr(affinity, "_npu_affinity", unexpected)
    with affinity.npu_worker_start_affinity(config(**options), 0, is_npu=npu):
        assert runtime["cpus"] == {0, 1, 2, 3}


def test_worker_start_auto_topology_failure_preserves_parent(runtime, monkeypatch):
    def unavailable(index):
        raise RuntimeError("topology unavailable")

    monkeypatch.setattr(affinity, "_npu_affinity", unavailable)
    with affinity.npu_worker_start_affinity(config(), 0, is_npu=True):
        assert runtime["cpus"] == {0, 1, 2, 3}
    with pytest.raises(RuntimeError, match="requested worker start"):
        with affinity.npu_worker_start_affinity(config(enable_cpu_binding=True), 0, is_npu=True):
            pytest.fail("explicit binding failure started a worker")


def test_worker_start_affinity_is_inherited_before_imports(monkeypatch):
    before = os.sched_getaffinity(0)
    local = {min(before)}
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (index, 0, local))
    with affinity.npu_worker_start_affinity(config(), 0, is_npu=True):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,os,sys; print(json.dumps([sorted(os.sched_getaffinity(0)), 'torch' in sys.modules]))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(result.stdout) == [sorted(local), False]
    assert os.sched_getaffinity(0) == before


def test_real_spawn_inherits_start_affinity(monkeypatch, tmp_path):
    # The probe module itself records affinity before any Torch imports. This
    # exercises the same spawn context used by the real diffusion executor.
    probe = tmp_path / "affinity_spawn_probe.py"
    probe.write_text(
        "import os\n"
        "IMPORT_AFFINITY = sorted(os.sched_getaffinity(0))\n"
        "def report(pipe):\n"
        "    pipe.send(IMPORT_AFFINITY)\n"
        "    pipe.close()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("affinity_spawn_probe")
    before = os.sched_getaffinity(0)
    local = {min(before)}
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (index, 0, local))
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=module.report, args=(writer,))
    try:
        with affinity.npu_worker_start_affinity(config(), 0, is_npu=True):
            process.start()
        writer.close()
        assert os.sched_getaffinity(0) == before
        assert reader.poll(20)
        assert reader.recv() == sorted(local)
        process.join(20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        reader.close()
        writer.close()
        sys.modules.pop("affinity_spawn_probe", None)


def test_explicit_false_skips_topology_and_preserves_affinity(runtime, monkeypatch):
    def unexpected(index):
        raise AssertionError("disabled binding queried topology")

    monkeypatch.setattr(affinity, "_npu_affinity", unexpected)
    assert not affinity.maybe_bind_npu_offload_cpu(
        config(enable_cpu_binding=False), SimpleNamespace(type="npu", index=0)
    )
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_existing_narrow_binding_is_preserved(runtime):
    runtime["cpus"] = {3}
    assert affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {3}


def test_existing_thread_pools_are_bound_without_widening(monkeypatch):
    masks = {0: {0, 1, 2, 3}, 7: {0, 1, 2, 3}, 9: {3}}
    monkeypatch.setattr(affinity, "_worker_threads", lambda: [0, 7, 9], raising=False)
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda tid: set(masks[tid]))
    monkeypatch.setattr(affinity.os, "sched_setaffinity", lambda tid, cpus: masks.update({tid: set(cpus)}))
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {2, 3, 99}))
    assert affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert masks == {0: {2, 3}, 7: {2, 3}, 9: {3}}


def test_partial_thread_binding_failure_restores_earlier_threads(monkeypatch):
    masks = {0: {0, 1, 2, 3}, 7: {0, 1, 2, 3}}
    monkeypatch.setattr(affinity, "_worker_threads", lambda: [0, 7], raising=False)
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda tid: set(masks[tid]))

    def bind(tid, cpus):
        if tid == 7:
            raise PermissionError("thread affinity denied")
        masks[tid] = set(cpus)

    monkeypatch.setattr(affinity.os, "sched_setaffinity", bind)
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {2, 3, 99}))
    assert not affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert masks == {0: {0, 1, 2, 3}, 7: {0, 1, 2, 3}}


def test_no_allowed_local_cpu_does_not_widen_limits(runtime, monkeypatch):
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {99}))
    assert not affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_topology_unavailable_falls_back_for_auto(runtime, monkeypatch):
    def unavailable(index):
        raise OSError("driver unavailable")

    monkeypatch.setattr(affinity, "_npu_affinity", unavailable)
    assert not affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_affinity_permission_failure_keeps_auto_loading_available(runtime, monkeypatch):
    def denied(pid, cpus):
        raise PermissionError("affinity denied")

    monkeypatch.setattr(affinity.os, "sched_setaffinity", denied)
    assert not affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_loader_binds_before_entering_weight_loading(monkeypatch):
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    calls = []
    loader = object.__new__(DiffusersPipelineLoader)
    loader.od_config, loader.quant_config = config(), None

    def bind(cfg, device):
        calls.append("bind")
        return True

    def enter_loading():
        assert calls == ["bind"]
        raise RuntimeError("weight-load sentinel")

    monkeypatch.setattr(affinity, "maybe_bind_npu_offload_cpu", bind)
    monkeypatch.setattr(loader, "_get_shared_weight_group", enter_loading)
    with pytest.raises(RuntimeError, match="weight-load sentinel"):
        loader.load_model("cpu", device=SimpleNamespace(type="npu", index=0))
    assert loader.cpu_affinity_bound


def test_worker_binds_before_device_runtime_initialization(monkeypatch):
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, current_omni_platform

    worker = object.__new__(DiffusionWorker)
    worker.od_config = config()
    worker.od_config.num_gpus = 16
    worker.rank = worker.local_rank = 0
    worker.distributed_init_method = "env://"
    calls = []

    def bind(cfg, device):
        calls.append("bind")
        return True

    monkeypatch.setattr(affinity, "maybe_bind_npu_offload_cpu", bind)
    monkeypatch.setattr(current_omni_platform, "get_torch_device", lambda rank: SimpleNamespace(type="npu", index=0))

    def initialize(device):
        assert calls == ["bind"]
        raise RuntimeError("device-init sentinel")

    monkeypatch.setattr(current_omni_platform, "set_device", initialize)
    with pytest.raises(RuntimeError, match="device-init sentinel"):
        worker.init_device()


def test_explicit_binding_reports_failure(runtime, monkeypatch):
    monkeypatch.setattr(affinity, "_npu_affinity", lambda index: (8, 1, {99}))
    with pytest.raises(RuntimeError, match="CPU binding"):
        affinity.maybe_bind_npu_offload_cpu(config(enable_cpu_binding=True), SimpleNamespace(type="npu", index=0))


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_other_devices_do_not_bind(runtime, device_type):
    assert not affinity.maybe_bind_npu_offload_cpu(config(), SimpleNamespace(type=device_type, index=0))
    assert runtime["cpus"] == {0, 1, 2, 3}


@pytest.mark.parametrize("lo,cache", [(False, False), (False, True), (True, False)])
def test_auto_requires_layerwise_and_cache(runtime, lo, cache):
    cfg = config()
    cfg.enable_layerwise_offload, cfg.cache_adaln = lo, cache
    assert not affinity.maybe_bind_npu_offload_cpu(cfg, SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {0, 1, 2, 3}


@pytest.mark.parametrize("native,u,cp,tp", [(True, 8, 2, 1), (False, 8, 1, 1), (False, 4, 4, 1), (False, 8, 1, 2)])
def test_auto_is_bound_to_sd16_u8_cp2(runtime, native, u, cp, tp):
    cfg = config()
    cfg.parallel_config.ulysses_degree = u
    cfg.parallel_config.allgather_degree = cp
    cfg.parallel_config.tensor_parallel_size = tp
    cfg.parallel_config.world_size = u * cp * tp
    if native:
        cfg.diffusion_attention_config.default.block_sparse.parallel = None
    assert not affinity.maybe_bind_npu_offload_cpu(cfg, SimpleNamespace(type="npu", index=0))
    assert runtime["cpus"] == {0, 1, 2, 3}


def test_sd16_per_role_attention_also_activates_auto(runtime):
    cfg = config()
    cfg.diffusion_attention_config.per_role["self"] = cfg.diffusion_attention_config.default
    cfg.diffusion_attention_config.default = None
    assert affinity.maybe_bind_npu_offload_cpu(cfg, SimpleNamespace(type="npu", index=0))


def test_string_override_is_rejected(runtime):
    with pytest.raises(ValueError, match="boolean"):
        affinity.maybe_bind_npu_offload_cpu(config(enable_cpu_binding="false"), SimpleNamespace(type="npu", index=0))


def test_visible_physical_device_order(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "12,8,15,9")
    assert affinity._visible_physical_device(1) == 8
    assert affinity._visible_physical_device(3) == 9
    with pytest.raises(ValueError):
        affinity._visible_physical_device(4)


@pytest.mark.parametrize("text", ["8,,9", "-1", "8,8"])
def test_bad_visibility_cannot_target_another_device(monkeypatch, text):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", text)
    with pytest.raises(ValueError):
        affinity._visible_physical_device(0)


def test_cpu_ranges_and_pcie_abi():
    assert affinity._cpu_list("0-2,7,120-121") == {0, 1, 2, 7, 120, 121}
    assert affinity.ctypes.sizeof(affinity._PcieInfo) == 64


@pytest.mark.parametrize("enabled", [False, True])
def test_public_additional_config_reaches_diffusion_stage(monkeypatch, enabled):
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.omni_config import extract_diffusion_stage_config_kwargs
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, revision: model)
    parser = TrackingArgumentParser()
    OmniServeCommand().subparser_init(parser.add_subparsers(dest="subcommand"))
    args = parser.parse_args(
        [
            "serve",
            "fake-model",
            "--omni",
            "--enable-layerwise-offload",
            "--cache-adaln",
            "--additional-config",
            json.dumps({"enable_cpu_binding": enabled}),
        ]
    )
    stages = StageConfigFactory.create_default_diffusion(
        {
            "model": "fake-model",
            "enable_layerwise_offload": True,
            "cache_adaln": True,
            "additional_config": args.additional_config,
        }
    )
    fields = extract_diffusion_stage_config_kwargs(dict(stages[0]["engine_args"]), stage_id=0)
    cfg = OmniDiffusionConfig(**{k: v for k, v in fields.items() if v is not None})
    assert cfg.additional_config["enable_cpu_binding"] is enabled
