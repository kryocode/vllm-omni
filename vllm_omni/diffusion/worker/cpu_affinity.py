# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU-only affinity for the MiniMax-H3 experiment workers.

Bind before device initialization and refresh after model loading. Linux CPU
affinity belongs to each thread; pinning only TID0 leaves existing runtime
threads unrestricted. No NUMA memory policy or torch thread-count changes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

ENABLE = 'VLLM_OMNI_H3_CPU_AFFINITY'
PLAN = 'VLLM_OMNI_H3_CPU_AFFINITY_FILE'
RECORDS = 'VLLM_OMNI_H3_CPU_AFFINITY_RECORD_DIR'


def parse_cpus(value: str) -> set[int]:
    cpus = set()
    for item in value.strip().split(','):
        if not item:
            raise ValueError('Empty CPU range')
        pair = item.split('-')
        if len(pair) == 1:
            lo = hi = int(pair[0])
        elif len(pair) == 2:
            lo, hi = map(int, pair)
        else:
            raise ValueError('Invalid CPU range')
        if lo < 0 or hi < lo:
            raise ValueError('Invalid CPU bounds')
        cpus.update(range(lo, hi+1))
    return cpus


def _thread_masks() -> dict[int, list[int]]:
    result = {}
    for task in Path('/proc/self/task').iterdir():
        try:
            result[int(task.name)] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            pass
    return result


def load_plan(local_rank: int, filename: str) -> tuple[dict, dict]:
    plan = json.loads(Path(filename).read_text())
    visible = [int(v) for v in os.environ.get('ASCEND_RT_VISIBLE_DEVICES','').split(',') if v]
    if visible != plan['visible_devices']:
        raise ValueError('Visible NPU order differs from frozen topology plan')
    if type(local_rank) is not int or not 0 <= local_rank < len(visible):
        raise ValueError('Invalid local rank')
    ranks = plan['ranks']
    if len(ranks) != len(visible):
        raise ValueError('Plan must cover every visible NPU')
    seen = set()
    online = parse_cpus(Path('/sys/devices/system/cpu/online').read_text())
    for index, entry in enumerate(ranks):
        cpus = entry['cpus']
        if entry['local_rank'] != index or entry['physical_npu'] != visible[index]:
            raise ValueError('Rank/device mapping differs')
        if not cpus or any(type(c) is not int or c < 0 for c in cpus) or cpus != sorted(set(cpus)):
            raise ValueError('CPU list must be nonempty, unique, sorted integers')
        selected = set(cpus)
        if seen & selected or not selected <= online:
            raise ValueError('Overlapping ranks or offline CPU')
        node_cpus = parse_cpus(Path(f"/sys/devices/system/node/node{entry['numa_node']}/cpulist").read_text())
        if not selected <= node_cpus or not selected <= parse_cpus(entry['npu_cpu_affinity']):
            raise ValueError('Selected CPUs do not match the local NPU/NUMA node')
        seen.update(selected)
    return plan, ranks[local_rank]


def bind_worker_cpu(local_rank: int, rank: int, phase: str) -> dict:
    flag = os.environ.get(ENABLE, '0')
    if flag not in ('0', '1'):
        raise ValueError(f'{ENABLE} must be 0 or 1')
    enabled = flag == '1'
    filename = os.environ.get(PLAN)
    records = os.environ.get(RECORDS)
    if not enabled and not records:
        return {'enabled': False}
    if not filename:
        raise ValueError('An explicit topology plan is required')
    plan, entry = load_plan(local_rank, filename)
    before = sorted(os.sched_getaffinity(0))
    target = set(entry['cpus'])
    if enabled and not target <= set(before):
        raise ValueError('Plan requests CPUs outside current process/cgroup allowance')
    masks = _thread_masks()
    before_masks = dict(masks)
    if enabled:
        # Constrain the caller first, so its newly spawned threads inherit this
        # mask while pre-existing native threads are brought into the same set.
        os.sched_setaffinity(0, target)
        for _ in range(4):
            for tid in masks:
                try:
                    os.sched_setaffinity(tid, target)
                except ProcessLookupError:
                    pass
            masks = _thread_masks()
            if masks and all(set(mask) == target for mask in masks.values()):
                break
        else:
            raise RuntimeError('Could not bind every live worker thread')
    value = dict(enabled=enabled,local_rank=local_rank,rank=rank,pid=os.getpid(),phase=phase,
                 epoch=time.time(),physical_npu=entry['physical_npu'],numa_node=entry['numa_node'],
                 planned_cpus=entry['cpus'],before_main_cpus=before,before_thread_masks=before_masks,
                 thread_masks=masks,all_threads_match_target=all(set(mask)==target for mask in masks.values()),
                 plan_sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
                 omp_num_threads=os.environ.get('OMP_NUM_THREADS'),mkl_num_threads=os.environ.get('MKL_NUM_THREADS'),
                 memory_policy_changed=False)
    if records:
        directory=Path(records);directory.mkdir(parents=True,exist_ok=True)
        file=directory/f'rank{rank}_{phase}_{os.getpid()}.json'
        temp=file.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(file)
    return value
