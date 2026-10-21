# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in host stage ranges; these do not synchronize accelerator work."""

import functools
import inspect
import json
import logging
import os
import threading
import time
from contextlib import contextmanager, nullcontext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_api_profile = None
_capture_epoch = 0
_capture_active = False
_thread_state = threading.local()


def begin_stage_capture():
    global _capture_epoch, _capture_active
    if os.environ.get('VLLM_OMNI_H3_STAGE_PROFILING') != '1':
        return
    _capture_epoch += 1
    _capture_active = True


def end_stage_capture():
    global _capture_active
    _capture_active = False


_session_lock = threading.Lock()
_sessions = {}
_session_roots = {'MiniMaxH3Pipeline.forward', 'api/generate_video_bytes'}
_quiet_stages = {
    'api/frame_conversion', 'mp4/video_frame_encode_and_mux',
    'mp4/queue_push', 'chunk/quantize', 'chunk/copy_wait',
    'chunk/copy_enqueue', 'chunk/slot_wait', 'chunk/feed_encoder',
}


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _begin_session():
    token = object()
    with _session_lock:
        if _sessions:
            for session in _sessions.values():
                session['valid'] = False
        _sessions[token] = {'valid': not _sessions, 'rows': {}}
    return token


def _record_interval(name, start, end, tid, status):
    if name not in _quiet_stages and name != 'vae/video_streaming':
        return
    with _session_lock:
        if len(_sessions) == 1:
            session = next(iter(_sessions.values()))
            if session['valid']:
                session['rows'].setdefault(name, []).append((start, end, tid, status))


def _end_session(token):
    with _session_lock:
        session = _sessions.pop(token)
    if not session['valid']:
        return {'valid': False, 'reason': 'concurrent request scopes', 'metrics': []}
    metrics = []
    for name, rows in sorted(session['rows'].items()):
        if name not in _quiet_stages:
            continue
        intervals = _merge([(a, b) for a, b, _, _ in rows])
        metrics.append(dict(name=name, count=len(rows),
                            sum_ms=sum(b-a for a, b, _, _ in rows)/1e6,
                            union_ms=sum(b-a for a, b in intervals)/1e6,
                            threads=sorted({tid for _, _, tid, _ in rows}),
                            errors=sum(status != 'ok' for _, _, _, status in rows)))
    video = session['rows'].get('vae/video_streaming', [])
    encode = session['rows'].get('mp4/video_frame_encode_and_mux', [])
    overlap = None
    if video and encode:
        shared = _merge([(max(a,c), min(b,d)) for a,b,_,_ in video
                         for c,d,_,_ in encode if a<d and c<b])
        overlap = dict(host_video_and_frame_encode_overlap_ms=sum(b-a for a,b in shared)/1e6,
                       encoding_before_video_end=min(a for a,_,_,_ in encode)<max(b for _,b,_,_ in video))
    return dict(valid=True, metrics=metrics, streaming_overlap=overlap)


@contextmanager
def stage_span(name):
    if os.environ.get('VLLM_OMNI_H3_STAGE_PROFILING') != '1':
        yield
        return
    token = _begin_session() if name in _session_roots else None
    start_ns = time.monotonic_ns()
    pid, tid = os.getpid(), threading.get_native_id()
    quiet = name in _quiet_stages
    if not quiet:
        logger.info('H3_STAGE_START name=%s start_ns=%d pid=%d tid=%d', name, start_ns, pid, tid)
    status = 'error'
    try:
        context = nullcontext()
        if _capture_active:
            import torch
            if (threading.current_thread() is not threading.main_thread()
                    and getattr(_thread_state, 'epoch', None) != _capture_epoch):
                import torch_npu
                torch_npu.profiler.profile.enable_profiler_in_child_thread(profile_memory=True)
                _thread_state.epoch = _capture_epoch
            context = torch.profiler.record_function('H3_STAGE/' + name)
        with context:
            yield
        status = 'ok'
    finally:
        end_ns = time.monotonic_ns()
        _record_interval(name, start_ns, end_ns, tid, status)
        aggregate = _end_session(token) if token is not None else None
        if not quiet:
            suffix = ' aggregate_json=' + json.dumps(aggregate, separators=(',', ':')) if aggregate is not None else ''
            logger.info(
                'H3_STAGE_END name=%s start_ns=%d end_ns=%d host_ms=%.6f pid=%d tid=%d status=%s%s',
                name, start_ns, end_ns, (end_ns - start_ns) / 1e6, pid, tid, status, suffix,
            )


def stage(name):
    def decorate(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with stage_span(name):
                    return await func(*args, **kwargs)
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with stage_span(name):
                return func(*args, **kwargs)
        return wrapper
    return decorate


def start_api_profile():
    global _api_profile
    path = os.environ.get('VLLM_OMNI_H3_API_PROFILE_DIR')
    if not path:
        return
    if _api_profile is not None:
        raise RuntimeError('API profiler already active')
    import torch_npu

    os.makedirs(path, exist_ok=True)
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path, analyse_flag=False),
    )
    profile.start()
    _api_profile = profile
    begin_stage_capture()
    logger.info('H3_API_PROFILE_STARTED pid=%d path=%s', os.getpid(), path)


def stop_api_profile():
    global _api_profile
    if _api_profile is None:
        return
    profile, _api_profile = _api_profile, None
    try:
        profile.stop()
    finally:
        end_stage_capture()
    logger.info('H3_API_PROFILE_STOPPED pid=%d', os.getpid())
