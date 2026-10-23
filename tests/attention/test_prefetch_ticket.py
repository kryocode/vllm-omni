# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host control-flow contracts; these tests do not prove NPU correctness."""

import ast
import importlib.util
import sys
import types
from contextlib import nullcontext
from itertools import chain
from pathlib import Path

import pytest
import torch
from test_mindiesd_bridge import attention, env as env, metadata, prepare, tensor  # noqa: F401

ROOT = Path(__file__).resolve().parents[2] / 'vllm_omni/diffusion'
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def ticket_module():
    spec = importlib.util.spec_from_file_location('contract_prefetch_ticket', ROOT / 'offloader/prefetch_ticket.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_ticket(module, *, defer=True, launch=None, generation=1, request=1, consumer=2):
    calls = []
    policy = module.PrefetchGatePolicy(defer, request, consumer, 'test')
    ticket = module.PrefetchTicket(
        policy,
        owner_hook=object(),
        slot=0,
        slot_generation=generation,
        launch_ag=launch or (lambda event: calls.append(event)),
        ready_event=object(),
    )
    ticket.h2d_submitted()
    return ticket, calls


def test_deferred_exact_once_and_scope(ticket_module):
    ticket, calls = make_ticket(ticket_module)
    assert ticket.state == 'ARMED'
    with ticket.scope():
        ticket.release_after_input(types.SimpleNamespace(index=0), event := object())
    assert calls == [event] and ticket.state == 'RELEASED'
    assert ticket.launch_attempts == 1
    with pytest.raises(RuntimeError):
        ticket.release_after_input(types.SimpleNamespace(index=0), event)
    assert calls == [event]


def test_event_release_is_independent_of_algorithm_chunk(ticket_module):
    ticket, calls = make_ticket(ticket_module)
    event = object()
    with ticket.scope():
        ticket.release_after_event(event)
    assert calls == [event] and ticket.launch_attempts == 1
    with pytest.raises(RuntimeError):
        ticket.release_after_event(event)


def test_request_receipt_reports_unreleased_ticket_failure(env, ticket_module):
    owner, layer = env.h3.MindIESDH3Parallel(), attention()
    counters = {}
    hook = types.SimpleNamespace(execution_snapshot=lambda: {
        'prefetch': dict(counters), 'failed': bool(counters.get('failed')), 'nz': None,
    })
    component = types.SimpleNamespace(
        attention=layer, num_heads=8, head_dim=4,
        _hook_registry=types.SimpleNamespace(get_hook=lambda name: hook),
    )
    model = types.SimpleNamespace(
        _mindiesd_parallel=owner, _rope_local_span=lambda seq: (5, 5),
        modules=lambda: (component,),
    )
    branch = types.SimpleNamespace(seq_len=10, used_len=7, device=tensor().device, static_kwargs={})
    with env.h3.prepare_h3_request(model, branch):
        policy = ticket_module.PrefetchGatePolicy(True, (id(owner), owner._request_id), id(layer), 'test')
        pending = ticket_module.PrefetchTicket(
            policy, owner_hook=object(), slot=0, slot_generation=1,
            launch_ag=lambda event: None, ready_event=object(), counters=counters,
        )
        pending.h2d_submitted()
        owner.register_prefetch_ticket(pending)
    assert pending.state == 'FAILED'
    assert owner.last_execution_receipt['outcome'] == 'failed'
    assert not owner.last_execution_receipt['executed']
    assert owner.last_execution_receipt['execution_status'] == 'partial_failed'
    assert owner.last_execution_receipt['request_delta']['prefetch']['failed'] == 1
    assert owner.last_execution_receipt['cumulative']['offload']['prefetch']['live'] == 0


def test_offload_snapshot_copies_existing_counts_and_hides_plan_storage():
    _, cls = offload_methods()
    hook = cls.__new__(cls)
    hook.prefetch_counters = {'released_after_input': 3, 'live': 1}
    hook._ticket_failed = False
    hook._qbmm_nz_plan = types.SimpleNamespace(stats=lambda: {'bind_count': 9})
    snapshot = hook.execution_snapshot()
    hook.prefetch_counters['live'] = 0
    assert snapshot['prefetch']['live'] == 1 and snapshot['nz']['bind_count'] == 9
    assert snapshot['failed'] is False


def test_immediate_no_callback_and_no_rescue(ticket_module):
    ticket, calls = make_ticket(ticket_module, defer=False)
    assert not calls
    ticket.release_ungated('unsupported')
    ticket.finish()
    assert calls == [None]
    deferred, calls = make_ticket(ticket_module)
    with pytest.raises(RuntimeError):
        deferred.release_ungated('rescue')
    assert calls == []


@pytest.mark.parametrize('index,event', [(1, object()), (0, None)])
def test_invalid_input_event_never_launches(ticket_module, index, event):
    ticket, calls = make_ticket(ticket_module)
    with pytest.raises(RuntimeError):
        ticket.release_after_input(types.SimpleNamespace(index=index), event)
    assert calls == []


def test_partial_launch_failure_poison_and_original_error(ticket_module):
    error = ValueError('dtype two failed after dtype one submitted')
    calls = []

    def launch(event):
        calls.append(event)
        raise error

    ticket, _ = make_ticket(ticket_module, launch=launch)
    with pytest.raises(ValueError) as caught:
        with ticket.scope():
            ticket.release_after_input(types.SimpleNamespace(index=0), object())
    assert caught.value is error and ticket.state == 'FAILED'
    assert ticket.launch_attempts == 1 and len(calls) == 1
    with pytest.raises(RuntimeError):
        ticket.release_ungated('rescue')
    assert len(calls) == 1


def test_missing_callback_scope_and_prebody_abort(ticket_module):
    ticket, calls = make_ticket(ticket_module)
    with pytest.raises(RuntimeError):
        with ticket.scope():
            pass
    assert ticket.state == 'FAILED' and not calls
    other, calls = make_ticket(ticket_module)
    other.abort(RuntimeError('prebody exception'))
    assert other.state == 'FAILED' and not calls


def test_identity_and_slot_generation_cannot_cross(ticket_module):
    ticket, calls = make_ticket(ticket_module)
    ticket.validate(request_id=1, consumer_id=2)
    for request, consumer in [(2, 2), (1, 3)]:
        with pytest.raises(RuntimeError):
            ticket.validate(request_id=request, consumer_id=consumer)
    with pytest.raises(RuntimeError):
        ticket.assert_ready(slot_generation=2)
    ticket.release_after_input(types.SimpleNamespace(index=0), object())
    ticket.assert_ready(slot_generation=1)
    with pytest.raises(RuntimeError):
        ticket.assert_ready(slot_generation=2)
    assert len(calls) == 1


def test_released_deferred_ticket_cannot_validate_for_another_attention(ticket_module):
    ticket, calls = make_ticket(ticket_module)
    ticket.release_after_input(types.SimpleNamespace(index=0), object())
    with pytest.raises(RuntimeError):
        ticket.validate(request_id=1, consumer_id=2)
    assert len(calls) == 1


def test_completed_ticket_scope_cannot_be_entered_again(ticket_module):
    ticket, calls = make_ticket(ticket_module, defer=False)
    ticket.release_ungated('immediate')
    with ticket.scope():
        pass
    with pytest.raises(RuntimeError):
        with ticket.scope():
            pass
    assert calls == [None]


def test_async_drain_failure_suppresses_collective_weight_restore():
    tree = ast.parse((ROOT / 'offloader/distributed_layerwise_backend.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DistributedLayerwiseOffloadBackend')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_disable')
    failure = RuntimeError('asynchronous HCCL failure during drain')
    restores = []

    def synchronize():
        raise failure

    def cleanup(steps):
        first = None
        for _, action in steps:
            try:
                action()
            except BaseException as error:
                if first is None:
                    first = error
        return first

    namespace = {
        'chain': chain,
        'run_cleanup_steps': cleanup,
        'current_omni_platform': types.SimpleNamespace(synchronize=synchronize),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), '<production-backend-disable>', 'exec'), namespace)
    ticket = types.SimpleNamespace(state='RELEASED')
    hook = types.SimpleNamespace(
        _slot_tickets=[ticket, None],
        _invocation_ticket=None,
        _ticket_failed=False,
        next_block=object(),
        dp_size=2,
        restore_next_block_to_cpu=lambda: restores.append('AG'),
    )
    hooks = [[hook]]
    backend = types.SimpleNamespace(
        _host_weight_lease=None,
        _host_registration=None,
        host_weight_plan=None,
        _blocks=[],
        _all_hook_groups=hooks,
        _encoder_modules=[],
        _staged_components=[],
        _resident_layer_group=None,
        _residency_pipeline_ref=None,
        enabled=True,
        _clear_residency_controller=lambda: None,
        _release_registered_mmap=lambda: None,
    )
    with pytest.raises(RuntimeError) as caught:
        namespace['_disable'](backend, restore_allgather_weights=True)
    assert caught.value is failure
    assert restores == [] and backend._all_hook_groups is hooks
    assert backend._poisoned_reason


def offload_methods():
    tree = ast.parse((ROOT / 'offloader/distributed_layerwise_backend.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DistributedLayerwiseOffloadHook')
    for method in cls.body:
        if isinstance(method, ast.FunctionDef):
            method.decorator_list = []
    namespace = {
        '__name__': 'offloader_contract',
        'torch': torch,
        'nn': torch.nn,
        'Any': object,
        'ModelHook': object,
        'set_tensor_storage': lambda *a: None,
    }
    selected = ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(selected), '<production-offloader-methods>', 'exec'), namespace)
    return namespace, namespace['DistributedLayerwiseOffloadHook']


def test_production_wiring_no_tls_or_classname():
    offload = (ROOT / 'offloader/distributed_layerwise_backend.py').read_text()
    assert '_first_head_gate' not in offload
    assert 'MiniMaxH3DiTBlock' not in offload
    tree = ast.parse((ROOT / 'models/minimax_h3/minimax_h3_transformer.py').read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    for name, methods in {
        'MiniMaxH3DiTBlock': ['forward', 'attention_prefetch_policy'],
        'MiniMaxH3Attention': ['forward', '_run_packed_attention'],
    }.items():
        for method in methods:
            fn = next(n for n in classes[name].body if isinstance(n, ast.FunctionDef) and n.name == method)
            if method != 'attention_prefetch_policy':
                assert 'attention_prefetch_ticket' in [a.arg for a in fn.args.kwonlyargs]
    block = ast.unparse(classes['MiniMaxH3DiTBlock'])
    assert '.scope()' in block and 'attention_prefetch_ticket=attention_prefetch_ticket' in block
    bridge = ast.parse((ROOT / 'attention/mindiesd_bridge.py').read_text())
    run = next(n for n in ast.walk(bridge) if isinstance(n, ast.FunctionDef) and n.name == 'run')
    assert 'on_input_chunk_ready' in [a.arg for a in run.args.kwonlyargs]
    assert any(k.arg == 'on_input_chunk_ready' for n in ast.walk(run) if isinstance(n, ast.Call) for k in n.keywords)


def test_wait_unreleased_ticket_never_waits_unrecorded_event(ticket_module):
    namespace, cls = offload_methods()
    waits = []
    namespace['current_omni_platform'] = types.SimpleNamespace(
        current_stream=lambda: types.SimpleNamespace(wait_event=waits.append)
    )
    hook = cls.__new__(cls)
    ticket, _ = make_ticket(ticket_module)
    hook._slot_tickets = [ticket, None]
    hook._slot_generations = [1, 0]
    hook.ready_events = [None, None]
    hook._prev_hook = None
    with pytest.raises(RuntimeError):
        hook._wait_for_weights(0)
    assert waits == []


def test_pre_forward_protocol_preserves_early_prefetch_without_kwarg_mutation(ticket_module):
    namespace, cls = offload_methods()
    namespace['PrefetchGatePolicy'] = ticket_module.PrefetchGatePolicy
    hook = cls.__new__(cls)
    hook.current_slot = 0
    hook._prev_hook = None
    hook._is_group_first = False
    hook._invocation_ticket = None
    hook._ticket_failed = False
    ticket, _ = make_ticket(ticket_module, defer=False)
    ticket.release_ungated('immediate')
    calls = []

    def prefetch(slot, **kwargs):
        calls.append((slot, kwargs))
        return ticket

    hook.prefetch_layer = prefetch
    module = types.SimpleNamespace(attention_prefetch_policy=lambda: ticket.policy)
    args, kwargs = hook.pre_forward(module, 'input', original=1)
    assert calls[0][0] == 1 and calls[0][1]['policy'] is ticket.policy
    assert kwargs['attention_prefetch_ticket'] is ticket and kwargs['original'] == 1
    assert args == ('input',)


def test_owner_policy_callback_filters_chunk_zero_and_request_finally_aborts(env, ticket_module, monkeypatch):  # noqa: F811
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    monkeypatch.setenv('MINDIESD_FIRST_HEAD_AG_GATE', '1')
    assert prepare(env, owner, layer, value)
    capability_calls = []

    def supports(**kwargs):
        capability_calls.append(kwargs)
        return True

    owner._bridge.executor.supports_input_chunk_events = supports
    policy = owner.prefetch_policy(layer)
    assert policy.defer and capability_calls == [{'device': value.device, 'dtype': torch.bfloat16, 'world': 2}]
    calls = []
    ticket = ticket_module.PrefetchTicket(
        policy, owner_hook=object(), slot=0, slot_generation=1, launch_ag=calls.append, ready_event=object()
    )
    ticket.h2d_submitted()
    owner.register_prefetch_ticket(ticket)

    def run(q, k, v, **kwargs):
        callback = kwargs['on_input_chunk_ready']
        callback(types.SimpleNamespace(index=0), event := object())
        callback(types.SimpleNamespace(index=1), object())
        assert calls == [event]
        return torch.zeros_like(q), {}

    monkeypatch.setattr(owner._bridge, 'run', run)
    owner.try_attention(layer, value, value, value, metadata(), prefetch_ticket=ticket)
    assert ticket.state == 'RELEASED'
    pending = ticket_module.PrefetchTicket(
        policy, owner_hook=object(), slot=1, slot_generation=2, launch_ag=calls.append, ready_event=object()
    )
    pending.h2d_submitted()
    owner.register_prefetch_ticket(pending)
    owner.finish_request()
    assert pending.state == 'FAILED' and owner._bridge.failed
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['disabled_gate', 'unsupported', 'legacy', 'unprepared'])
def test_owner_unsupported_routes_are_immediate_before_arming(env, monkeypatch, kind):  # noqa: F811
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    monkeypatch.setenv('MINDIESD_FIRST_HEAD_AG_GATE', '1' if kind != 'disabled_gate' else '0')
    if kind == 'legacy':
        owner.bridge_enabled = False
    if kind != 'unprepared':
        prepare(env, owner, layer, value)
    if owner._bridge.executor is not None:
        owner._bridge.executor.supports_input_chunk_events = lambda **kwargs: kind != 'unsupported'
    assert not owner.prefetch_policy(layer).defer


def test_gate_policy_change_after_request_agreement_is_fatal(env, monkeypatch):  # noqa: F811
    owner, layer, value = env.h3.MindIESDH3Parallel(), attention(), tensor()
    monkeypatch.setenv('MINDIESD_FIRST_HEAD_AG_GATE', '0')
    prepare(env, owner, layer, value)
    monkeypatch.setenv('MINDIESD_FIRST_HEAD_AG_GATE', '1')
    with pytest.raises(RuntimeError):
        owner.prefetch_policy(layer)
    assert owner._bridge.failed


@pytest.mark.parametrize('partial_failure', [False, True])
def test_real_prefetch_early_h2d_multidtype_ready_publication(ticket_module, monkeypatch, partial_failure):
    namespace, cls = offload_methods()
    namespace.update(PrefetchGatePolicy=ticket_module.PrefetchGatePolicy, PrefetchTicket=ticket_module.PrefetchTicket)
    trace = []

    class Buffer:
        def __getitem__(self, key):
            return self

        def numel(self):
            return 2

        def copy_(self, other, **kwargs):
            trace.append('h2d')

    class Stream:
        def wait_stream(self, other):
            trace.append('wait_stream')

        def wait_event(self, event):
            assert event.recorded
            trace.append('wait_input')

    class Event:
        recorded = False

        def record(self, stream):
            self.recorded = True
            trace.append('ready_record')

    platform = types.SimpleNamespace(current_stream=lambda: Stream(), Event=Event, stream=lambda _: nullcontext())
    namespace['current_omni_platform'] = platform

    def gather(*args, **kwargs):
        trace.append('ag')
        if partial_failure and trace.count('ag') == 2:
            raise ValueError('second dtype AG')

    monkeypatch.setattr(torch.distributed, 'all_gather_into_tensor', gather)
    hook = cls.__new__(cls)
    hook._ticket_failed = False
    hook._slot_tickets = [None, None]
    hook._slot_generations = [0, 0]
    hook.ready_events = [None, None]
    hook._prev_hook = None
    hook._prefetch_done = None
    hook._shared_slot_group = [-1, -1]
    hook._group_id = 1
    hook._cached_repoint = ()
    hook.prefetch_counters = {}
    hook.copy_stream, hook.comm_stream = Stream(), Stream()
    hook.dp_size, hook.dp_group = 2, object()
    hook.cpu_shards = {torch.bfloat16: Buffer(), torch.float32: Buffer()}
    hook.gpu_buffers = [dict(hook.cpu_shards), dict(hook.cpu_shards)]
    hook.gpu_shard_buffers = [dict(hook.cpu_shards), dict(hook.cpu_shards)]
    hook._ag_output_sizes = {dtype: 4 for dtype in hook.cpu_shards}
    policy = ticket_module.PrefetchGatePolicy(True, 1, 2, 'supported')
    ticket = hook.prefetch_layer(0, policy=policy)
    assert trace.count('h2d') == 2 and 'ag' not in trace
    assert hook.ready_events[0] is None
    input_event = types.SimpleNamespace(recorded=True)
    if partial_failure:
        with pytest.raises(ValueError, match='second dtype'):
            ticket.release_after_input(types.SimpleNamespace(index=0), input_event)
        assert hook._ticket_failed and hook.ready_events[0] is None
        with pytest.raises(RuntimeError):
            hook._wait_for_weights(0)
        assert 'ready_record' not in trace
    else:
        ticket.release_after_input(types.SimpleNamespace(index=0), input_event)
        assert trace.index('h2d') < trace.index('wait_input') < trace.index('ag') < trace.index('ready_record')
        assert trace.count('ag') == 2 and hook.ready_events[0] is ticket.ready_event
