# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 metadata adaptation for the optional prepared MindIE bridge.

The pipeline CLI mode enables the new route after shared request preparation.
Explicit CLI policy refuses unprepared legacy fallback; old environment parsing
is retained only for compatibility callers.
Graph1 publishes prefetch at input0; Graph2 explicitly publishes at block end.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from importlib import import_module

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.mindiesd_native import NativeBackendFactory, rf_execution_policy
from vllm_omni.diffusion.attention.mindiesd_bridge import AttentionBridge, RouteContract
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_ulysses_mode
from vllm_omni.diffusion.offloader.execution_stats import collect_offload_execution_snapshot, counter_delta
from vllm_omni.diffusion.offloader.prefetch_ticket import PrefetchGatePolicy

from ..optimization import H3OptimizationConfig
from ..rf_layout import H3RFLayoutAdapter
from ._mindiesd_h3_legacy import MindIESDH3Parallel as _LegacyParallel
from ._mindiesd_h3_legacy import _graph_signature

logger = init_logger(__name__)

_ALLOWED_EXTRA = frozenset(
    (
        "h3_num_requests",
        "valid_kv_length",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "npu_attn_varlen",
        "laser_input_scale",
        "kv_cache_dtype",
    )
)
_MASK_FIELDS = (
    "attn_mask",
    "joint_attn_mask",
    "joint_query",
    "joint_key",
    "joint_value",
    "full_attn_spans",
    "query_ranges",
)


def _layout_key(layout):
    """Serialize immutable host layout fields without tensor reads or addresses."""
    if layout is None:
        return None
    if is_dataclass(layout):
        return _layout_key(asdict(layout))
    if isinstance(layout, dict):
        return tuple((key, _layout_key(value)) for key, value in sorted(layout.items()))
    if isinstance(layout, (tuple, list)):
        return tuple(_layout_key(value) for value in layout)
    if isinstance(layout, (str, int, float, bool)):
        return layout
    # Typed VideoTokenLayout/VideoTokenSpan may be ordinary classes in callers.
    fields = ("used_len", "prefix_len", "latent_grid", "video_spans", "start", "role", "pre_rearranged")
    values = tuple((key, _layout_key(getattr(layout, key))) for key in fields if hasattr(layout, key))
    if not values:
        raise ValueError("H3 video layout must contain host-only typed geometry")
    return values


def _graph_runtime_policy(config=None):
    # The RF switch may be driven by a sentinel file, so freeze its effective
    # value, not only the environment spelling. A changed rank-local graph key
    # must be rejected before one peer replays while another enters cold agree.
    rf = import_module("mindiesd.layers.flash_attn.rf_geometry")
    return (False if config is not None and config.isolated else bool(rf.rf_pool_split_enabled())), os.environ.get(
        "MINDIE_SD_FA_TYPE", ""
    )


def _local_metadata(attention, metadata):
    if metadata is None or metadata.extra.get("h3_num_requests") != 1:
        return None
    if metadata.extra.keys() - _ALLOWED_EXTRA or any(
        getattr(metadata, name, None) is not None for name in _MASK_FIELDS
    ):
        return None
    metadata = attention._with_kv_cache_dtype(metadata)
    if metadata.extra.get("kv_cache_dtype") not in (None, "float", "auto"):
        return None
    used = metadata.extra.get("valid_kv_length")
    if isinstance(used, bool) or not isinstance(used, int) or used <= 0:
        return None
    extra = dict(metadata.extra)
    if any(extra.get(name) != used for name in ("max_seqlen_q", "max_seqlen_k")) or not extra.get("npu_attn_varlen"):
        return None
    for name in ("cu_seqlens_q", "cu_seqlens_k"):
        cu = extra.get(name)
        if not torch.is_tensor(cu) or cu.ndim != 1 or cu.shape[0] not in (2, 3):
            return None
        extra[name] = cu[:2]
    return replace(metadata, extra=extra, packed_padding=None)


class MindIESDH3Parallel:
    """H3 model configuration and prefix/padding semantics; no transport algorithm."""

    def __init__(self, config=None):
        self.optimization_config = config if config is not None else H3OptimizationConfig.from_env()
        if not self.optimization_config.isolated:
            self.optimization_config.activate()
        self._legacy = _LegacyParallel(config=self.optimization_config)
        self.enabled = self._legacy.enabled
        self.overlap = self._legacy.overlap
        self.verify_first = self._legacy.verify_first
        self.head_chunk_size = self._legacy.head_chunk_size
        self.bridge_enabled = self._setting("VLLM_OMNI_MINIMAX_H3_MINDIESD_BRIDGE") == "1"
        self._bridge = AttentionBridge(
            overlap=self.overlap,
            head_chunk_size=self.head_chunk_size,
            execution_options=self.optimization_config.execution_options(),
        )
        self._backends = NativeBackendFactory()
        self.state = None
        self.calls = self.dense_calls = self.sparse_calls = 0
        self._verified_attention = set()
        self._request_id = 0
        self._request_layers = ()
        self._request_layer_ids = frozenset()
        self._request_policies = {}
        self._prepared_switches = None
        self._layout = None
        self._request_device = None
        self._request_world = None
        self._prepared_gate = None
        self._prepared_graph = None
        self._prepared_graph_single = None
        self._prepared_graph_policy = None
        self._prepared_qbmm_nz_env = None
        self._prepared_weight_contracts = frozenset()
        self._qbmm_nz_agreed_contracts = frozenset()
        self._qbmm_nz_agreed_group = None
        self._prefetch_tickets = []
        self.last_execution_receipt = None

    def _setting(self, name, default="0"):
        config = getattr(self, "optimization_config", None)
        return config.value(name, default) if config is not None else os.environ.get(name, default)

    def _graph_policy(self):
        config = getattr(self, "optimization_config", None)
        return _graph_runtime_policy(config) if config is not None and config.isolated else _graph_runtime_policy()

    def _validate_gate_policy(self):
        if self._bridge.failed:
            raise RuntimeError("Failed attention bridge: terminate participating workers")
        self._bridge.validate_execution_policy()
        if self._bridge.prepared and self._setting("MINDIESD_FIRST_HEAD_AG_GATE", "0") != self._prepared_gate:
            self._bridge.failed = True
            raise RuntimeError("Prefetch gate policy changed after request agreement")
        if self._bridge.prepared and self._setting("MINDIESD_QBMM_NZ", "0") != self._prepared_qbmm_nz_env:
            self._bridge.failed = True
            raise RuntimeError("QBMM NZ policy changed after request agreement; restart workers")
        if self._bridge.prepared and self._setting("MINDIESD_ATTENTION_GRAPH", "0") != self._prepared_graph:
            self._bridge.failed = True
            raise RuntimeError("Attention graph policy changed after request agreement; restart workers")
        if (
            self._bridge.prepared
            and self._setting("MINDIESD_ATTENTION_GRAPH_SINGLE", "0") != self._prepared_graph_single
        ):
            self._bridge.failed = True
            raise RuntimeError("Attention graph variant changed after request agreement; restart workers")
        if self._bridge.prepared and self._prepared_graph_policy is not None:
            if self._graph_policy() != self._prepared_graph_policy:
                self._bridge.failed = True
                raise RuntimeError("Attention graph consumer policy changed after request agreement; restart workers")

    def prefetch_policy(self, attention):
        """Decide on host facts before the offloader submits or arms a ticket."""
        self._validate_gate_policy()
        if self._bridge.prepared:
            switches = (self.enabled, self.bridge_enabled, self.overlap, self.verify_first, self.head_chunk_size)
            if (
                switches != self._prepared_switches
                or (id(attention) not in self._request_layer_ids and not attention.skip_sequence_parallel)
                or (
                    id(attention) in self._request_layer_ids
                    and self._backends.contract(attention) != self._request_policies[id(attention)]
                )
            ):
                self._bridge.failed = True
                raise RuntimeError("H3 prefetch consumer changed after request agreement")
        supported = bool(
            self._bridge.prepared
            and self._bridge.accepted
            and id(attention) in self._request_layer_ids
            and self._prepared_gate == "1"
            and self._bridge.executor.supports_input_chunk_events(
                device=self._request_device,
                dtype=torch.bfloat16,
                world=self._request_world,
            )
        )
        return PrefetchGatePolicy(
            supported,
            (id(self), self._request_id),
            id(attention),
            "input0 event" if supported else "immediate route",
        )

    def register_prefetch_ticket(self, ticket):
        # Registration supports teardown only; forwards must receive the exact
        # ticket explicitly. Keep unresolved work plus the current invocation.
        ticket.validate(request_id=(id(self), self._request_id), consumer_id=ticket.policy.consumer_id)
        self._prefetch_tickets = [item for item in self._prefetch_tickets if item.state not in ("RELEASED", "RETIRED")]
        self._prefetch_tickets.append(ticket)

    def abort_prefetch_tickets(self, error):
        self._bridge.failed = True
        for ticket in self._prefetch_tickets:
            ticket.abort(error)

    def validate_weight_layout_setup(self):
        """Capability check before immutable host NZ packing begins."""
        if not self.requires_request_agreement or self._bridge.failed:
            raise RuntimeError("QBMM NZ request owner is disabled or failed")

    @property
    def requires_request_agreement(self):
        # NZ owns request/group/slot proofs independently of attention routing.
        # Keep that control agreement even when attention uses the native route.
        return self.enabled or self.optimization_config.weight_layout == "nz"

    def assert_weight_layout_live(self, contract, agreement_group, *, historical=False):
        """Validate dynamic request proof without exposing owner internals."""
        self.validate_weight_layout_setup()
        proof_group = self._qbmm_nz_agreed_group if historical else self._bridge.group
        if proof_group is not agreement_group:
            raise RuntimeError("QBMM NZ request agreement group changed")
        if historical:
            if contract not in self._qbmm_nz_agreed_contracts:
                raise RuntimeError(
                    "QBMM NZ restoration requires a successful historical layout agreement; restart workers"
                )
        elif not self.request_prepared:
            raise RuntimeError(
                "QBMM NZ prefetch requires an active shared request; direct and step calls are unsupported"
            )
        elif contract not in self._prepared_weight_contracts:
            raise RuntimeError("QBMM NZ block contract was not covered by this request agreement")

    @property
    def request_prepared(self):
        return self._bridge.prepared

    def execution_snapshot(self):
        """Existing successful host returns and public transport/Graph counters.

        This does not synchronize or prove device completion. Counts include
        all prior requests; the request boundary computes separate deltas.
        """
        state = self.state
        transport = {} if state is None else state.stats()
        cache = None if state is None else state.graph_cache
        return {
            "request_sequence": self._request_id,
            "route": dict(getattr(self, "route_summary", {})),
            "optimization_groups": self.optimization_config.group_snapshot(),
            "failed": self._bridge.failed,
            "attention": {
                "successful_host_returns": self.calls,
                "prepared_host_returns": self._bridge.calls,
                "dense_host_returns": self.dense_calls,
                "sparse_host_returns": self.sparse_calls,
            },
            "transport": {
                name: transport.get(name, 0) for name in ("calls", "engaged", "single_stream_calls", "overlapped_calls")
            },
            "graph": {} if cache is None else cache.snapshot(),
        }

    def prepare_request(
        self, *, plan_id, shape, used_len, device, attention_layers, video_layout=None, weight_contract=()
    ):
        """All SP ranks call once before entering the request's block hooks.

        ``shape`` is this rank's equal BSND Q/K/V shape; ``attention_layers``
        contains the framework Attention consumers, in model order, excluding
        native-only/token-refiner layers. No current Q/K/V tensors are retained.
        """
        sp = get_sp_group()
        world = dist.get_world_size(sp.ulysses_group)
        self._request_layers = tuple(attention_layers)
        self._request_layer_ids = frozenset(id(layer) for layer in self._request_layers)
        self._request_policies = {
            id(layer): self._backends.contract(layer, declared=True) for layer in self._request_layers
        }
        self._prepared_switches = (
            self.enabled,
            self.bridge_enabled,
            self.overlap,
            self.verify_first,
            self.head_chunk_size,
        )
        self._layout = _layout_key(video_layout)
        self._request_device = device
        self._request_world = world
        self._prepared_gate = self._setting("MINDIESD_FIRST_HEAD_AG_GATE", "0")
        self._prepared_graph = self._setting("MINDIESD_ATTENTION_GRAPH", "0")
        self._prepared_graph_single = self._setting("MINDIESD_ATTENTION_GRAPH_SINGLE", "0")
        graph_capability = None
        if any(layer.attn_backend.get_name() == "RAINFUSION_ATTN" for layer in self._request_layers):
            self._prepared_graph_policy = self._graph_policy()
        if self._prepared_graph == "1":
            graph_api = import_module("mindiesd.parallel")
            graph_capability = graph_api.input_gate_supported()
            if self._prepared_graph_single == "1":
                single_support = getattr(graph_api, "single_graph_supported", None)
                graph_capability = graph_capability and callable(single_support) and bool(single_support())
            self._prepared_graph_policy = self._graph_policy()
        self._prepared_qbmm_nz_env = self._setting("MINDIESD_QBMM_NZ", "0")
        prefix_reason = None
        if self._prepared_graph == "1" and self._setting("MINDIESD_RF_GEOMETRY_CACHE") == "0":
            prefix_reason = "RF Graph requires prepared static geometry cache"
        if self._setting("MINDIESD_ATTENTION_GRAPH_PREFIX", "0") == "1":
            if self._prepared_graph != "1":
                prefix_reason = "prefix Graph requires attention Graph enabled"
            elif not self._request_layers:
                prefix_reason = "prefix Graph requires main DiT RF consumers"
            else:
                for layer in self._request_layers:
                    prefix_reason = self._backends.graph_prefix_reason(layer, video_layout, used_len)
                    if prefix_reason is not None:
                        break
        semantic = (
            "h3_prefix_v1",
            self._layout,
            self.enabled,
            self.bridge_enabled,
            self.verify_first,
            tuple(
                (name, self._setting(name, "0"))
                for name in (
                    "MINDIESD_ATTENTION_GRAPH",
                    "MINDIESD_ATTENTION_GRAPH_SINGLE",
                    "MINDIESD_ATTENTION_PR658",
                    "MINDIESD_ATTENTION_PIPELINE",
                    "MINDIESD_ATTENTION_PREPACK",
                    "MINDIESD_ATTENTION_D",
                    "MINDIESD_FIRST_HEAD_AG_GATE",
                )
            ),
            tuple(self._request_policies[id(layer)] for layer in self._request_layers),
            ("qbmm_nz_request_v1", self._prepared_qbmm_nz_env, weight_contract),
            ("graph_input_gate_v1", graph_capability, self._prepared_graph_policy),
            ("prefix_eligibility_v1", prefix_reason),
        )
        contract = RouteContract(tuple(shape), str(torch.bfloat16), used_len, semantic)
        candidate = bool(
            self.enabled
            and self.bridge_enabled
            and device.type == "npu"
            and len(shape) == 4
            and shape[0] == 1
            and min(shape) > 0
            and world > 1
            and world == sp.world_size
            and sp.ring_world_size == 1
            and shape[2] % world == 0
            and 0 < used_len <= shape[1] * world
            and get_ulysses_mode(default="strict") == "strict"
            and self._request_layers
            and all(self._backends.supports(layer, declared=True) for layer in self._request_layers)
            and prefix_reason is None
        )
        accepted = self._bridge.prepare_request(
            plan_id,
            contract,
            group=sp.ulysses_group,
            device=device,
            candidate=candidate,
            stream=self._legacy.state.stream if self._legacy.state is not None else None,
            graph_requested=self._prepared_graph == "1",
        )
        self.route_summary = {
            "requested": self.enabled and self.bridge_enabled,
            "supported": candidate,
            "selected": "prepared_ulysses" if accepted else "legacy",
            "reason": prefix_reason or ("eligible" if accepted else "outside prepared BF16 pure-Ulysses contract"),
            "rf_layout": self.optimization_config.rf_layout,
            "graph_requested": self._prepared_graph == "1",
            "graph_variant": (
                ("graph2" if self._prepared_graph_single == "1" else "graph1") if self._prepared_graph == "1" else "off"
            ),
        }
        logger.info_once("MiniMax-H3 optimization route: %s", json.dumps(self.route_summary, sort_keys=True))
        if prefix_reason is not None:
            self._bridge.failed = True
            raise RuntimeError("H3 requested prefix Graph is unsupported: " + prefix_reason)
        if dict(self.optimization_config.requested_groups).get("COMM_MODE") == "pipeline" and not accepted:
            self._bridge.failed = True
            raise RuntimeError("H3 COMM_MODE=pipeline is unsupported for this request; refuse legacy downgrade")
        if self._setting("MINDIESD_ATTENTION_GRAPH_PREFIX", "0") not in ("0", "1"):
            self._bridge.failed = True
            raise RuntimeError("MINDIESD_ATTENTION_GRAPH_PREFIX must be 0 or 1")
        if self._prepared_graph not in ("0", "1"):
            self._bridge.failed = True
            raise RuntimeError("MINDIESD_ATTENTION_GRAPH must be 0 or 1")
        if self._prepared_graph_single not in ("0", "1") or (
            self._prepared_graph_single == "1"
            and (self._prepared_graph != "1" or self._setting("MINDIESD_ATTENTION_GRAPH_PREFIX", "0") == "1")
        ):
            self._bridge.failed = True
            raise RuntimeError("Invalid or conflicting Graph1/Graph2 policy")
        if self._prepared_graph == "1":
            if self._prepared_graph_single == "1":
                signature = self._bridge.executor.execution_signature(graph_requested=True)
                graph_capability = graph_capability and ("graph_single", "1") in signature
            if not accepted or not graph_capability or self.head_chunk_size != 1 or not self.overlap:
                self._bridge.failed = True
                raise RuntimeError("Prepared sparse graph route is unsupported; refuse legacy downgrade")
        # Do this after all ranks completed the existing fixed agreement, so a
        # changed rank-local flag cannot strand a peer in the control exchange.
        if any(contract[1] != self._prepared_qbmm_nz_env for _, contract in weight_contract):
            self._bridge.failed = True
            raise RuntimeError("QBMM NZ setup layout differs from request policy; restart workers")
        self._prepared_weight_contracts = frozenset(contract for _, contract in weight_contract)
        self._qbmm_nz_agreed_contracts = self._qbmm_nz_agreed_contracts | self._prepared_weight_contracts
        self._qbmm_nz_agreed_group = sp.ulysses_group
        self.state = self._bridge.state
        # Direct/step-mode and rejected requests retain compatibility behavior,
        # sharing the control agreement's persistent collective issuer.
        if self._legacy.state is None:
            parallel = import_module("mindiesd.parallel")
            self._legacy.state = parallel.ParallelState(group=sp.ulysses_group, stream=self.state.stream)
        return accepted

    def finish_request(self):
        pending = [ticket for ticket in self._prefetch_tickets if ticket.state not in ("RELEASED", "RETIRED")]
        if pending:
            self.abort_prefetch_tickets(RuntimeError("Request ended with unresolved prefetch tickets"))
        self._prefetch_tickets.clear()
        self._bridge.finish_request()
        self._request_layers = ()
        self._request_layer_ids = frozenset()
        self._request_policies = {}
        self._prepared_switches = None
        self._layout = None
        self._request_device = self._request_world = self._prepared_gate = None
        self._prepared_qbmm_nz_env = None
        self._prepared_graph = None
        self._prepared_graph_single = None
        self._prepared_graph_policy = None
        self._prepared_weight_contracts = frozenset()

    @torch.compiler.disable
    def try_attention(self, attention, query, key, value, metadata, *, prefetch_ticket=None):
        try:
            self._validate_gate_policy()
            if prefetch_ticket is not None:
                prefetch_ticket.validate(request_id=(id(self), self._request_id), consumer_id=id(attention))
                policy = self.prefetch_policy(attention)
                # Rank-local weights can choose immediate mode on a capable
                # attention route. An armed deferred ticket cannot downgrade.
                if prefetch_ticket.policy.defer and not policy.defer:
                    raise RuntimeError("Deferred prefetch lost its prepared input-event route")
            return self._try_attention(attention, query, key, value, metadata, prefetch_ticket=prefetch_ticket)
        except BaseException as error:
            self._bridge.failed = True
            if prefetch_ticket is not None:
                prefetch_ticket.abort(error)
            raise

    def _try_attention(self, attention, query, key, value, metadata, *, prefetch_ticket=None):
        if self._bridge.failed:
            raise RuntimeError("Failed attention bridge: terminate participating workers")
        if self._bridge.prepared:
            try:
                switches = (self.enabled, self.bridge_enabled, self.overlap, self.verify_first, self.head_chunk_size)
                if switches != self._prepared_switches:
                    raise RuntimeError("H3 owner switches differ from the prepared request contract")
                # Native-only refiner layers were excluded at the boundary.
                # An included layer becoming native-only is a contract failure.
                if id(attention) not in self._request_layer_ids and attention.skip_sequence_parallel:
                    return None
                if (
                    id(attention) not in self._request_layer_ids
                    or self._backends.contract(attention) != self._request_policies[id(attention)]
                ):
                    raise RuntimeError("H3 layer route differs from the prepared request contract")
                if not self._bridge.accepted:
                    return self._run_legacy(attention, query, key, value, metadata)
                return self._run_prepared(attention, query, key, value, metadata, prefetch_ticket=prefetch_ticket)
            except BaseException:
                self._bridge.failed = True
                raise
        if not self.enabled or attention.skip_sequence_parallel:
            return None
        if not self.bridge_enabled or not self._bridge.prepared or not self._bridge.accepted:
            return self._run_legacy(attention, query, key, value, metadata)

    def _run_legacy(self, attention, query, key, value, metadata):
        config = getattr(self, "optimization_config", None)
        if config is not None and config.isolated:
            raise RuntimeError(
                "Explicit pipeline requires a prepared shared request; direct/step legacy fallback is unsupported"
            )
        self._legacy.enabled = self.enabled
        self._legacy.verify_first = self.verify_first
        before = (self._legacy.calls, self._legacy.dense_calls, self._legacy.sparse_calls)
        try:
            output = self._legacy.try_attention(attention, query, key, value, metadata)
        except BaseException:
            self._bridge.failed = True
            raise
        self.state = self._legacy.state
        self.calls += self._legacy.calls - before[0]
        self.dense_calls += self._legacy.dense_calls - before[1]
        self.sparse_calls += self._legacy.sparse_calls - before[2]
        return output

    def _run_prepared(self, attention, query, key, value, metadata, *, prefetch_ticket=None):
        local = _local_metadata(attention, metadata)
        if (
            local is None
            or id(attention) not in self._request_layer_ids
            or local.extra.get("valid_kv_length") != self._bridge.contract.used_len
            or _layout_key(local.video_layout) != self._layout
        ):
            self._bridge.failed = True
            raise RuntimeError("H3 attention metadata differs from its prepared request contract")
        backend = self._backends.build(attention, local)
        callback = None
        if prefetch_ticket is not None and prefetch_ticket.policy.defer:
            if self._prepared_graph == "1" and self._prepared_graph_single == "1" and backend.is_sparse:
                prefetch_ticket.defer_to_block_end()
            else:

                def callback(chunk, event):
                    if chunk.index == 0:
                        prefetch_ticket.release_after_event(event)
                        # The real launch callback records its completion fence.
                        return prefetch_ticket.ready_event

        graph_signature = None
        if self._prepared_graph == "1" and backend.is_sparse:
            graph_signature = (
                _graph_signature(attention.attention, backend.plan, local, config=self.optimization_config)
                if self.optimization_config.isolated
                else _graph_signature(attention.attention, backend.plan, local)
            )
            if graph_signature is None:
                raise RuntimeError("Requested sparse graph consumer has no stable graph signature")
            if graph_signature[-2:] != self._prepared_graph_policy:
                raise RuntimeError("Attention graph consumer policy changed while building its graph signature")
            # Cache identity always uses the agreed policy, even if the external
            # sentinel changes between the two host validation reads.
            graph_signature = (
                graph_signature[:-2]
                + (("rf_row_layout", bool(getattr(backend.plan, "pre_rearranged", False))),)
                + (
                    (("rf_execution_policy", rf_execution_policy(self.optimization_config)),)
                    if rf_execution_policy(self.optimization_config)
                    else ()
                )
                + self._prepared_graph_policy
            )

        output, _ = self._bridge.run(
            query,
            key,
            value,
            backend=backend,
            contract=self._bridge.contract,
            on_input_chunk_ready=callback,
            graph_signature=graph_signature,
        )
        rank = dist.get_rank(self._bridge.group)
        used = self._bridge.contract.used_len
        local_used = min(query.shape[1], max(0, used - rank * query.shape[1]))
        if local_used < query.shape[1]:
            output[:, local_used:].copy_(query[:, local_used:])
        if self.verify_first:
            signature = (attention, tuple(query.shape), query.dtype, used, repr(backend.plan))
            if signature not in self._verified_attention:
                reference = attention(query, key, value, metadata)
                if not torch.equal(output[:, :local_used], reference[:, :local_used]):
                    self._bridge.failed = True
                    raise AssertionError("MindIE-SD parallel attention is not bit-exact on valid rows")
                self._verified_attention.add(signature)
        self.calls += 1
        self.sparse_calls += int(backend.is_sparse)
        self.dense_calls += int(not backend.is_sparse)
        return output


@contextmanager
def _observe_h3_request(owner, model):
    """One local-rank host receipt, including setup/body/cleanup failures.

    Module scans and JSON logging have a small host cost at this boundary.
    No device operation, synchronization or telemetry collective is added.
    """
    owner.route_summary = {
        "requested": owner.enabled and owner.bridge_enabled,
        "supported": None,
        "selected": "not_prepared" if owner.enabled else "disabled",
        "rf_layout": owner.optimization_config.rf_layout,
    }
    before = owner.execution_snapshot()
    offload_before = collect_offload_execution_snapshot(model)
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        after = owner.execution_snapshot()
        offload_after = collect_offload_execution_snapshot(model)
        attention_delta = counter_delta(before["attention"], after["attention"])
        transport_delta = counter_delta(before["transport"], after["transport"])
        graph_delta = counter_delta(before["graph"], after["graph"])
        prefetch_delta = counter_delta(offload_before["prefetch"], offload_after["prefetch"])
        nz_delta = {
            name: offload_after["nz"][name] - offload_before["nz"][name]
            for name in ("packed_weights", "view_count", "bind_count")
        }
        returned = attention_delta["successful_host_returns"]
        activity = any(
            value > 0
            for values in (attention_delta, transport_delta, graph_delta, prefetch_delta, nz_delta)
            for value in values.values()
        )
        failed = failure is not None or after["failed"] or offload_after["failed"]
        receipt = {
            "event": "h3_denoise_scope_end",
            "rank": dist.get_rank() if dist.is_initialized() else None,
            "pid": os.getpid(),
            "request_sequence": after["request_sequence"],
            "request_id_source": "local_model_sequence",
            "outcome": "failed" if failed else "host_scope_returned",
            "error_type": None if failure is None else type(failure).__name__,
            "executed": returned > 0 and not failed,
            "execution_status": ("partial_failed" if activity else "failed_before_return")
            if failed
            else ("host_returns_observed" if returned else "no_calls"),
            "device_completion": "not_observed",
            "nz_evidence": "weight_binding_only_not_kernel_execution",
            "request_delta": {
                "attention": attention_delta,
                "transport": transport_delta,
                "graph": graph_delta,
                "prefetch": prefetch_delta,
                "nz": nz_delta,
            },
            "cumulative": {"attention_owner": after, "offload": offload_after},
        }
        owner.last_execution_receipt = receipt
        logger.info("MiniMax-H3 execution receipt: %s", json.dumps(receipt, sort_keys=True))


@contextmanager
def prepare_h3_request(model, branch):
    """Shared diffuse boundary. Step-mode and direct layer calls stay legacy.

    With MINDIESD and NZ off this retains the zero-collective native route and
    emits only a host receipt. NZ requires its weight-layout agreement even
    when optimized attention is off. Every participating rank agrees once,
    including BRIDGE=0 participants;
    rank-local switches never decide collective participation.
    """
    owner = getattr(model, "_mindiesd_parallel", None)
    if owner is None:
        yield
        return
    owner._request_id += 1
    with _observe_h3_request(owner, model):
        if not owner.requires_request_agreement:
            yield
            return
        modules = tuple(
            module
            for module in model.modules()
            if hasattr(module, "attention")
            and callable(getattr(module.attention, "_get_active_parallel_strategy", None))
            and not module.attention.skip_sequence_parallel
        )
        layers = tuple(module.attention for module in modules)
        local_len = model._rope_local_span(branch.seq_len)[1]
        shape = (1, local_len, modules[0].num_heads, modules[0].head_dim) if modules else (1, local_len, 1, 1)
        named_modules = getattr(model, "named_modules", None)
        weight_contract = (
            tuple(
                (name, module._mindiesd_qbmm_nz_contract)
                for name, module in named_modules()
                if hasattr(module, "_mindiesd_qbmm_nz_contract")
            )
            if callable(named_modules)
            else ()
        )
        owner.prepare_request(
            plan_id=owner._request_id,
            shape=shape,
            used_len=branch.used_len,
            device=branch.device,
            attention_layers=layers,
            video_layout=H3RFLayoutAdapter.block_metadata(
                branch.static_kwargs.get("video_token_layout"),
                enabled=owner.optimization_config.rf_layout == "post_embed",
            ),
            weight_contract=weight_contract,
        )
        try:
            yield
        except BaseException as error:
            owner.abort_prefetch_tickets(error)
            raise
        finally:
            owner.finish_request()
