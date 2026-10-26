# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-independent eager BF16 transport bridge.

``prepare_request`` is a collective request-boundary operation, never a lazy
per-layer cache miss. All ranks enter it exactly once, even when opting into
the legacy route. Forward calls issue only the existing transport collectives.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib import import_module

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RouteContract:
    """Rank-common tensor geometry and caller-owned semantic policy signature.

    ``semantic_key`` contains host scalars/tuples only. The caller includes all
    model/backend policies that could select a different collective route.
    Device ordinal and process-local object identities are intentionally absent.
    """

    shape: tuple[int, ...]
    dtype: str
    used_len: int
    semantic_key: tuple = ()


class AttentionBridge:
    """Persistent transport owner with explicit, rank-agreed request scope."""

    def __init__(self, *, overlap=True, head_chunk_size=1, execution_options=None):
        self.execution_options = dict(execution_options or {})
        self.overlap = overlap
        self.head_chunk_size = head_chunk_size
        self.state = None
        self.executor = None
        self._executor_key = None
        self._transport_context = None
        self.prepared = False
        self.accepted = False
        self.failed = False
        self.contract = None
        self.group = None
        self.calls = 0
        self.agreements = 0
        self._execution_policy = None
        self._graph_requested = False
        self.local_sequence_window = None

    def prepare_request(self, plan_id, contract, *, group, device, candidate=True, stream=None, graph_requested=False):
        """Agree once on a shared request boundary using fixed-size metadata.

        There is deliberately no local cache-hit shortcut: a rank-local miss
        must never decide whether to participate in this collective. A mismatch
        is fatal before attention; do not continue the denoise loop on one rank.
        """
        if self.failed:
            raise RuntimeError("Failed attention bridge: terminate participating workers")
        if self.prepared:
            raise RuntimeError("Previous attention request has not been finished")
        world = dist.get_world_size(group)
        context = (id(group), world, dist.get_rank(group), str(device))
        if self._transport_context is not None and context != self._transport_context:
            self.failed = True
            raise RuntimeError("Attention transport group/device context cannot change")
        parallel = import_module("mindiesd.parallel")
        window_fn = getattr(parallel, "rank_window", None)
        candidate = bool(candidate and callable(window_fn))
        if self.state is None:
            self.state = parallel.TransportState(group=group, stream=stream)
            if self.overlap:
                self.state.make_stream(device)
            self._transport_context = context
        elif stream is not None and self.state.stream is not stream:
            self.failed = True
            raise RuntimeError("Prepared attention must retain one live transport stream")
        per_rank = contract.shape[2] // world
        chunk = self.head_chunk_size or per_rank
        key = ("head_pipeline" if chunk == 1 else "generic", chunk)
        if chunk > 0:
            if self._executor_key != key:
                config = parallel.UlyssesConfig(implementation=key[0], head_chunk_size=chunk, **self.execution_options)
                self.executor = parallel.AttentionExecutor(config=config, state=self.state)
                self._executor_key = key
            execution_policy = self.executor.execution_signature(graph_requested=graph_requested)
        else:
            execution_policy = ("unsupported_chunk", chunk)
            candidate = False
        policy_error = None
        lookahead = self.state.lookahead
        if lookahead is not None and (type(lookahead) is not int or lookahead < 1):
            policy_error = "Attention lookahead must be a positive integer or None"
        if not self.execution_options and os.environ.get("MINDIESD_ATTENTION_GRAPH_PREFIX", "0") not in ("0", "1"):
            policy_error = "MINDIESD_ATTENTION_GRAPH_PREFIX must be 0 or 1"
        candidate = bool(candidate and policy_error is None)
        payload = {
            "plan_id": plan_id,
            "contract": asdict(contract),
            "overlap": self.overlap,
            "head_chunk_size": self.head_chunk_size,
            "eager": True,
            "execution_policy": execution_policy,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).digest()
        local = torch.tensor([int(candidate), *digest], dtype=torch.int64, device=device)
        gathered = torch.empty(world * local.numel(), dtype=torch.int64, device=device)
        try:
            # Control metadata is an issuer on this communicator too. Bind the
            # stream before the first agreement, including the legacy route.
            issuer = self.state.stream
            producer = torch.npu.current_stream(device) if issuer is not None else None
            if issuer is not None:
                issuer.wait_stream(producer)
            with torch.npu.stream(issuer) if issuer is not None else nullcontext():
                dist.all_gather_into_tensor(gathered, local, group=group)
                if issuer is not None:
                    local.record_stream(issuer)
                    gathered.record_stream(issuer)
            if issuer is not None:
                producer.wait_stream(issuer)
            # One host inspection at the request boundary; never per layer/head.
            rows = gathered.view(world, local.numel()).cpu().tolist()
        except BaseException:
            self.failed = True
            raise
        self.agreements += 1
        if any(row != rows[0] for row in rows[1:]) or rows[0][0] not in (0, 1):
            self.failed = True
            raise RuntimeError("Attention request rank agreement failed: route/config/contract differ")
        self.contract, self.group = contract, group
        self.prepared, self.accepted = True, bool(rows[0][0])
        self.local_sequence_window = (
            window_fn(context[2], world, contract.shape[1], contract.used_len) if self.accepted else None
        )
        self._execution_policy = execution_policy
        self._graph_requested = graph_requested
        if policy_error is not None:
            self.failed = True
            raise RuntimeError(policy_error)
        return self.accepted

    def validate_execution_policy(self):
        """Reject rank-local schedule drift before prefetch or data submission."""
        if self.failed:
            raise RuntimeError("Failed attention bridge: terminate participating workers")
        if self.prepared and self.accepted:
            if self.executor.execution_signature(graph_requested=self._graph_requested) != self._execution_policy:
                self.failed = True
                raise RuntimeError("Attention execution policy changed after request agreement")

    def run(self, q, k, v, *, backend, contract, on_input_chunk_ready=None, graph_signature=None):
        """Return ``(output, stats)`` or None before an agreed transport route.

        Once a route is accepted, changed geometry/metadata is an invariant
        violation, not permission for rank-local fallback. Collective/backend
        exceptions poison this owner and propagate without native retry.
        """
        if self.failed:
            raise RuntimeError("Failed attention bridge: terminate participating workers")
        if not self.prepared or not self.accepted:
            return None
        self.validate_execution_policy()
        if contract != self.contract or tuple(q.shape) != self.contract.shape:
            self.failed = True
            raise RuntimeError("Attention call differs from its prepared request contract")
        if (
            q.ndim != 4
            or k.shape != q.shape
            or v.shape != q.shape
            or q.dtype != torch.bfloat16
            or str(q.dtype) != contract.dtype
            or q.device.type != "npu"
            or any(t.dtype != q.dtype or t.device != q.device for t in (k, v))
            or self.contract.shape[2] % dist.get_world_size(self.group)
        ):
            self.failed = True
            raise RuntimeError("Attention Q/K/V violate the prepared BF16 Ulysses contract")
        try:
            output, stats = self.executor(
                q,
                k,
                v,
                backend=backend,
                group=self.group,
                used_len=contract.used_len,
                return_stats=True,
                on_input_chunk_ready=on_input_chunk_ready,
                graph_signature=graph_signature,
            )
            if output.shape != q.shape or output.dtype != q.dtype or output.device != q.device:
                raise RuntimeError("MindIE-SD did not restore the local BSND tensor contract")
        except BaseException:
            self.failed = True
            raise
        self.calls += 1
        return output, stats

    def finish_request(self):
        self.prepared = self.accepted = False
        self.contract = None
        self.group = None
        self._execution_policy = None
        self._graph_requested = False
        self.local_sequence_window = None
