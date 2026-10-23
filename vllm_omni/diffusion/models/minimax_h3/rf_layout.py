# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3's explicit canonical-embedding / RF-block / canonical-output boundary.

RF owns the permutation geometry. This adapter owns distributed row routing and
the original token owner's AdaLN semantics. It does not alter model inputs or
install hooks, and has no knowledge of attention transport implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RowRoute:
    send_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]


def build_row_route(permutation, rank: int, world: int) -> RowRoute:
    """Plan one equal row shard, retaining variable all-to-all split sizes."""
    p = np.asarray(permutation, dtype=np.int64)
    n = len(p)
    if world < 1 or not 0 <= rank < world or n % world or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError("RF row routing requires a complete permutation and equal rank shards")
    span = n // world
    inverse = np.argsort(p)
    old = np.arange(rank * span, (rank + 1) * span)
    destinations = inverse[old] // span
    send = np.argsort(destinations, kind="stable")
    desired = p[rank * span : (rank + 1) * span]
    sources = desired // span
    received = np.concatenate([np.sort(desired[sources == i]) for i in range(world)])
    lookup = np.full(n, -1, dtype=np.int64)
    lookup[received] = np.arange(span)
    return RowRoute(
        tuple(send.tolist()),
        tuple(lookup[desired].tolist()),
        tuple(np.bincount(destinations, minlength=world).tolist()),
        tuple(np.bincount(sources, minlength=world).tolist()),
    )


class _DeviceRoute:
    def __init__(self, plan: RowRoute, device):
        self.plan = plan
        self.send = torch.tensor(plan.send_indices, dtype=torch.long, device=device)
        self.output = torch.tensor(plan.output_indices, dtype=torch.long, device=device)

    def run(self, rows, group):
        packed = rows.index_select(0, self.send).contiguous()
        received = torch.empty_like(packed)
        dist.all_to_all_single(
            received,
            packed,
            output_split_sizes=list(self.plan.receive_counts),
            input_split_sizes=list(self.plan.send_counts),
            group=group,
        )
        return received.index_select(0, self.output)


@dataclass(frozen=True)
class H3RFBlockLayout:
    """One forward's block layout; model-specific semantics stay in H3."""

    forward: _DeviceRoute
    reverse: _DeviceRoute
    indices: torch.Tensor
    group: object
    world: int
    local_length: int

    def enter(self, hidden, rope, inverse_indices, token_tags, *, modality_count: int, adaln_rows: int):
        if hidden.shape[0] != self.local_length or rope.shape[0] != self.local_length or rope.dtype != hidden.dtype:
            raise ValueError("H3 RF boundary requires canonical, equally sharded hidden and RoPE rows")
        packed = self.forward.run(torch.cat((hidden, rope), dim=-1), self.group)
        combined = (
            inverse_indices.reshape(-1).index_select(0, self.indices) * modality_count
            + token_tags.reshape(-1).index_select(0, self.indices).clamp(min=0)
            + (self.indices // self.local_length) * adaln_rows
        )
        return packed[:, : hidden.shape[1]].contiguous(), packed[:, hidden.shape[1] :].contiguous(), combined

    def restore(self, hidden):
        return self.reverse.run(hidden, self.group)

    def adaln_parameters(self, parameters):
        """Preserve each token's original owner's BF16 projection result."""
        packed = torch.cat(parameters, dim=-1).contiguous()
        gathered = torch.empty(
            (self.world * packed.shape[0], packed.shape[1]), dtype=packed.dtype, device=packed.device
        )
        dist.all_gather_into_tensor(gathered, packed, group=self.group)
        return tuple(gathered.chunk(len(parameters), dim=-1))


class H3RFLayoutAdapter:
    """One bounded geometry entry per model; no request tensors are retained."""

    def __init__(self):
        self._key = None
        self._plan = None
        self._active = False

    @contextmanager
    def forward_scope(self, *, enabled: bool):
        """Keep the original non-reentrant limit without retaining call data."""
        if not enabled:
            yield
            return
        if self._active:
            raise RuntimeError("H3 post-embedding RF forward is not reentrant")
        self._active = True
        try:
            yield
        finally:
            self._active = False

    @staticmethod
    def block_metadata(layout, *, enabled: bool):
        if not enabled or layout is None or not layout.video_spans:
            return layout
        if layout.pre_rearranged:
            raise ValueError("H3 post-embedding RF layout requires canonical model inputs")
        return replace(layout, pre_rearranged=True)

    def prepare(self, layout, *, seq_len: int, used_len: int, device, sp):
        if sp.world_size <= 1 or sp.ring_world_size != 1 or seq_len % sp.world_size:
            raise ValueError("H3 RF post-embedding routing currently requires equal-shard pure Ulysses")
        key = (
            seq_len, used_len, sp.world_size, sp.rank_in_group, id(sp.device_group), str(device),
            tuple((span.start, tuple(span.latent_grid)) for span in layout.video_spans),
        )
        if key != self._key:
            from mindiesd.layers.flash_attn.rf_geometry import rf_multi_span_permutation

            permutation, _ = rf_multi_span_permutation(
                [{"start": span.start, "latent_shape": list(span.latent_grid)} for span in layout.video_spans],
                used_len, 128, "BSND", torch.device("cpu"),
            )
            if seq_len > used_len:
                permutation = torch.cat((permutation, torch.arange(used_len, seq_len, dtype=torch.long)))
            inverse = torch.argsort(permutation)
            rank, world = sp.rank_in_group, sp.world_size
            local = seq_len // world
            self._plan = H3RFBlockLayout(
                _DeviceRoute(build_row_route(permutation.numpy(), rank, world), device),
                _DeviceRoute(build_row_route(inverse.numpy(), rank, world), device),
                permutation[rank * local : (rank + 1) * local].to(device),
                sp.device_group, world, local,
            )
            self._key = key
        return self._plan
