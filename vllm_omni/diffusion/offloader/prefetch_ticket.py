# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit framework ownership of a single next-block weight prefetch."""

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefetchGatePolicy:
    defer: bool
    request_id: object
    consumer_id: object
    reason: str


class PrefetchTicket:
    """One irreversible AG submission; failure never authorizes a retry.

    Events are recorded by the transport/launch callback before publication.
    Recorded events need no host synchronization: callers enqueue stream waits.
    This object retains launch resources through the backend's transfer drain.
    """

    def __init__(self, policy, *, owner_hook, slot, slot_generation, launch_ag, ready_event, counters=None):
        self.policy = policy
        self.owner_hook = owner_hook
        self.slot = slot
        self.slot_generation = slot_generation
        self._launch_ag = launch_ag
        self.ready_event = ready_event
        self.input_event = None
        self.state = "NEW"
        self.launch_attempts = 0
        self.error = None
        self._scope_active = False
        self._scope_used = False
        self._block_end_release = False
        self.counters = counters if counters is not None else {}
        self._count("live", 1)

    def _count(self, name, amount=1):
        self.counters[name] = self.counters.get(name, 0) + amount

    def h2d_submitted(self):
        if self.state != "NEW":
            raise RuntimeError("Prefetch H2D already submitted")
        self.state = "H2D_SUBMITTED"
        self._count("h2d_submitted")
        if self.policy.defer:
            self.state = "ARMED"
            self._count("deferred")

    def validate(self, *, request_id, consumer_id):
        if (request_id, consumer_id) != (self.policy.request_id, self.policy.consumer_id):
            raise RuntimeError("Foreign prefetch request/consumer identity")
        if self.state not in ("ARMED", "H2D_SUBMITTED", "RELEASED"):
            raise RuntimeError("Prefetch ticket cannot be used after failure or retirement")
        if self._scope_used and not self._scope_active or self.policy.defer and self.state == "RELEASED":
            raise RuntimeError("Prefetch ticket cannot be reused by another attention invocation")

    def _release(self, event, counter):
        # Mark before entering a callback that may partially submit multiple AGs.
        self.state = "RELEASING"
        self.launch_attempts += 1
        self.input_event = event
        try:
            self._launch_ag(event)
        except BaseException as error:
            self.abort(error)
            raise
        self.state = "RELEASED"
        self._count(counter)
        self._count("live", -1)

    def release_after_event(self, event):
        """Release once after the adapter's recorded dependency event."""
        if self.state != "ARMED" or event is None or self._block_end_release:
            raise RuntimeError("Prefetch release requires a recorded event and an armed ticket")
        self._release(event, "released_after_input")

    def defer_to_block_end(self):
        """Graph2 deliberately has no host callback inside a replayed region.

        Only an active, deferred invocation may opt into block-end publication.
        A failing block never publishes; Graph1 retains its strict input gate.
        """
        if self.state != "ARMED" or not self._scope_active or self._block_end_release:
            raise RuntimeError("Block-end prefetch requires one active armed ticket")
        self._block_end_release = True

    def release_after_input(self, chunk, event):
        """Compatibility entrance retaining the original Ulysses head0 check."""
        if getattr(chunk, "index", None) != 0:
            raise RuntimeError("Legacy prefetch release requires input0")
        self.release_after_event(event)

    def release_ungated(self, reason):
        if self.policy.defer or self.state != "H2D_SUBMITTED":
            raise RuntimeError("Immediate prefetch release is not a deferred rescue path")
        self._release(None, "released_ungated")

    def abort(self, error):
        if self.state in ("FAILED", "RETIRED"):
            return
        if self.state != "RELEASED":
            self._count("live", -1)
        self.state = "FAILED"
        self.error = error
        self._count("failed")
        # Retain resources and events, but invalidate public readiness.
        if hasattr(self.owner_hook, "_ticket_failed"):
            self.owner_hook._ticket_failed = True
        if hasattr(self.owner_hook, "ready_events"):
            self.owner_hook.ready_events[self.slot] = None

    def assert_ready(self, *, slot_generation):
        if self.state != "RELEASED" or slot_generation != self.slot_generation or self.launch_attempts != 1:
            raise RuntimeError("Prefetch slot is stale, failed, or has no recorded ready event")

    def finish(self):
        if self._block_end_release and self.state == "ARMED":
            self._release(None, "released_at_block_end")
        if self.state != "RELEASED" or self.launch_attempts != 1:
            error = RuntimeError("Block completed without its promised prefetch release")
            self.abort(error)
            raise error

    def retire(self):
        self.assert_ready(slot_generation=self.slot_generation)
        self.state = "RETIRED"
        self._launch_ag = None
        self.input_event = None

    @contextmanager
    def scope(self):
        if self._scope_used:
            raise RuntimeError("Prefetch block invocation is single-use and not reentrant")
        self._scope_used = True
        self._scope_active = True
        try:
            yield self
            self.finish()
        except BaseException as error:
            self.abort(error)
            raise
        finally:
            self._scope_active = False
