# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Process-local resources whose lifetime is bounded by a live SP coordinator."""

from collections.abc import Callable
from typing import Protocol, TypeVar, cast


class ParallelResource(Protocol):
    def close(self) -> None:
        """Drain submitted work and release storage before process-group teardown."""


R = TypeVar("R", bound=ParallelResource)


class SequenceParallelResources:
    """Share resources across layers, without a module-global communicator cache.

    Workers serialize forwards. Keys include the resource kind, process-group
    object and device. Construction and close must follow the same order on
    every participating rank; execution failures never trigger a local retry.
    """

    def __init__(self) -> None:
        self._resources: dict[tuple, ParallelResource] = {}
        self._closed = False

    def get_or_create(self, key: tuple, factory: Callable[[], R]) -> R:
        if self._closed:
            raise RuntimeError("sequence-parallel resources are closed")
        if key not in self._resources:
            self._resources[key] = factory()
        return cast(R, self._resources[key])

    def close(self) -> None:
        # Keep failed resources and their storage alive. A failed close must
        # prevent PG teardown, and can be retried by the worker shutdown path.
        for key in tuple(self._resources):
            self._resources[key].close()
            del self._resources[key]
        self._closed = True
