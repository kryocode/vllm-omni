# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-only offload observations at request boundaries; no device operations."""


def collect_offload_execution_snapshot(model):
    """Discover existing DLO hooks without exposing their plan storage to H3.

    Counts are hook-scoped, not unique global tensor or kernel counts. Shared
    hook objects are counted once. The function never creates a registry/hook.
    """
    result = {
        "hooks": 0,
        "failed": False,
        "prefetch": {
            name: 0
            for name in (
                "live",
                "h2d_submitted",
                "deferred",
                "released_after_input",
                "released_at_block_end",
                "released_ungated",
                "failed",
            )
        },
        "nz": {"plans": 0, "selected_weights": 0, "packed_weights": 0, "view_count": 0, "bind_count": 0},
    }
    seen = set()
    for module in model.modules():
        registry = getattr(module, "_hook_registry", None)
        if registry is None:
            continue
        hook = registry.get_hook("distributed_layerwise_offload")
        if hook is None or id(hook) in seen:
            continue
        seen.add(id(hook))
        snapshot = hook.execution_snapshot()
        result["hooks"] += 1
        result["failed"] = result["failed"] or snapshot["failed"]
        for name, value in snapshot["prefetch"].items():
            result["prefetch"][name] = result["prefetch"].get(name, 0) + value
        nz = snapshot["nz"]
        if nz is not None:
            result["nz"]["plans"] += 1
            for name in ("selected_weights", "packed_weights", "view_count", "bind_count"):
                result["nz"][name] += nz[name]
    return result


def counter_delta(before, after):
    """Subtract existing numeric counters; booleans/statuses are not counters."""
    return {name: value - before.get(name, 0) for name, value in after.items() if type(value) is int and name != "live"}
