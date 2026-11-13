# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Minimal controls for MindIE-SD's complete distributed attention pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MindIEParallelConfig:
    """Enable the SDK pipeline and optionally hoist its RF row permutation.

    Precision, sparse algorithm and operator selection belong to the parent
    block-sparse attention config.  The first integration deliberately fixes
    those contracts to BF16 RF-v3 on the stock CANN consumer.
    """

    overlap_comm: bool = True
    reorder_hoist: bool = True

    def __post_init__(self) -> None:
        for name in ("overlap_comm", "reorder_hoist"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"MindIE parallel {name} must be a boolean and agree across ranks")
