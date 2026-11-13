# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit controls for MindIE-SD's complete distributed attention pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MindIEParallelConfig:
    overlap_comm: bool = False
    optimize_attention_pipeline: bool = True
    sparse_type: str = "rf_v3"
    bsa_backend: str = "cann"
    reorder_hoist: bool = False
    head_major_qkv: bool = False

    def __post_init__(self) -> None:
        for name in ("overlap_comm", "optimize_attention_pipeline", "reorder_hoist", "head_major_qkv"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"MindIE parallel {name} must be a boolean and agree across ranks")
        if self.sparse_type not in ("rf_v2", "rf_v3"):
            raise ValueError("MindIE parallel sparse_type must be rf_v2 or rf_v3")
        if self.bsa_backend not in ("cann", "eagle"):
            raise ValueError("MindIE parallel bsa_backend must be cann or eagle")
        if (self.reorder_hoist or self.bsa_backend == "eagle") and self.sparse_type != "rf_v3":
            raise ValueError("RF reorder hoist and Eagle BSA require rf_v3")
        if self.head_major_qkv and not (self.reorder_hoist and self.overlap_comm and self.optimize_attention_pipeline):
            raise ValueError("head_major_qkv requires reorder_hoist, overlap_comm and optimize_attention_pipeline")
