# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Default-off FRACTAL_NZ weights for MiniMax-H3 DiT.

Offload ``reactor`` transports a one-time CPU permutation and creates a
non-owning format-29 view of the uploaded slot. Its owner must retain the source
buffer until the last consumer finishes. ``cast`` allocates an NZ destination
and copies the ND weight on each upload, allowing padded shapes. Resident DiTs
always cast once after the loader has finalized quantization.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

WEIGHT_NZ_ENV = "VLLM_OMNI_MINIMAX_H3_WEIGHT_NZ"
NZ_MODE_ENV = "VLLM_OMNI_MINIMAX_H3_WEIGHT_NZ_MODE"
_ACL_FORMAT_FRACTAL_NZ = 29


def weight_nz_enabled() -> bool:
    """Resolve the opt-in without taking over the legacy QBMM switch."""
    from vllm_omni.diffusion.layers import minimax_h3_acceleration

    enabled = minimax_h3_acceleration.enabled("weight_nz")
    legacy = os.environ.get("H3_QBMM_WEIGHT_LAYOUT", "").strip().lower()
    if enabled and legacy in ("device_nz", "host_nz"):
        raise ValueError(f"{WEIGHT_NZ_ENV} conflicts with H3_QBMM_WEIGHT_LAYOUT={legacy}")
    return enabled


def nz_mode() -> str:
    """Choose the upload mechanism; MODE alone never enables weight NZ."""
    requested = os.environ.get(NZ_MODE_ENV, "").strip().lower()
    if requested in ("", "reactor"):
        return "reactor"
    if requested == "cast":
        return "cast"
    if weight_nz_enabled():
        raise ValueError(f"{NZ_MODE_ENV} must be reactor or cast, got {requested!r}")
    return "reactor"


# Preserve the H3 helper API while loading Omni primitives only for selected paths.
def _npu_format(weight):
    from vllm_omni.diffusion.layers.npu_weight_layout import get_npu_format

    return get_npu_format(weight)


def maybe_cast_weight_to_nz(weight):
    from vllm_omni.diffusion.layers.npu_weight_layout import maybe_cast_weight_to_nz as operation

    return operation(weight)


def validate_weight_nz(mode):
    from vllm_omni.diffusion.layers.npu_weight_layout import validate_weight_nz as operation

    return operation(mode)


def _selected_weight(module: nn.Module) -> torch.Tensor | None:
    if not isinstance(module, LinearBase):
        return None
    weight = getattr(module, "weight", None)
    if weight is None or weight.dim() != 2:
        return None
    method = getattr(module, "quant_method", None)
    if isinstance(method, UnquantizedLinearMethod):
        return weight if weight.dtype in (torch.bfloat16, torch.float16) else None
    from vllm_omni.quantization.int8_config import NPUInt8LinearMethod, NPUInt8OnlineLinearMethod

    if weight.dtype != torch.int8 or not isinstance(method, NPUInt8LinearMethod):
        return None
    if isinstance(method, NPUInt8OnlineLinearMethod):
        return weight if getattr(module, "_already_called_process_weights_after_loading", False) else None
    # Offline INT8 has no finalization marker. Its loader transposes [N,K] to
    # [K,N] and squeezes [N,1] scale only after loading has completed.
    scale = getattr(module, "weight_scale", None)
    final_shape = (
        getattr(module, "input_size_per_partition", None),
        getattr(module, "output_size_per_partition", None),
    )
    return weight if tuple(weight.shape) == final_shape and scale is not None and scale.dim() <= 1 else None


def nz_reactor_weight_names(model: nn.Module) -> set[str]:
    """Select H3 DiT/block linear weights; never scan arbitrary 2-D tensors."""
    return {
        f"{name}.weight" if name else "weight"
        for name, module in model.named_modules()
        if _selected_weight(module) is not None
    }


def install_minimax_h3_dit_weight_nz(model: nn.Module, od_config: Any | None = None) -> int:
    """Cast resident H3 DiT weights once after loading/final quantization."""
    if not weight_nz_enabled():
        return 0
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    from vllm_omni.diffusion.offloader.config import DIT_COMPONENT, OffloadStrategy, resolve_offload

    if not isinstance(model, MiniMaxH3DiTModel):
        return 0
    offload = resolve_offload(od_config)
    if offload.strategy is not OffloadStrategy.NONE and offload.offloads(DIT_COMPONENT):
        return 0
    # MODE controls offload uploads; resident weights always use a one-time cast.
    nz_mode()
    selected = [(module, weight) for module in model.modules() if (weight := _selected_weight(module)) is not None]
    pending = [(module, weight) for module, weight in selected if _npu_format(weight) != _ACL_FORMAT_FRACTAL_NZ]
    if not pending:
        return 0
    validate_weight_nz("cast")
    for module, weight in pending:
        old_data = weight.data
        cast = maybe_cast_weight_to_nz(old_data)
        try:
            weight.data = cast
            if _npu_format(weight) != _ACL_FORMAT_FRACTAL_NZ:
                raise RuntimeError("MiniMax-H3 resident parameter did not retain NZ format 29")
        except Exception:
            weight.data = old_data
            raise
    return len(pending)
