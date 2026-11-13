# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generic FRACTAL_NZ primitives with explicit storage ownership.

Host packing allocates a complete CPU slot. Cast allocates owning format-29
storage. nz_view_over_buffer is non-owning: the caller must retain its source
buffer until the last device consumer finishes. Probes run only on request;
importing this module and packing on the host never imports torch_npu.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

_ACL_FORMAT_FRACTAL_NZ = 29
_NZ_BLOCK = {torch.bfloat16: (16, 16), torch.float16: (16, 16), torch.int8: (16, 32)}


def _npu_format(weight: torch.Tensor) -> int:
    import torch_npu

    return int(torch_npu.get_npu_format(weight))


def _is_npu_weight(weight: torch.Tensor) -> bool:
    return weight.device.type == "npu"


def maybe_cast_weight_to_nz(weight: torch.Tensor) -> torch.Tensor:
    """Cast a selected weight, raising if format 29 cannot be installed.

    Callers select by module and quant method before reaching this function.
    A fresh allocation is required: casting a non-owning contiguous upload view
    directly can retain ND storage despite the requested format.
    """
    if weight.dim() != 2 or weight.dtype not in _NZ_BLOCK or not _is_npu_weight(weight):
        raise ValueError("NZ cast requires a selected 2-D BF16, FP16 or INT8 NPU weight")
    import torch_npu

    out = torch_npu.empty_with_format(
        tuple(weight.shape), dtype=weight.dtype, device=weight.device, acl_format=_ACL_FORMAT_FRACTAL_NZ
    )
    out.copy_(weight)
    return out


def nz_reactor_can_hold(shape: Sequence[int], stride: Sequence[int], dtype: torch.dtype) -> bool:
    """Whether an unpadded NZ weight fits this contiguous 2-D ND slot."""
    block = _NZ_BLOCK.get(dtype)
    if block is None or len(shape) != 2:
        return False
    n, k = shape
    return n > 0 and k > 0 and tuple(stride) == (k, 1) and n % block[0] == 0 and k % block[1] == 0


def permute_host_weight_to_nz(weight: torch.Tensor) -> torch.Tensor:
    """Pack the complete CPU ``[N,K]`` tensor as ``[K1,N1,N0,K0]``."""
    if weight.device.type != "cpu" or not nz_reactor_can_hold(weight.shape, weight.stride(), weight.dtype):
        raise ValueError("NZ reactor requires a contiguous CPU weight with aligned fractal blocks")
    block_n, block_k = _NZ_BLOCK[weight.dtype]
    n, k = weight.shape
    return weight.reshape(n // block_n, block_n, k // block_k, block_k).permute(2, 0, 1, 3).contiguous().reshape(-1)


def unpermute_host_weight_from_nz(raw: torch.Tensor, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """Restore every CPU slot element to a contiguous logical ND weight."""
    if len(shape) != 2:
        raise ValueError("NZ reactor restoration requires a 2-D shape")
    n, k = shape
    if (
        raw.device.type != "cpu"
        or raw.dtype != dtype
        or raw.numel() != n * k
        or not raw.is_contiguous()
        or not nz_reactor_can_hold(shape, (k, 1), dtype)
    ):
        raise ValueError("NZ reactor restoration requires the complete aligned CPU slot")
    block_n, block_k = _NZ_BLOCK[dtype]
    return raw.reshape(k // block_k, n // block_n, block_n, block_k).permute(1, 2, 0, 3).contiguous().reshape(n, k)


def nz_view_over_buffer(buffer: torch.Tensor, offset: int, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """Create a non-owning NZ view; the backend must retain ``buffer``."""
    if len(shape) != 2:
        raise ValueError("NZ reactor requires a 2-D slot")
    n, k = shape
    if not nz_reactor_can_hold(shape, (k, 1), dtype):
        raise ValueError("NZ reactor slot must fit aligned fractal blocks without padding")
    if buffer.dtype != dtype or not buffer.is_contiguous() or offset < 0 or offset + n * k > buffer.numel():
        raise ValueError("NZ reactor slot does not fit the source buffer")
    import torch_npu

    metadata = {
        "size": (n, k),
        "stride": (k, 1),
        "nbytes": n * k * dtype.itemsize,
        "dtype": dtype,
        "data_ptr": buffer.data_ptr() + offset * dtype.itemsize,
        "storage_offset": 0,
        "device": buffer.device,
        "npu_format": _ACL_FORMAT_FRACTAL_NZ,
        "layout": torch.strided,
        "requires_grad": False,
    }
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)


def validate_weight_nz(mode: str) -> None:
    """Probe required APIs and format 29 once when installing a backend.

    The caller owns the setup lifetime; uploads never repeat this allocation.
    There is deliberately no reactor-to-cast or NZ-to-ND fallback.
    """
    import torch_npu

    if mode not in ("reactor", "cast"):
        raise ValueError(f"NZ mode must be reactor or cast, got {mode!r}")
    # torch_npu's config exposes a setter only. Read the corresponding backend
    # option instead of accessing config.allow_internal_format as an attribute.
    get_option = getattr(getattr(torch_npu, "_C", None), "_npu_getOption", None)
    if not callable(get_option):
        raise RuntimeError("weight NZ requires torch_npu._C._npu_getOption")
    internal_format = get_option("ALLOW_INTERNAL_FORMAT")  # pylint: disable=not-callable  # Dynamic C API, guarded above.
    if internal_format != b"enable":
        raise RuntimeError(
            f"weight NZ requires torch.npu.config.allow_internal_format=True; ALLOW_INTERNAL_FORMAT={internal_format!r}"
        )
    for name in ("empty_with_format", "get_npu_format"):
        if not callable(getattr(torch_npu, name, None)):
            raise RuntimeError(f"weight NZ requires torch_npu.{name}")
    if mode == "reactor":
        for name in ("_construct_storage_from_data_pointer", "_construct_NPU_Tensor_From_Storage_And_Metadata"):
            if not callable(getattr(getattr(torch_npu, "_C", None), name, None)):
                raise RuntimeError(f"NZ reactor requires torch_npu._C.{name}")
    for dtype in _NZ_BLOCK:
        if mode == "cast":
            probe = torch_npu.empty_with_format((16, 32), dtype=dtype, device="npu", acl_format=29)
        else:
            buffer = torch_npu.empty_with_format((16 * 32,), dtype=dtype, device="npu", acl_format=2)
            probe = nz_view_over_buffer(buffer, 0, (16, 32), dtype)
        if _npu_format(probe) != _ACL_FORMAT_FRACTAL_NZ:
            raise RuntimeError(f"weight NZ {mode} probe did not produce format 29 for {dtype}")


def get_npu_format(weight: torch.Tensor) -> int:
    """Read the existing backend format code on explicit caller request."""
    return _npu_format(weight)


__all__ = [
    "get_npu_format",
    "maybe_cast_weight_to_nz",
    "nz_reactor_can_hold",
    "permute_host_weight_to_nz",
    "unpermute_host_weight_from_nz",
    "nz_view_over_buffer",
    "validate_weight_nz",
]
