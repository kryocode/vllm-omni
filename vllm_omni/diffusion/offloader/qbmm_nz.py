# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional MindIE-owned INT8 NZ packing for the DLO shard transport.

This is an isolated fork adapter. Model quantization and QBMM calls remain with
their existing method. Setup freezes the layout: rollback requires worker restart
with --dlo-weight-format nd. No request-time reinterpretation or device cast fallback.
"""

import logging
import os
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


class QbmmNZOffloadPlan:
    """One block's host layout and two slots of explicitly owned NZ views."""

    def __init__(self, targets, names, *, library=None):
        if library is None:
            from mindiesd.quantization import qbmm_nz as library
        self.library = library
        self.explicit_policy = False
        self.targets = targets
        self.names = frozenset(names)
        self.specs = {}
        self.skipped = {}
        self.slots = {}
        self.failed = False
        self.packed_weights = 0
        self.view_count = 0
        self.bind_count = 0
        self.request_owner = None
        self.block = None
        self.contract = None
        self.agreement_group = None
        self.transport_group = None
        self.restoring = False

    def _enabled(self):
        if self.explicit_policy:
            return self.library.qbmm_nz_enabled(requested=True)
        return self.library.qbmm_nz_enabled()

    def _assert_scope(self, *, historical=False):
        owner = self.request_owner
        if owner is None:
            # Mapping-only fixtures construct this internal plan directly.
            # The DLO factory always attaches its actual request owner/group.
            return
        if self.contract is None or getattr(self.block, "_mindiesd_qbmm_nz_contract", None) != self.contract:
            raise RuntimeError("QBMM NZ block contract changed after setup")
        owner.assert_weight_layout_live(self.contract, self.agreement_group, historical=historical)

    @contextmanager
    def restore_scope(self, *, drained):
        """Teardown only; no first-request bootstrap collective is permitted."""
        if not drained:
            raise RuntimeError("QBMM NZ restoration requires a prior healthy stream drain")
        if self.failed or self.restoring or self.request_owner is None:
            raise RuntimeError("failed or unauthorized QBMM NZ restoration; restart workers")
        if not self._enabled():
            raise RuntimeError("QBMM NZ configuration changed before restoration; restart workers")
        self._assert_scope(historical=True)
        self.restoring = True
        try:
            yield
        finally:
            self.restoring = False

    def describe(self, name, value, *, offset):
        """Called with the actual runtime-layout value, before copying any bytes."""
        if name not in self.names:
            return False
        try:
            if offset % 32:
                raise ValueError("offset is not aligned to 32 bytes")
            spec = self.library.QbmmNZSpec.from_tensor(value)
        except ValueError as error:
            self.skipped[name] = str(error)
            self.specs.pop(name, None)
            logger.info("MindIE QBMM NZ keeping %s ND: %s", name, error)
            return False
        self.specs[name] = spec
        return True

    def pack(self, name, value):
        spec = self.specs[name]
        # Setup only. A loader may have quantized on NPU before handing DLO its
        # master; host permutation is still performed exactly once per shard.
        result = self.library.pack_host_weight(value.detach().cpu(), spec)
        self.packed_weights += 1
        return result

    def prepare_bind(self, slot, buffers, metadata, *, generation, compute_stream, transfer_streams=()):
        """Validate/cache/bind before this prefetch submits ANY H2D or AllGather."""
        if self.failed:
            raise RuntimeError("failed QBMM NZ plan cannot be retried; recreate workers")
        try:
            # CPU master shards already contain NZ bytes. Toggling the flag
            # cannot turn those bytes back into ND; require a fresh setup.
            if not self._enabled():
                raise RuntimeError("QBMM NZ configuration changed; restart workers with the desired flag")
            self._assert_scope(historical=self.restoring)
            if self.contract is not None:
                actual = tuple(
                    (meta["name"], meta["offset"], tuple(meta["shape"]), tuple(meta["stride"]))
                    for metas in metadata.values()
                    for meta in metas
                    if meta.get("qbmm_nz")
                )
                if actual != self.contract[2]:
                    raise RuntimeError("QBMM NZ metadata contract changed after setup")
            entries = [(dtype, meta) for dtype, metas in metadata.items() for meta in metas if meta.get("qbmm_nz")]
            identity = tuple((dtype, id(buffers[dtype]), buffers[dtype].data_ptr()) for dtype, _ in entries)
            previous = self.slots.get(slot)
            if previous is not None:
                if previous[0] != identity:
                    raise RuntimeError("QBMM NZ slot buffer changed after descriptor construction")
                views = previous[1]
            else:
                # Construct the WHOLE slot before exposing any view. A private
                # API failure aborts before transfers and never falls back.
                views = {
                    meta["name"]: self.library.NZBufferView(buffers[dtype], meta["offset"], self.specs[meta["name"]])
                    for dtype, meta in entries
                }
                for view in views.values():
                    for stream in transfer_streams:
                        if stream is not None:
                            view.owner.record_stream(stream)
                self.slots[slot] = identity, views
                self.view_count += len(views)
                logger.info("MindIE QBMM NZ prepared slot=%d weights=%d; no device transdata", slot, len(views))
            selected = set()
            for name, view in views.items():
                target = self.targets[name]
                view.bind_to(target, generation=generation, stream=compute_stream)
                selected.add(id(target))
                self.bind_count += 1
            return selected
        except BaseException:
            self.failed = True
            raise

    def restore(self, name, packed):
        """Explicit inverse from raw flat bytes; never rely on an NZ .cpu cast."""
        return self.library.unpack_host_weight(packed, self.specs[name])

    def release_target(self, target):
        self.library.release_target_owner(target)

    def stats(self):
        return dict(
            selected_weights=len(self.specs),
            packed_weights=self.packed_weights,
            view_count=self.view_count,
            bind_count=self.bind_count,
            skipped=dict(self.skipped),
            shapes={name: dict(shape=spec.shape, stride=spec.source_stride) for name, spec in self.specs.items()},
        )


def validate_agreement_participants(transport_group, agreement_group, *, dp_size):
    """The existing Ulysses agreement must cover this exact ordered AG group."""
    if dp_size <= 1:
        return ()
    try:
        transport = tuple(torch.distributed.get_process_group_ranks(transport_group))
        agreement = tuple(torch.distributed.get_process_group_ranks(agreement_group))
    except Exception as error:
        raise RuntimeError("cannot validate QBMM NZ DLO/attention participants before packing") from error
    if len(transport) != dp_size or transport != agreement:
        raise RuntimeError(
            "QBMM NZ DLO ordered participants differ from the shared Ulysses agreement; unsupported topology"
        )
    return transport


def create_qbmm_nz_plan(block, *, device, rank_local_mmap, dp_group=None, dp_size=1):
    """Select semantic W8A8 weights, excluding scales, shared storage and DTensor."""
    owner_factory = getattr(block, "qbmm_nz_request_owner", None)
    if owner_factory is None:
        return None  # Non-H3/refiner blocks never consume this model's NZ setting.
    if not callable(owner_factory):
        raise RuntimeError("QBMM NZ H3 request owner protocol must be callable")
    policy_owner = owner_factory()
    config = getattr(policy_owner, "optimization_config", None)
    explicit = config is not None and config.isolated
    requested = config.value("MINDIESD_QBMM_NZ") if explicit else os.environ.get("MINDIESD_QBMM_NZ", "0")
    if requested == "0":
        return None
    if requested != "1":
        raise ValueError("MINDIESD_QBMM_NZ must be 0 or 1")
    missing_protocol = object()
    request_owner = getattr(block, "qbmm_nz_request_owner", missing_protocol)
    if request_owner is missing_protocol:
        # Token-refiner/encoder groups share DLO but not the main DiT request
        # protocol. Exclude them before importing NZ capabilities or packing.
        logger.info("MindIE QBMM NZ excluding non-target block %s; keeping native ND", type(block).__name__)
        return None
    if not callable(request_owner):
        raise RuntimeError("QBMM NZ H3 request owner protocol must be callable")
    owner = request_owner()
    setup = getattr(owner, "validate_weight_layout_setup", None)
    validate = getattr(owner, "assert_weight_layout_live", None)
    if not callable(setup) or not callable(validate):
        raise RuntimeError("QBMM NZ request owner must provide setup and live contract validation")
    setup()
    if explicit and (device.type != "npu" or rank_local_mmap):
        raise RuntimeError("Explicit --dlo-weight-format nz requires NPU DiT AllGather storage")
    from mindiesd.quantization import qbmm_nz

    if not (qbmm_nz.qbmm_nz_enabled(requested=True) if explicit else qbmm_nz.qbmm_nz_enabled()):
        return None
    if device.type != "npu" or rank_local_mmap or not qbmm_nz.reactor_available():
        if explicit:
            raise RuntimeError(
                "Explicit --dlo-weight-format nz requires supported NPU NZ reactor and AllGather storage"
            )
        logger.warning(
            "MindIE QBMM NZ unavailable before sharding (device=%s mmap=%s private=%s); keeping ND",
            device,
            rank_local_mmap,
            qbmm_nz.reactor_available(),
        )
        return None
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    agreement_group = get_sp_group().ulysses_group
    validate_agreement_participants(dp_group, agreement_group, dp_size=dp_size)
    from mindiesd.quantization.layer import W8A8QuantLinear
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

    targets = dict(block.named_parameters())
    targets.update(dict(block.named_buffers()))
    storage_users = {}
    aliased = set()
    for name, value in list(block.named_parameters(remove_duplicate=False)) + list(
        block.named_buffers(remove_duplicate=False)
    ):
        if value.is_meta or hasattr(value, "to_local"):
            continue
        key = (str(value.device), value.untyped_storage().data_ptr())
        if key in storage_users:
            aliased.update((name, storage_users[key]))
        else:
            storage_users[key] = name
    names = set()
    for prefix, module in block.named_modules():
        vllm_w8a8 = isinstance(module, LinearBase) and isinstance(
            getattr(module, "quant_method", None), NPUInt8LinearMethod
        )
        mindie_w8a8 = isinstance(module, W8A8QuantLinear) and module.is_dynamic
        if not (vllm_w8a8 or mindie_w8a8):
            continue
        name = f"{prefix}.weight" if prefix else "weight"
        weight = targets.get(name)
        if weight is None or name in aliased or hasattr(weight, "to_local") or weight.dtype != torch.int8:
            continue
        names.add(name)
    if not names:
        if explicit:
            raise RuntimeError("Explicit --dlo-weight-format nz found no eligible main-DiT INT8 W8A8 weights")
        logger.warning("MindIE QBMM NZ requested but no eligible semantic int8 W8A8 weights were found; keeping ND")
        return None
    logger.info("MindIE QBMM NZ setup: candidate_weights=%d; immutable until worker restart", len(names))
    plan = QbmmNZOffloadPlan(targets, names, library=qbmm_nz)
    plan.explicit_policy = explicit
    plan.request_owner = owner
    plan.block = block
    plan.agreement_group = agreement_group
    plan.transport_group = dp_group
    return plan
