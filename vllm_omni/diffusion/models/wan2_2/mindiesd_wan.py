# SPDX-License-Identifier: Apache-2.0
"""Optional Wan self-attention caller of the model-independent eager bridge.

VLLM_OMNI_WAN_MINDIESD_BRIDGE=1 opts into dense, unpadded strict Ulysses4.
An explicit request scope agrees once, including rejected native routes.
Unscoped model calls remain native and issue no new control collective.
"""

from contextlib import contextmanager
import math
import os

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.mindiesd_native import NativeBackendFactory
from vllm_omni.diffusion.attention.mindiesd_bridge import AttentionBridge, RouteContract
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_ulysses_mode


_MASK_FIELDS = (
    'attn_mask',
    'joint_attn_mask',
    'joint_query',
    'joint_key',
    'joint_value',
    'full_attn_spans',
    'query_ranges',
    'packed_padding',
)


class MindIESDWanParallel:
    """One persistent generic transport owner shared by all real Wan blocks."""

    def __init__(self):
        self.enabled = os.environ.get('VLLM_OMNI_WAN_MINDIESD_BRIDGE', '0') == '1'
        self._bridge = AttentionBridge(overlap=True, head_chunk_size=1)
        self._backends = NativeBackendFactory()
        self._request_id = 0
        self._layers = {}
        self._token_grid = None
        self._prepared_enabled = None
        self.calls = 0

    def prepare_request(self, *, plan_id, shape, used_len, device, dtype, attention_layers, token_grid, candidate=True):
        """All participating ranks call this once before any model block.

        Rejection is a shared native route, never an error recovery after A2A.
        No tensor contents or process-local addresses enter the agreement.
        """
        layers = tuple(attention_layers)
        token_grid, shape = tuple(token_grid), tuple(shape)
        world = dist.get_world_size(get_sp_group().ulysses_group)
        signatures = []
        supported = bool(layers)
        for layer in layers:
            try:
                signature = self._backends.contract(layer, declared=True)
                supported = supported and self._backends.supports(layer, declared=True)
                supported = supported and layer.attn_backend.get_name() == 'FLASH_ATTN'
            except (ValueError, TypeError, AttributeError):
                signature = ('unsupported consumer',)
                supported = False
            signatures.append(signature)
        accepted = bool(
            candidate
            and self.enabled
            and supported
            and world == 4
            and get_sp_group().ring_world_size == 1
            and get_ulysses_mode(default='strict') == 'strict'
            and getattr(device, 'type', None) == 'npu'
            and dtype == torch.bfloat16
            and len(shape) == 4
            and all(isinstance(n, int) and n > 0 for n in shape)
            and shape[2] % world == 0
            and used_len == shape[1] * world
            and len(token_grid) == 3
            and all(isinstance(n, int) and n > 0 for n in token_grid)
            and math.prod(token_grid) == used_len
        )
        contract = RouteContract(
            shape,
            str(dtype),
            used_len,
            ('wan dense self', token_grid, tuple(signatures), 'strict unpadded TP1 PP1 CFG1'),
        )
        result = self._bridge.prepare_request(
            plan_id,
            contract,
            group=get_sp_group().ulysses_group,
            device=device,
            candidate=accepted,
        )
        self._layers = {id(layer): signature for layer, signature in zip(layers, signatures)}
        self._token_grid = token_grid
        self._prepared_enabled = self.enabled
        return result

    def _validate_metadata(self, metadata):
        if metadata is None or any(getattr(metadata, name, None) is not None for name in _MASK_FIELDS):
            raise RuntimeError('Prepared Wan dense route cannot acquire masks, joint tokens or padding')
        if metadata.extra.keys() - {'vsa_dit_seq_shape', 'kv_cache_dtype'}:
            raise RuntimeError('Prepared Wan dense route acquired unsupported native metadata')
        if metadata.extra.get('kv_cache_dtype') not in (None, 'float', 'auto'):
            raise RuntimeError('Prepared Wan dense route acquired quantized KV')
        if tuple(metadata.extra.get('vsa_dit_seq_shape', ())) != self._token_grid:
            raise RuntimeError('Wan token grid differs from its prepared request')
        layout = metadata.video_layout
        if (
            layout is None
            or layout.prefix_len != 0
            or tuple(layout.latent_grid) != self._token_grid
            or layout.used_len != self._bridge.contract.used_len
        ):
            raise RuntimeError('Wan native video layout differs from its prepared request')

    @torch.compiler.disable
    def try_attention(self, attention, query, key, value, metadata):
        if self._bridge.failed:
            raise RuntimeError('Failed Wan bridge: terminate participating workers')
        if not self._bridge.prepared:
            return None
        try:
            if self.enabled != self._prepared_enabled:
                raise RuntimeError('Wan enable policy changed after request agreement')
            if not self._bridge.accepted:
                return None
            if (
                id(attention) not in self._layers
                or self._backends.contract(attention, declared=True) != self._layers[id(attention)]
            ):
                raise RuntimeError('Wan consumer changed after request agreement')
            active = attention._get_active_parallel_strategy()
            if not active.enabled or active.name != 'ulysses' or get_ulysses_mode(default='strict') != 'strict':
                raise RuntimeError('Prepared Wan route lost its active strict Ulysses strategy')
            # Local metadata is used directly by native compute after the
            # bridge exchanges QKV. Never call the full Attention layer there.
            local = attention._with_kv_cache_dtype(metadata)
            self._validate_metadata(local)
            if query.shape != key.shape or query.shape != value.shape:
                raise RuntimeError('Wan bridge only supports equal-length self-attention QKV')
            backend = self._backends.build(attention, local)
            result, _ = self._bridge.run(query, key, value, backend=backend, contract=self._bridge.contract)
        except BaseException:
            self._bridge.failed = True
            raise
        self.calls += 1
        return result

    def finish_request(self):
        self._bridge.finish_request()
        self._layers.clear()
        self._token_grid = self._prepared_enabled = None


@contextmanager
def prepare_wan_request(model, *, latents, single_model=True, cfg_parallel_size=1):
    """Explicit full-transformer boundary; unscoped pipeline calls stay native.

    Latents and conditioning stay owned by the caller; no current activations
    are saved. Rejected geometry stays native for the entire request. On a
    non-NPU entry point this NPU-only adapter is inactive.
    """
    owner = getattr(model, '_mindiesd_wan', None)
    if owner is None or latents.device.type != 'npu':
        yield
        return
    blocks = tuple(model.blocks)
    complete_model = bool(
        blocks
        and getattr(model, 'start_layer', 0) == 0
        and getattr(model, 'end_layer', len(blocks)) == len(blocks)
        and all(hasattr(block, 'attn1') for block in blocks)
    )
    # PP partitions contain PPMissingLayer placeholders. Reject before
    # dereferencing them, while still entering the shared route agreement.
    attentions = tuple(block.attn1.attn for block in blocks) if complete_model else ()
    first = blocks[0].attn1 if complete_model else None
    patch = model.config.patch_size
    grid = tuple(size // p for size, p in zip(latents.shape[2:], patch))
    used = math.prod(grid)
    world = dist.get_world_size(get_sp_group().ulysses_group)
    shape = (
        latents.shape[0],
        (used + world - 1) // world,
        first.num_heads if first is not None else 1,
        first.head_dim if first is not None else 1,
    )
    candidate = bool(
        single_model
        and cfg_parallel_size == 1
        and complete_model
        and first.num_heads == getattr(model.config, 'num_attention_heads', first.num_heads)
        and all(size % p == 0 for size, p in zip(latents.shape[2:], patch))
        and all(block.attn1.num_heads == shape[2] and block.attn1.head_dim == shape[3] for block in blocks)
    )
    owner._request_id += 1
    owner.prepare_request(
        plan_id=owner._request_id,
        shape=shape,
        used_len=used,
        device=latents.device,
        dtype=latents.dtype,
        attention_layers=attentions,
        token_grid=grid,
        candidate=candidate,
    )
    try:
        yield
    except BaseException:
        owner._bridge.failed = True
        raise
    finally:
        owner.finish_request()
