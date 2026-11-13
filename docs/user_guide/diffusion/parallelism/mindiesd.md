# MiniMax-H3 MindIE-SD parallel attention

MiniMax-H3 can delegate an eligible complete distributed sparse-attention call
to MindIE-SD. MindIE-SD owns forward Q/K/V all-to-all, RF-v3 mask preparation,
BF16 block-sparse attention, reverse output all-to-all, and communication/
compute overlap. Ordinary Omni attention remains the path for calls declined
before communication.

This integration intentionally does not enable MindIE-SD fusion operators,
FRACTAL_NZ weights, quantized attention, Eagle BSA, or head-major QKV. It is a
complete executor, not a replacement for one Ulysses scatter/gather operation.

## Requirements

Use a MindIE-SD build from `dev` that exposes:

```python
from mindiesd import build_rf_attention_plan
from mindiesd.parallel import ParallelState, distributed_sparse_attention
```

The validated source identity must accompany any result; a package version by
itself does not prove that the matching native plugin and OPP were built.

The initial contract is deliberately narrow:

- Ascend NPU, BF16 Q/K/V, RF-v3, stock CANN BSA;
- batch size 1 with equal four-dimensional Q/K/V;
- strict pure Ulysses with exact head divisibility;
- no Ring, AllGather-KV, paged KV, reference-KV, joint attention, VDN,
  gate compression, or attention schedule; and
- one request per model forward when RF reorder hoist is enabled.

## Configuration

The presence of `block_sparse.parallel` explicitly enables the SDK executor.
No environment feature list or implicit installed-package route enables it.

```json
{
  "default": {
    "backend": "RAINFUSION_ATTN",
    "block_sparse": {
      "sparsity": 0.8,
      "start_step": 0,
      "precision": "bf16",
      "sparse_type": "rf_v3",
      "parallel": {
        "overlap_comm": true,
        "reorder_hoist": true
      }
    }
  }
}
```

`overlap_comm` selects the SDK communication-stream schedule. Set it to false
only for diagnosis against the SDK serial schedule. `reorder_hoist` computes
the RF row plan once before RoPE and updates MiniMax-H3's text, image, audio and
output position labels. MindIE-SD then skips per-layer row permutation and
preserves RF order in its returned local shard.

Omitting `parallel` leaves the complete executor disabled. The ordinary
RainFusion backend may still run after Omni's normal sequence-parallel exchange.

## Admission and fallback

Omni evaluates every supported condition before allocating SDK state or
submitting a collective. A declined original-order call uses the existing
attention path. Once a request has been relabeled into RF order, every main-DiT
attention layer must use the admitted executor; handing those rows to an
unaware fallback would be incorrect and raises instead.

An execution error poisons the shared resource and propagates. Omni never
replays ordinary Ulysses after a partially submitted SDK collective because
different ranks could enter incompatible collective sequences.

## State ownership

The sequence-parallel coordinator owns one resource per live Ulysses process
group and device. Its MindIE-SD `ParallelState` is reused across layers and
denoise steps, including RF mappings, chains, receive buffers, communication
stream and events. All layers sharing that resource must use the same overlap
mode.

Resources are drained and released before process-group destruction. The
current SDK has no public close method, so Omni synchronizes the device before
dropping the state. A future SDK close API can replace that compatibility
boundary without changing model admission.

## Validation

CPU admission tests prove configuration, decline-before-collective, state
reuse, poisoning, teardown order and row-label updates. They do not prove NPU
numerics or performance.

Native validation must compare the ordinary and SDK paths on the same source,
SDK build, model, input, precision, sparsity, NFE and card group. Each formal
service window requires two successful complete warmups before timing. Record
HTTP completion, per-rank actual NFE, complete video/audio decode, collective
engagement from `ParallelState.last_call_stats`, DiT/E2E latency and peak
memory. Do not reuse measurements from variants that also enabled fusion or NZ
features as the parallel-only result.
