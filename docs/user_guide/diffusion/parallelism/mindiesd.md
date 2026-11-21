# MiniMax-H3 MindIE-SD parallel attention

MindIE-SD owns Q/K/V prepacking, forward/reverse all-to-all, sparse pooling/mask
preparation, sparse attention and communication/compute overlap. Omni supplies
request metadata, admission and state lifetime; admitted calls bypass native
Ulysses communication. This integration enables parallelism and RF reorder
hoist. Sparse provider selection follows the four paths below.

## Requirements and configuration

Use a matching MindIE-SD `lly/minimax-h3-u8-kvgather2-20261110` build exposing `mindiesd.build_rf_attention_plan` and
`mindiesd.parallel.{ParallelState, distributed_sparse_attention}`, with matching
native plugin/OPP. The distributed executor supports exactly these paths:

| Path | `sparse_type` | `precision` | `bsa_backend` | Communication | Hardware |
| --- | --- | --- | --- | --- | --- |
| RainFusionAttention | `rf_v2` | `bf16` / `fp16` | `cann` | BF16 / FP16 Q/K/V | A2 / A3 |
| Eagle BSA | `rf_v3` | `bf16` / `fp16` | `eagle` | BF16 / FP16 Q/K/V | A2 / A3 |
| Eagle QBSA | `rf_v3` | `mix` | `eagle` | INT8 Q/K, FP8 V | A5 |
| Native MXFP8 BSA | `rf_v3` | `mxfp8` | `cann` | MXFP8 Q/K/V and E8M0 scales | A5 |

Omitting `bsa_backend` resolves the provider shown above. A conflicting explicit
provider or another algorithm/precision tuple raises at configuration time.
Admission requires Ascend NPU, equal `[1,S_local,H,D]` Q/K/V, strict Ulysses and
heads divisible by its degree. FP16 converts BF16 producer operands explicitly
and restores the producer dtype; quantized paths consume BF16 producer tensors.
Pure Ulysses passes `cp_group=None` and accepts either RF input order.
CP x Ulysses uses the existing `ulysses_allgather_kv` strategy and passes the
coordinator's orthogonal `allgather_group` as the actual `cp_group`. It requires
`pre_rearranged=True`, Ulysses degree 4 or 8, CP degree 2 or 4, global sequence
world 8, 16 or 32, and D128. MXFP8 also requires D128 in pure Ulysses.
Ring, paged/reference KV, joint attention, VDN and gate compression are unsupported.
RF reorder hoist also requires every main-DiT layer to use compatible settings,
no attention schedule, and one request per forward.

Enable the executor through `block_sparse.parallel`:

```json
{
  "default": {
    "backend": "RAINFUSION_ATTN",
    "block_sparse": {
      "sparsity": 0.8,
      "start_step": 0,
      "precision": "bf16",
      "sparse_type": "rf_v3",
      "bsa_backend": "eagle",
      "parallel": {
        "overlap_comm": true,
        "reorder_hoist": true,
        "optimize_attention_pipeline": true,
        "chunk_ready_pipeline": true
      }
    }
  }
}
```

Omit `parallel` to retain ordinary Omni attention. `overlap_comm=false` selects
the SDK serial schedule for diagnosis. All four controls default to true.
`optimize_attention_pipeline` enables SDK preparation/buffer reuse and
`chunk_ready_pipeline` enables its chunk scheduling; both can be disabled explicitly. `reorder_hoist` builds the RF plan once
before RoPE, relabels text/image/audio/output positions and lets SD skip per-layer
row permutation while preserving RF order. Sparse step/layer gates still apply.

To select the AICPU communication mode, set
`VLLM_OMNI_ULYSSES_HCCL_EXPANSION_MODE=2` before startup. It selects the HCCL
engine for both Ulysses and AllGather-KV (CP) groups. Both paths therefore use
AICPU rather than leaving CP on the global default engine. Verify all
participating ranks' Ulysses and AllGather-KV group logs.

## Execution and state ownership

Unsupported original-order calls decline before SDK state allocation or
communication. A rejected RF-reordered sparse call raises to prevent an unaware
fallback. Execution failures poison the shared resource and propagate without
collective retry.

The SP coordinator owns one `ParallelState` per Ulysses group, CP group, device,
algorithm, precision and provider, reusing
SDK mappings, chains, receive buffers, stream and events across layers/steps.
Calls must be serialized and share the same parallel controls. Before group teardown,
Omni synchronizes the device and releases state; the SDK has no public close API.

Configuration lives beside `BlockSparseSpec` in `data.py`; executor resources
live with their owner in `group_coordinator.py`. The two H3 attention helpers
handle per-layer SDK execution and request-level reorder, sharing one Ulysses
topology check. Other SDK execution paths are outside this integration.

## Validation

CPU contracts cover admission, configuration, state reuse/failure/teardown and
row labels across all four paths. A2/A3 paths require native numerical and
performance validation; A5-only paths may be qualified by CPU contracts separately
when A5 hardware is unavailable, without claiming native validation. Native A/B tests must match source, SDK build, model, input,
precision, sparsity, NFE and cards, with two complete warmups before formal
timing. Record HTTP/media completion, actual NFE, SDK engagement, Diffuse/E2E
latency and measured peak memory. CPU tests do not establish native numerics;
fusion/NZ measurements do not establish this parallel-only path's performance.
