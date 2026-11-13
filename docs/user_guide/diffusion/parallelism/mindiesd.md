# MiniMax-H3 MindIE-SD acceleration

MiniMax-H3 can hand an entire sparse attention call to MindIE-SD before
Ulysses communication. MindIE-SD performs both all-to-all directions and the
attention kernel. Ordinary Ulysses still handles dense steps and declined
calls. This executor is separate from the Ulysses scatter/gather transport
hooks; it cannot be substituted for an individual all-to-all operation.

All model acceleration features default off. They are selected in Omni;
MindIE-SD receives explicit state and operator arguments. The SDK does not
import Omni model configuration.

## Requirements

The integrated public SDK contracts are available in MindIE-SD branch
`lly/minimax-h3-x8-test-archive-20261105`, commit
`1476bdfdc1e47f3d09953707ea385815bbeef6bf`. Install its Python package together
with the matching native plugin and OPP packages. A Python package version
alone does not establish that the required kernels were built.

With this SDK revision, F (`modulation_fusions`) uses an indexed gate kernel
with the legacy Triton-Ascend `tl.load(..., care_padding=False)` argument.
Triton-Ascend 3.6.0.post2 does not accept that argument. Leave F disabled on
that stack; the other features can run with the unmodified SDK. This integration
does not patch the SDK or substitute a different kernel when F is selected.

Use BF16 activations, one packed request, TP1, pure strict Ulysses, and
ordinary layerwise offload for the complete configuration. The H3 production
configuration uses eight Ascend 910B cards. Head-major QKV has a narrower
whitelist: 56 equal Q/K/V heads, D128, batch 1, pre-rearranged RF-v3 input,
and `(world, chunks) = (8, 7)` or `(4, 14)`.

Paged KV, reference-KV reuse, VDN, VSA gates, joint attention, Ring and
AllGather-KV retain their existing paths. RF reorder hoist rejects incompatible
execution before changing row labels. It supports the full request denoise
loop; disable it for multi-request continuous batching and attention schedules.

## Feature selection

Set `VLLM_OMNI_MINIMAX_H3_ACCELERATION_FEATURES` before worker startup. It
accepts a comma-separated set; selecting one feature does not select another.

| Point | Feature | Behavior |
| --- | --- | --- |
| A | `dual_stream_attention` | Full distributed sparse attention with overlapping communication |
| B | `attention_pipeline_optimizations` | SDK packing, pooling, mapping/chain/receive caches, and request-level RF reorder hoist |
| C | `weight_nz` | Finalized DiT linear weights in FRACTAL_NZ; host packing for ordinary offload |
| D | `eagle_bsa` | Eagle BF16 RF-v3 BSA consumer |
| E | `split_norm_rope_concat` | Existing SDK SNRC fused producer |
| F | `modulation_fusions` | Indexed gate, RMSNorm and modulation operators |
| G | `qbmm_swiglu` | INT8 fc1 QBMM plus SwiGLU; native dynamic quantization and fc2 remain |
| H | Separate head-major switch | SNRC BNSD storage and SDK contiguous round-robin send views |

```bash
export VLLM_OMNI_MINIMAX_H3_ACCELERATION_FEATURES=dual_stream_attention,attention_pipeline_optimizations,weight_nz,eagle_bsa,split_norm_rope_concat,qbmm_swiglu
export VLLM_OMNI_MINIMAX_H3_HEAD_MAJOR_QKV=1
export VLLM_OMNI_MINIMAX_H3_WEIGHT_NZ_MODE=reactor
export VLLM_OMNI_ULYSSES_HCCL_EXPANSION_MODE=2
```

The HCCL setting controls only Ulysses process groups: `2` selects AICPU and
`3` selects AIV; unset preserves the runtime default. Other communication
groups keep their configuration. H requires A, B and E together. Set
`VLLM_OMNI_MINIMAX_H3_RF_HOIST_REORDER=0` to disable only B's global reorder;
H must also be disabled in that configuration.

The example enables A–E, G and H. Add `modulation_fusions` only when the SDK's
indexed gate kernel is compatible with the installed Triton-Ascend version.

C supports `reactor` and `cast`. Reactor packs aligned CPU master weights once,
uploads their raw slots and constructs non-owning format-29 views. The offload
producer retains and records the underlying buffers on the consumer stream
until the block finishes. Cast creates an owning NZ allocation on upload.
Resident weights are cast after quantization finalization. FP32 model weights
are excluded. `H3_QBMM_WEIGHT_LAYOUT=host_nz|device_nz` conflicts with C.

For a baseline with A–H off:

```bash
export VLLM_OMNI_MINIMAX_H3_ACCELERATION_FEATURES=
export VLLM_OMNI_MINIMAX_H3_HEAD_MAJOR_QKV=0
unset VLLM_OMNI_ULYSSES_HCCL_EXPANSION_MODE
```

Keep the same INT8 quantization, request, BF16 attention precision, RF-v3
algorithm and sparsity in both arms:

```text
--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"start_step":0,"precision":"bf16","sparse_type":"rf_v3"}}}'
```

Use `num_inference_steps=8` for eight evaluations on this mainline. Do not reuse
the old runtime's nine-boundary request as evidence of an eight-step run.

## Typed parallel-only configuration

For the executor alone, without the model's C/E/F/G features, configure
`block_sparse.parallel` explicitly:

```json
{
  "default": {
    "backend": "RAINFUSION_ATTN",
    "block_sparse": {
      "sparsity": 0.8,
      "precision": "bf16",
      "sparse_type": "rf_v3",
      "parallel": {
        "overlap_comm": true,
        "optimize_attention_pipeline": true,
        "sparse_type": "rf_v3",
        "bsa_backend": "cann",
        "reorder_hoist": false,
        "head_major_qkv": false
      }
    }
  }
}
```

An absent `parallel` block leaves the executor disabled unless A or the legacy
explicit `block_sparse.overlap_comm` route is selected. An explicit block takes
precedence over the feature-derived parallel controls. All layers sharing one
communicator/device must use the same controls.

## Ownership and failure behavior

The SP coordinator owns a process-local resource registry. States survive
layer/step boundaries but never outlive coordinator teardown. Omni drains SDK
streams, drops its state references and collects cyclic chains before destroying
the process groups; this does not require an SDK close API. Failed drains retain
the resources and prevent process-group destruction.

Admission happens before SDK state allocation and collective submission.
After an admitted execution fails, the resource is poisoned and subsequent
calls fail. Retrying through ordinary Ulysses after a partially submitted
collective could desynchronize ranks and is never performed.

The producer and consumer share an immutable request-local `QKVLayoutPlan`.
The H switch is snapshotted at the request boundary; unsupported metadata takes
the established correct layout and is reported as a fallback rather than actual
head-major activation.

## Validation and measurement

`VLLM_OMNI_MINIMAX_H3_ACCELERATION_DIAGNOSTICS=1` reports successful request
counters in `[MINIMAX_H3_FEATURE_STATS]`. On the 50-block, NFE8 configuration,
A/D/E/F/G each execute 400 times per participating rank. H should report
400 head-major calls, 1,200 contiguous send views and 400 round-robin merges.
Inspect actual `last_call_stats`; a requested flag alone is not engagement.
For the F-disabled configuration, all F counters must be zero, while the
enabled A–E, G and H counters must still satisfy their contracts.

The SNRC, QBMM and head-major VERIFY switches are diagnostics and default off.
Disable them for formal timing. C/B/H storage operations require bit-equivalence
checks. A/D/E/F/G need numerical and complete media validation because their
execution or rounding can differ.

Measure both arms from the same source revision, dependency stack, weights,
input and hardware group. Each service window requires exactly two successful
complete warmups followed by formal requests. Report E2E, each rank's DiT wall
window and its eight-rank maximum, actual NFE, per-card AI Core frequencies,
and complete video/audio decoding. Recheck card ownership between windows.
CPU contract tests do not establish native kernel correctness or a speedup.

For aligned DiT frequency windows, enable
`VLLM_OMNI_MINIMAX_H3_ACCELERATION_TIMING=1` in both arms. It synchronizes at
the full denoise-loop boundaries and reports `[MINIMAX_H3_DIT_TIMING]` with
monotonic timestamps, rank and duration. With diagnostics enabled, `nfe`
counts successful transformer evaluations rather than inferring them from
the requested step count. This observer defaults off and must be symmetric
between the compared configurations.

### Recorded Ascend experiment

On main `d112af22e4e21af3976e6f83c342d77898a96e0f` plus this integration,
one Ref2VA case at 480×864, seed 42 and NFE8 was measured on eight Ascend 910B
cards. Both service windows completed two full warmups and three formal requests.
The candidate enabled A–E, G and H; F was disabled in both arms.

| Metric | All features off | A–E, G and H on | Time reduction |
| --- | --- | --- | --- |
| Mean E2E | 35.797 s | 34.931 s | 2.42% |
| Mean maximum-rank DiT | 21.609 s | 18.954 s | 12.29% |

This used vLLM 0.33.0, torch/torch_npu 2.10, CANN 9.0.1 and Triton-Ascend
3.6.0.post2, with a Python-only vLLM-Ascend installation and the SDK's native
operators. Complete media decoding and feature counters passed. The two
configurations produced different media: decoded video SSIM was 0.720 and
audio waveform cosine was 0.574. These timing observations do not establish
output equivalence or dataset-level quality acceptance. CUDA/ROCm and
FlashInfer RDMA were not hardware-tested by this experiment.
