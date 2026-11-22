# Layerwise Offloading

Layerwise, or blockwise, offloading keeps one transformer block on the
accelerator and prefetches the next block while the current block computes.
It is best suited to compute-heavy video DiTs whose block execution can hide
host-to-device transfers.

## Execution flow

Each block has a pre-forward and post-forward hook. Parameters are consolidated
in pinned host tensors and rematerialized for execution on a dedicated copy
stream.

| Block | Pre-forward hook | Forward | Post-forward hook |
| --- | --- | --- | --- |
| block 0 | Prefetch block 1 | Compute block 0 | Free block 0 |
| block 1 | Prefetch block 2 | Compute block 1 | Free block 1 |
| ... | ... | ... | ... |
| last block | Prefetch block 0 | Compute last block | Free last block |

Selected, plan-declared text-encoder blocks can use the same rank-local
streaming mechanism. Image encoders, VAEs, and non-block DiT modules remain
device resident.

## Usage

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    diffusion_offload_config={
        "mode": "layer",
        "components": ["dit"],
    },
)
```

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --diffusion-offload-config '{"mode":"layer","components":["dit"]}'
```

## Optional AdaLN weight cache

`--cache-adaln` keeps the loaded DiT tensors in AdaLN-named modules on the
device while ordinary layerwise offload continues to stream other weights.
It is disabled by default. Use `--no-cache-adaln` to explicitly disable it.
The same Python/stage-config fields are `cache_adaln`,
`adaln_cache_max_gib`, and `adaln_cache_reserve_gib`.

```bash
vllm serve /path/to/model --omni --enable-layerwise-offload \
  --cache-adaln --adaln-cache-max-gib 25 --adaln-cache-reserve-gib 8

vllm serve /path/to/model --omni --enable-layerwise-offload --no-cache-adaln
```

It also works with `--diffusion-offload-config
'{"mode":"layer","components":["dit"]}'`. Model-level offload, distributed
layerwise offload, and an encoder-only offload selection reject this option.

The per-device cache limit defaults to 25 GiB. Admission also requires at
least 8 GiB free after allocation, configurable with
`--adaln-cache-reserve-gib`. Exceeding a limit, unavailable memory telemetry,
unsupported tensors, or cache allocation/copy failure leaves ordinary LO
active. Logs state whether caching was enabled and the fallback reason.
The reserve is a startup admission check, not an upper bound on future
activation or VAE memory; choose it for the workload and disable the cache
when additional headroom is needed.

Weights, biases, and quantization buffers/scales retain their loaded dtype and
stride. The cache policy is independent of attention precision. MiniMax-H3's
RF-v2 BF16/FP16 and RF-v3 BF16/FP16, `mix` (A5 Eagle QBSA), and `mxfp8` (A5
CANN) accept the same policy. INT8/FP8 weights and E8M0 scale storage are not
converted to BF16. A5 still requires its attention runtime and operators;
CPU config/storage checks do not establish A5 device precision, memory
capacity, or performance acceptance.

MiniMax-H3 keeps its original 101-tensor cache scope for the current INT8
checkpoint. The final AdaLN projection's `linear.weight` and `weight_scale`
already remain on device outside the streamed blocks and are excluded from
this cache; its BF16 bias remains included. Quantized AdaLN weights and scales
inside transformer blocks continue to retain their loaded representations.

This option caches AdaLN only; it does not enable the retired additional
MLP fc1 cache or per-request weight staging experiment.

## NPU CPU locality

For the MindIE-SD 16-device Ulysses-8 x CP-2 path, with NPU layerwise offload
and `cache_adaln` enabled, the loader automatically limits the worker's CPU
affinity to allowed CPUs on the target NPU's NUMA node before loading weights.
Native Omni and SD8 keep their incoming affinity by default.
This helps host allocation and weight-transfer locality
when DiT compute becomes too short to hide H2D transfers. Physical device
visibility and PCIe topology determine the node; affinity never expands beyond
the incoming CPU allowance. Tensor dtype, layout, cache size and CPU recovery
are unchanged.

Use the existing public `additional_config` option to override this policy:

```bash
# Preserve the worker's incoming CPU affinity.
vllm serve /path/to/model --omni --enable-layerwise-offload --cache-adaln \
  --additional-config '{"enable_cpu_binding":false}'

# Require successful NPU CPU binding.
vllm serve /path/to/model --omni --enable-layerwise-offload --cache-adaln \
  --additional-config '{"enable_cpu_binding":true}'
```

The Python/stage-config equivalent is
`additional_config={"enable_cpu_binding": False}` or `True`. The automatic
policy preserves incoming affinity if topology or permitted local CPUs are
unavailable; explicit `True` reports a binding failure. Binding only affects
the current worker process and does not migrate process-wide NUMA pages or
change device IRQ affinity.

## Component selection

Add `dit`, `text_encoder`, or both to the `components` list. Omitting
`layer_options` uses safe rank-local defaults:

- `["dit"]` streams only DiT blocks.
- `["text_encoder"]` streams only declared text-encoder blocks.
- Listing both streams both components.

```bash
# DiT-only layer offload
vllm serve /path/to/model --omni \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit"]}'

# Stream a model-declared text encoder while keeping the DiT resident
vllm serve /path/to/model --omni \
  --diffusion-offload-config \
  '{"mode":"layer","components":["text_encoder"]}'
```

Encoder categories are resolved from `OffloadPlan.encoder_component_types`
first. A name-based fallback is retained for pipelines that predate
`OffloadPlan`.

## Model integration

Transformer classes declare containers of executable blocks:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]


class Flux2Transformer2DModel(nn.Module):
    _layerwise_offload_blocks_attrs = [
        "transformer_blocks",
        "single_transformer_blocks",
    ]
```

Auxiliary components use declarative pipeline metadata:

```python
from vllm_omni.diffusion.offloader import OffloadPlan


class MyPipeline(nn.Module):
    _encoder_modules = ["prompt_model"]
    _offload_plan = OffloadPlan(
        encoder_component_types={"prompt_model": "text_encoder"},
        encoder_block_attrs={"prompt_model": ("encoder.layers",)},
    )
```

See the [layerwise design](../../../design/feature/offloader/layerwise_offload.md)
for the discovery and hook invariants. Both ordinary and distributed layerwise
offload consume the same `OffloadPlan` metadata.

## Limitations

- The default weight transfer is `rank-local`. Set a selected component's
  `weight_transfer` to `allgather` to shard its host weights across a
  compatible multi-device group; backend selection is automatic.
- Setup consolidates and pins block parameters, increasing cold-start time.
- Performance depends on block compute time and host-to-device bandwidth;
  lightweight blocks may not hide transfers.
