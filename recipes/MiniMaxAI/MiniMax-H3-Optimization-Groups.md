# MiniMax H3 optimization groups

Configure the six options below on `vllm serve`. The resolved
`OmniDiffusionConfig` is passed to H3 before worker execution, where
`H3OptimizationConfig` constructs explicit policies for the executor, RF,
Graph and DLO. This path does not write or reread optimization environment
variables. These options do not change model precision, sparsity, denoising
steps, device count, or the existing DLO enablement flag.

Names follow the functional owner: `diffusion-attn-` for attention,
`head-overlap` for its communication and Graph paths, `minimax-h3-` for
model-specific row placement, and `dlo-` for AllGather gating and weight
format. Enum selectors use `-mode`, distinguishing them from the colleague's
integer head-group count and boolean Graph switches. Values, defaults and
execution policies are unchanged by this rename.

| CLI option | Values | Default | Behavior |
| --- | --- | --- | --- |
| `--diffusion-attn-head-overlap-mode` | `legacy`, `pipeline` | `legacy` | Original framework attention route, or prepared one-head communication pipeline with prepacked sends and caller-owned output storage. |
| `--diffusion-attn-rf-cache-mode` | `native`, `cached` | `native` | Original uncached RF preprocessing, or cached static geometry. Layout is selected separately. |
| `--minimax-h3-rf-reorder-mode` | `none`, `post_embed`, `pre_embed` | `none` | Original layout, method 1, or method 2. |
| `--diffusion-attn-head-overlap-graph-mode` | `off`, `graph1`, `graph2` | `off` | Eager dispatch, prefix/body Graph1, or whole-region Graph2 with stable BSND inputs. |
| `--dlo-allgather-gate-mode` | `off`, `first_head` | `off` | Immediate DLO AllGather after H2D, or deferred release. Graph2 uses block-end release as described below. |
| `--dlo-weight-format` | `nd`, `nz` | `nd` | Original weight layout, or host-pack eligible main-DiT INT8 weights and bind NZ views over DLO buffers. |

Defaults enable none of these additional optimizations. There is no public
`u8` optimization preset; use the existing parallelism options for device
count. Public RF choices exclude `optimized`, and public Graph choices
exclude the historical `standard` alias.

## Dependencies

- Graph1 and Graph2 require `pipeline` and `cached` RF.
- RF layout and geometry caching are independent: `post_embed` and `pre_embed` work with `native` or `cached` RF. Graph still requires `cached`.
- Weight gating requires `pipeline` and DLO AllGather. It does not require Graph.
- NZ currently requires NPU H3 INT8 and DLO AllGather. Ineligible non-DiT
  modules and non-weight parameters retain their existing formats.
- Invalid combinations fail during initialization. All ranks must use the
  same settings, and configuration changes require worker restart.
- Geometry caches store static rules, not current Q/K/V, pooled values or
  masks. RF continues to use original QKV pooling. Uncached RF is not a
  Graph ablation: captured replay does not rerun Python geometry preparation.
- Graph input fusion is outside these options; ordinary send prepacking
  remains part of the pipeline.
- NZ retains request-level weight/group agreement even with `legacy`;
  attention data still uses the original framework route.

## Example: optimized digital case

Append these arguments to the existing H3 service command with its model,
INT8, RF, parallelism and DLO options:

```shell
--diffusion-attn-head-overlap-mode pipeline \
--diffusion-attn-rf-cache-mode cached \
--minimax-h3-rf-reorder-mode post_embed \
--diffusion-attn-head-overlap-graph-mode graph1 \
--dlo-allgather-gate-mode first_head \
--dlo-weight-format nz
```

Use `--dlo-allgather-gate-mode off` to retain Graph without weight gating.
Use `--diffusion-attn-head-overlap-graph-mode off` for eager execution.

## Switching Graph1 and Graph2

Replace the Graph argument and restart workers:

```shell
--diffusion-attn-head-overlap-graph-mode graph1
# Or:
--diffusion-attn-head-overlap-graph-mode graph2
```

| Variant | Captured region | Behavior with `--dlo-allgather-gate-mode first_head` |
| --- | --- | --- |
| `graph1` | First-input communication prefix and remaining transport/attention body, two graphs per signature. | A real event and host callback between replays release next-block AllGather after the first input communication completes. |
| `graph2` | Entire weight-free head-split transport/attention region, one graph per signature. | The sparse invocation marks its deferred ticket for release at the end of the current block. No offload callback or AllGather is captured. |

Graph2 replay does not execute captured Python callbacks. Its actual gate
policy is block-end release, despite the requested `first_head` value; report
this policy in logs and snapshots. Normal block completion publishes exactly
once; failed blocks do not publish pending communication. Graph1 retains the
strict missing-first-head check.

With gating `off`, prefetch stays immediate in both variants. Graph2 dense
phases stay eager and retain the ordinary first-head callback; only eligible
sparse calls use the graph and block-end policy. Graph1 retains its full RF
schedule eligibility rules. RF layout is independent of the Graph variant.
Default Graph mode is `off`.

The paired MindIE-SD must expose `single_graph_supported()` and include
`graph_single=1` in its execution agreement for Graph2. Older or partially
upgraded builds are refused. Explicit executor policies select the Graph
variant without exporting low-level Graph environment variables.

Request receipts identify `graph_variant`. Cache counters distinguish
`graph_single_captures` / `graph_single_replays` from prefix replays;
prefetch counters distinguish `released_at_block_end` from
`released_after_input`. These host observations do not prove device kernel
completion or performance improvement.

## Switching RF layout methods

Select one layout independently of `--diffusion-attn-rf-cache-mode native|cached`:

| `--minimax-h3-rf-reorder-mode` | Method |
| --- | --- |
| `post_embed` | Method 1: embed in original order, redistribute into RF order, and preserve original-rank AdaLN parameters in every block. Existing G3–G6 measurements used this layout. |
| `pre_embed` | Method 2: rewrite packed token positions once before embedding so embedding directly produces RF order. It skips method 1 redistribution and per-block original-owner AdaLN synchronization. Output may differ from original layout; baseline MD5 equivalence is not promised. |
| `none` | Disable layout hoisting; RF caching remains independently selectable. |

For method 2 replace the example layout argument with:

```shell
--minimax-h3-rf-reorder-mode pre_embed
```

`legacy_input_hoist` may remain a historical internal resolved name, but it is
not a public CLI value. Method 2 requires packed requests with explicit video
spans and MindIE-SD's `pre_rearranged` API. Request proof reports
`request_hoist_calls=1` and `pre_rearranged=true` when it runs. Unsupported
requests retaining existing fallback must not be reported as method-2
execution merely because the layout was selected.

## Compatibility and observations

The service path uses CLI/configuration objects exclusively for these six
options. Old optimization environment variables and the `u8` preset are not
public service configuration inputs. Any retained old parser is for internal
compatibility tests, not an environment-to-config-to-environment bridge.
Existing platform, input and communication determinism environment settings
remain separate from these six optimization options.

Historical experiment logs retain their actual old startup snapshots. When
reading them, old `RF_MODE=optimized` means cached RF plus the selected layout,
`GRAPH_MODE=standard` maps to `graph1`, and `WEIGHT_GATE=0/1` maps to
`off/first_head`. This mapping does not revalidate historical hardware results
against the new CLI path.

An explicitly requested pipeline is rejected when its request contract is
unsupported, without silently substituting the original route. Direct/step
invocations with an explicit pipeline also require shared request preparation;
missing preparation raises an error instead of inheriting a legacy
environment-controlled fallback.
`H3OptimizationConfig.group_snapshot()` reports requested and resolved groups.
The H3 execution snapshot reports route selection and transport/Graph counters;
a resolved request alone does not prove device kernel completion.

Paired MindIE-SD policy support is required. CPU contract checks cannot replace
same-configuration NPU numerical, complete-video and performance validation.

## Validation of the CLI migration (2026-10-26)

- CPU: 182 vllm-omni tests plus 82 subtests passed; 111 MindIE-SD tests passed
  with one existing expected failure. Static checks passed.
- Shared two-card NPU tensor probes: eager, Graph1 and Graph2 each matched a
  `q+k+v` reference bit for bit for three input sets on each rank. Graph1 had
  three prefix/body replays and Graph2 had two single-graph replays per rank.
  These are transport/graph checks with a lightweight consumer, not a full
  RainFusion attention or H3 model run.
- PrefetchTicket publication occurred exactly once per invocation, after the
  first input in eager/Graph1 and at block end for Graph2. Real model weight
  AllGather was not part of this probe.
- Three distributed RF row permutations preserved hidden/RoPE values,
  original-owner indices and inverse routing exactly. Full pre-embedding
  placement and sparse attention were not exercised by this boundary probe.
- Eighteen real NPU INT8 ND/NZ matmul checks covered three small shapes, normal
  and transposed storage, and alternating slot reuse. Packed-byte recovery,
  inverse host packing and operator outputs were bit-identical (max diff 0).

The NPU probes used physical cards 2 and 3 without an exclusive reservation;
no model service, profiling or performance benchmark was started. This sample
validation does not establish complete-video MD5 equivalence or performance.

The six public names and corresponding configuration fields were subsequently
renamed without changing choices, defaults or execution policies. After the
rename, the 182 vllm-omni tests and 82 subtests passed again, including CLI
registration, worker field transport and rejection of the previous names.
After explicit approval, the renamed code also passed the NPU probes above
on shared physical cards 2 and 3. All 16 transferred source/test files passed
SHA256 verification. The transport/Graph probe parsed the six new option
names from production CLI registration code; the NZ probe used the renamed
configuration field. All 18 ND/NZ operator comparisons were bit-identical.
Both cards were healthy with no remaining device processes after the probes.
The earlier frozen evidence remains separate; neither run loaded the full
H3 model or measured performance.


RF reorder/cache decoupling regression (2026-10-26): 186 CPU tests and 97
subtests passed. On shared physical NPU card 2, 18 real BF16 RF operator
comparisons covered native/cached geometry and none/post_embed/pre_embed row
layouts across three inputs; outputs restored to original order were
bit-identical (max diff 0). The pre_embed probe used production metadata
rewriting; the post_embed operator probe used local row permutation, with
distributed hidden/RoPE/owner routing checked separately on cards 2 and 3.
Graph1/2 regressions also passed with their required caches. These probes
did not load the complete H3 model, run full embedding/AdaLN, validate full
videos or measure performance.

## Paired source archive: 2026-10-26

Archive ID: `h3-optimization-archive-20261026`. Both repositories use branch
`minimax-h3-u8-refactor-20261022`; install this revision together with the
matching `MindIE-SD` archive. The companion JSON manifest records normalized
source hashes, feature scope and validation limits. Current source commits,
not the historical `apply_bundle.py` patches, define this archive.

The archive includes head-split BF16 overlap, send prepacking, output views,
independent RF cache/layout policies, post_embed and pre_embed row placement,
Graph1/Graph2, DLO gating and NZ weights, plus the six explicit CLI options.
Removed fused input-packing and Q/K-only pooling kernels remain removed.

Archival CPU verification: MindIE-SD 111 passed and one existing xfail;
vllm-omni 189 passed and 97 subtests passed. Full H3 model execution, videos,
MD5 comparisons and exclusive performance were not rerun for this archive.
Historical device results apply only to their recorded source and probe scope.
