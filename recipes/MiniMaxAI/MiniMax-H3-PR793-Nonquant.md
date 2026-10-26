# MiniMax H3 PR793 non-quantized integration

Use branch `minimax-h3-pr793` in both `kryocode/vllm-omni` and
`kryocode/MindIE-SD`. It starts from archived H3 optimizations at
omni `4da23576` and MindIE-SD `2b67d9d`.

The reviewed source is [MindIE-SD PR793](https://gitcode.com/Ascend/MindIE-SD/pull/793)
at `d405d3646e40d19873e77a921b818d1fa985e60a`.

## Included changes

- MindIE-SD centralizes the reverse-A2A cut/merge pair in `parallel/topology.py`
  and exports the non-quantized `rank_window` helper.
- The prepared omni bridge resolves the local valid sequence window once
  through that helper after rank-common route agreement. H3 restores padding
  using the recorded valid-row count and clears the window at request end.
- Missing helper support prevents entry to the prepared attention route through
  the existing agreement. The legacy caller checks it before data collectives.
- Existing BF16 head partitioning, send prepacking, output views, RF geometry
  caching, Graph variants, DLO gating and NZ support remain on the archived path.
- Tests cover partial and fully padded shards, reverse layout round trips,
  view storage sharing, the paired API and request lifecycle.

## Excluded changes

The PR793 quantized entry, Q/K INT8 and V FP8 payloads, scale AllReduce,
boundary quantization repair, quantized packet/pooling protocol, quantizer
workspaces and quantized BSA consumer are excluded. This does not disable
the separately configured existing Transformer INT8 linear layers.

No new communication/compute scheduler is implemented. The original archive's
optional BF16 dual-stream route remains inherited; this port does not change
its collective order, stream policy or event behavior. New overlap work waits
for the user's decision.

## Validation scope

MindIE-SD CPU verification: 125 passed, one existing xfail, including actual
two-process Gloo A2A parity. Omni CPU verification: 191 passed and 97
subtests passed. New tests first failed before the paired implementation.


The isolated paired NPU probe passed on physical devices 2/3 using real
HCCL A2A and the native BF16 RainFusion consumer. Two datasets for each of
used lengths 1024 and 256 matched the independent full-sequence per-head
reference bitwise on both ranks, including a fully padded rank. Each rank
issued 16 BF16 A2A operations per geometry and zero quantization AllReduce.
The probe disabled overlap and Graph to isolate this integration's behavior.

This small carrier binds the real process group through an adapter and fixes
the RF plan; it does not initialize the full H3 pipeline or load model weights.
The 57 deployed files were checked: 56 are byte-identical after LF normalization;
topology.py differs only in formatting, with identical Python AST. The isolated
wheel needed its original mindiesd.libs directory on the child library path.
Copied omni version metadata emitted a vLLM version mismatch warning, so this
probe does not establish full-model dependency compatibility. Full H3/video
correctness and performance remain untested. No speedup claim is made.

## Review draft

PR title: `[Refactor][minimax-h3]Reuse non-quantized PR793 shard topology`

### Which issue(s) this PR fixes or accomplishes

Issue: pending. Selectively integrate non-quantized topology contracts from
MindIE-SD PR793 into the paired H3 feature branches.

### Purpose

Use one library-owned shard-window definition for framework padding and retain
the tested BF16 execution path. Reject mismatched library capability before
attention data collectives. Quantized transport and attention are excluded.

### Test Plan

Run topology/window contracts, paired bridge lifecycle tests and existing
RF/Graph/DLO CPU regressions. Check real two-rank Gloo layout parity.

### Test Report

SD: 125 passed, one existing xfail. Omni: 191 passed and 97 subtests passed.
Paired two-card BF16 HCCL/RF probe passed bitwise; full H3/video/performance not run.

## Minimal-cost overlap proposal (analysis only)

The paired branch already inherits a BF16 dual-stream executor. Reusing it
costs less than adding a second scheduler to PR793's quantized pipeline:

- `TransportState.make_stream()` owns one persistent communication stream.
- `Transport.issue()` submits Q/K/V A2A on that stream and records input-ready
  events; the compute stream waits only for the head it consumes.
- The next head's input exchanges are queued before the current head's
  reverse-A2A dependency wait. Otherwise that wait can serialize the pipeline.
- Compute-complete and reverse-complete events protect the output exchange
  and final assembly. `record_stream` and retained send buffers protect storage.

The framework selects the prepared pipeline and provides the native BF16 RF
consumer; scheduling belongs in MindIE-SD. No framework-side duplicate stream
or communicator should be added. Begin any future overlap evaluation with
Graph and weight gating disabled, then compare the existing single-stream and
side-stream modes with identical inputs and inspect the device timeline.
Stream counters establish route selection, not actual concurrent execution.

If an independent PR793-based executor is required later, the smallest design
is to adapt that same persistent-stream/event schedule around its chunk loop,
keeping quantization excluded. It also needs lifetime, collective-order,
padding and paired-rank tests. Graph capture and buffer reuse across in-flight
heads require separate validation; a single reused send buffer is unsafe.
This proposal adds no scheduler code and awaits the user's decision.
