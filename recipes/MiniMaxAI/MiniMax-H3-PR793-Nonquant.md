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

The isolated NPU paired probe is awaiting approval for source transfer.
These CPU results do not establish full-model/video correctness, NPU kernel
equivalence or performance improvement.

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
NPU source transfer awaits approval; full H3/video/performance not run.
