# PR793 non-quantized branch after removing the custom optimization stack

Use `minimax-h3-pr793` in both repositories. The original
`minimax-h3-u8-refactor-20261022` branches are preserved without changes.
Earlier paired-probe results from the inherited optimization stack do not
validate this replacement implementation.

The clean source bases are MindIE-SD `39950728895b3f00a8cbe641a5329705e73dfaac`
and vllm-omni `7149a6d4551caaba138b63f413e3d197d01bd16a`.
Original native H3/RF support remains. Only the PR793 adaptation and its
necessary integration differ from these bases.

Source: [Ascend/MindIE-SD PR793](https://gitcode.com/Ascend/MindIE-SD/pull/793),
reviewed head `d405d3646e40d19873e77a921b818d1fa985e60a`.
This is a BF16 adaptation of its non-quantized mechanisms, not an unmodified
copy of its quantized pipeline.

## Retained PR793 mechanisms

| Mechanism | Current implementation |
| --- | --- |
| Per-head A2A chain | Current-stream Q/K/V A2A, BF16 attention, output A2A, then next chunk. One head per destination per chunk. |
| Send prepacking | Prepare contiguous Q/K/V sends before entering the chunk loop. |
| Direct output destination | Inverse RF gather writes directly into the payload view of the reverse-A2A buffer. Each chunk owns a fresh padded buffer. |
| Layout transforms | Adjacent reverse-A2A split/merge helpers, including the contiguous batch=width=1 view fast path. |
| Rank window | Shared valid-row count; omni restores the original query padding, including fully padded ranks. |
| Static geometry | Caller-owned permutation, inverse, dense and first-frame block sets, reused for identical spans/length/device. Bounded FIFO of four plans. |
| Persistent owner | One model-owned state shared by DiT layers; group/rank/device validation, transport-plan reuse and failure/reentry protection. |

Q/K/V, pooling results, scores and masks are recomputed from current BF16
inputs. No quantized payload, scale AllReduce, boundary repair, quantized
packet or quantized attention consumer is imported. Existing unrelated
Transformer linear quantization and native DLO features stay at the clean base.

## Removed custom features

- Side communication stream, lookahead/event overlap and first-head handoff.
- Graph1/Graph2 capture/replay and associated stable buffer caches.
- Custom DLO AllGather tickets/gating, NZ weight ownership and packing.
- post_embed/pre_embed row hoisting and original-rank AdaLN synchronization.
- Custom RFGeometryCache facade, stream/version-checked geometry cache and
  CPU geometry construction enhancements from the previous branch.
- The six custom CLI optimization controls, old H3 overlap adapters, Wan
  integration and their archived patch bundles/tests/documentation.

Removing those custom features does not remove original upstream support
already present at the stated clean bases. Native RF pooling and mask math
are unchanged; the new consumer calls those original library functions.

## Paired enablement

Set `VLLM_OMNI_MINIMAX_H3_PR793_NONQUANT=1` on every worker before startup.
The default is 0. Old custom optimization flags no longer select this route.
The adapter supports A2/A3 NPU, BF16 full-query multi-span RF, one packed
request and strict Ulysses without ring/context/allgather hybrid parallelism.
Dense RF phases retain the original framework route. Sparse rank signatures
must agree before attention data A2A. Unsupported requested cases raise.
A5 and quantized attention require their original framework configuration
with this new route disabled. No double-stream implementation is added.

## Validation

- CPU: MindIE-SD 9 passed, including actual two-rank Gloo BF16 communication,
  head placement, batch 1/2, partial/empty padding, serial collective order,
  failure-state rejection and bounded cache tests.
- CPU: omni 8 passed, covering admission, shared ownership, padding, dense
  routing, rank disagreement, quantized-attention rejection and old flags.
- New paired NPU validation passed on physical cards 2/3: real HCCL A2A,
  native BF16 RF, used lengths 1024/256 with two datasets each. Both ranks
  matched the independent native per-head reference bitwise, including
  partial and fully padded shards. Each geometry issued 16 BF16 A2As
  per rank and zero quantization AllReduce; two static plans were reused.
  The carrier fixes small RF plans and binds a real process group without
  constructing the full H3 pipeline. All 25 deployed source/probe files
  match current normalized source hashes. Prior results are superseded.
- The isolated wheel uses original mindiesd.libs through child-only
  LD_LIBRARY_PATH. Copied omni version metadata warns about a vLLM
  version mismatch; full-model dependency compatibility is not established.
- Full model weights, complete H3 video, output MD5 and performance are not run.

The change restores baseline sources and replaces the inherited stack. New
adaptation code was implemented before the new CPU tests were recorded;
there is no test-first failure record for this cleanup revision. No performance
claim is made. Static validation covers new code and baseline-source identity.

## Review draft

PR title: `[Refactor][parallel]Isolate PR793 BF16 subset from custom H3 optimizations`

### Which issue(s) this PR fixes or accomplishes

Issue: pending. Remove the custom optimization stack from the paired PR793
branches while preserving the original archive branches.

### Purpose

Keep a reviewable current-stream BF16 adaptation of PR793 and a thin H3 adapter.
Exclude quantized communication/attention and the independent custom stack.

### Test Plan

Verify clean-base source identity, serial BF16 collective order, head mapping,
padding, owner lifetime and rank agreement. Run a paired native RF NPU probe.

### Test Report

SD: 9 CPU tests passed. Omni: 8 CPU tests passed. New two-card native BF16 RF/HCCL probe passed bitwise.
Full-model/video/performance tests are outside this result.

## Minimal-change dual-stream proposal (not implemented)

Keep the existing serial implementation as the default and add one optional
side-stream schedule around the same BF16 buffers and consumer. No RF math,
geometry, packet format, Graph, DLO or NZ change is needed.

| File | Proposed change |
| --- | --- |
| SD `parallel/state.py` | Lazily own one communication stream per live group/device; retain a fixed scheduling policy for this owner. |
| SD `parallel/comm.py` | Extract input issue, add input-ready/compute-done/output-ready events, prefetch exactly one next head, defer reverse-output assembly to the end. |
| SD `parallel/pipeline.py` | Pass the owner's stream/policy when constructing a cached Transport. Existing prepack and BF16 RF consumer stay unchanged. |
| Omni `attention/pr793_nonquant.py` | Pass one opt-in overlap flag and include it in rank agreement; keep serial selection for A/B validation. |

At entry, the communication stream waits once for prepacked inputs produced
on the compute stream. The input0 exchange is issued first. The compute
stream waits for each head's input-ready event and submits its RF work.
Before enqueueing that head's output wait/A2A, issue the next head's Q/K/V
A2A. That order lets next-input communication overlap current RF and avoids
putting the next input behind a compute dependency it does not need.

Do not add a compute-stream wait to every next-input issue: all sends were
already prepacked. Do not keep the current per-head output assembly wait
inside the compute loop: it would prevent the next head's RF from overlapping
the prior output A2A. Store reverse receive buffers/events and assemble only
after all heads' RF work has been submitted.

Use record_stream and explicit buffer/event ownership for tensors crossing
streams. Keep the existing separate padded send buffer for every in-flight
head. All attention A2As must be issued by the same communication stream on
each rank, with identical rank-wide collective order. The schedule adds
temporary buffer residency, including deferred output receives; its memory
cost must be measured. A blanket async_op change alone is insufficient.

The production change is expected to touch these four files, with most code
in comm.py (roughly 100-160 added/changed lines overall, excluding tests;
an estimate, not a fixed budget). First validate serial/overlap bitwise parity
on the current small HCCL/RF probe, changed inputs, padding and repeated
calls. Then use an exclusive-device timeline to establish concurrent RF/A2A
execution and measure net latency. Stream creation/counters alone do not
prove overlap or speedup. Implementation awaits the user's decision.
