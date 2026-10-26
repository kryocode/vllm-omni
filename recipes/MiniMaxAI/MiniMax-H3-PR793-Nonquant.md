# PR793 BF16 dual-stream integration

Both repositories use branch `minimax-h3-pr793`. The archive branches
`minimax-h3-u8-refactor-20261022` remain unchanged.

The PR793-only BF16 baseline was SD `007fa43` and omni `29ab5a98`.
This revision imports only the archive's eager dual-stream scheduling into
the existing classes. No production class, function or method is added.
Graph, DLO gating/NZ and RF row hoisting remain removed.

## Optimization changes

- One persistent communication stream per model owner/group/device.
- Forward Q/K/V A2A on that stream, with one initial dependency on prepacking.
- The archived default full lookahead: submit all head inputs before reverse
  communication waits; compute waits for each head's own input-ready event.
- RF preprocessing, attention and inverse gather stay on the compute stream.
- Compute-complete events gate output A2A; assembly waits are deferred until
  all head computations have been submitted.
- record_stream and explicit event/send-buffer ownership protect lifetime.
  Each in-flight head retains its own padded output send buffer.
- Retained-block device indices are built once in the existing geometry cache,
  before A2A dispatch, and reused by every head.
- Rank admission uses the existing SP Gloo CPU group. Strict topology ensures
  its members match Ulysses; collective disagreement checks remain per call.
- Existing prepacked BF16 sends, direct output views, reverse-layout view fast
  path and caller-owned static geometry/transport caches are retained.

The implementation uses existing SD transport/state/pipeline, the RF mask
helper, and one existing omni adapter. Stream identity and overlap policy are part of
owner context; changing either after binding is rejected. All ranks agree
on the overlap setting before sparse attention A2A. Serial mode is preserved.

## Defaults and enablement

| Feature | Default |
| --- | --- |
| H3 PR793 route | OFF: set VLLM_OMNI_MINIMAX_H3_PR793_NONQUANT=1 on every worker to select it. |
| Dual-stream within that route | ON: VLLM_OMNI_MINIMAX_H3_PR793_OVERLAP defaults to 1. Set 0 for serial comparison. |
| SD ParallelState overlap option | True by default; side stream is created only for multi-rank NPU execution. CPU and world=1 use the current stream. |
| Input prepacking/output views | Always used within the selected route. |
| Static geometry/transport caching | Always used within the selected route, bounded to four entries each. |
| Quantized communication/attention | Not imported by this route. |
| Graph, custom DLO gating/NZ, post_embed/pre_embed | Not included or enabled by this revision. |

Defaults do not globally replace the original H3 attention route. The adapter
still requires A2/A3, BF16 multi-span RF, one packed request, and strict
Ulysses without hybrid parallelism. Dense phases use the original framework
route. Unsupported requested sparse configurations fail before data A2A.

## Validation

- CPU: SD 13 tests and omni 11 tests passed; targeted Ruff checks passed.
- Real two-rank HCCL/native BF16 RF: serial and dual-stream both matched the
  reference bitwise on eight heads, two lengths and three datasets per case.
- Full H3 Jimeng case: USP8, INT8 linears, BF16 RF sparsity 0.8, short edge 480,
  4:3, 24 fps, 15 seconds, seed 42, 20 sigma points / 19 actual DiT steps.
- Three formal DiT totals: 47.522 / 47.686 / 47.570 seconds, mean 47.592667.
  First-step mean 2.526667 seconds; final progress-bar mean 2.49 seconds.
- Historical baseline mean: 52.836667 seconds. Current latency is 9.9249%
  lower relative to that historical reference; baseline was not rerun at the
  user's request. This is not a same-window causal A/B result.
- All formal MP4 files match baseline MD5 ac7dbc20d921a7bb4d82e5a802724bff.
  The first formal output was downloaded and SHA256/size verified.
- Separate rank0 Level1 / PipeUtilization profile passed all output checks.
  Within the first-to-last A2A span of one profiled DiT step, A2A overlaps
  AI computation by 89.3% and RF main kernels by 52.9%. These percentages use
  communication interval union as denominator; sums across streams are not
  wall time. The earlier version had zero overlap with RF main kernels.
- Profiled host launch spans show _to_copy cumulative time decreasing from
  2089.355 to 42.719 ms. Nested to/_to_copy/copy intervals must not be added.
- Frozen remote experiment: h3_pr793_fix_20261027-dfc75d37. Production Python
  hashes match the tested payload; documentation was updated afterwards.
- Local evidence: PR793同步修复_20261027 under 即梦case/03_实验与性能.

## Review draft

PR title: `[Bugfix][parallel]Remove host synchronization from BF16 head pipeline`

### Which issue(s) this PR fixes or accomplishes

Restore useful A2A/RF overlap when per-head static tensor construction and
NPU-backed object admission serialize the host and device queues.

### Purpose

Cache retained RF indices in existing geometry plans and perform rank
admission on the existing CPU process group. No new production class or method.

### Test Plan

Run CPU contracts, real two-rank BF16 parity, and three full Jimeng formal
requests followed by a separate profile. Verify final MP4 hashes.

### Test Report

SD 13 / omni 11 CPU tests and real two-rank parity passed. Full-model totals
47.522 / 47.686 / 47.570 seconds; final videos identical to historical baseline.
Level1 / PipeUtilization profile complete; measured RF/A2A overlap restored.
