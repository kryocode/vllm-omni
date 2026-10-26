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
- Existing prepacked BF16 sends, direct output views, reverse-layout view fast
  path and caller-owned static geometry/transport caches are retained.

The changes touch three existing SD source files (comm/state/pipeline) and
one existing omni adapter. Stream identity and overlap policy are part of
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

- Test-first: new stream/default tests failed before implementation.
- CPU: SD 11 passed, including actual two-rank Gloo and owner policy/stream
  mutation rejection; omni 10 passed, including defaults and serial opt-out.
- NPU: real two-card HCCL and native BF16 RF passed in serial and dual-stream
  modes. Eight heads, used lengths 1024/256, three changed datasets per case;
  both ranks matched the independent native per-head reference bitwise.
- Covers partial/fully padded shards, persistent stream reuse, hot geometry
  caches and additional allocations after dispatch. Each case uses 48 BF16
  A2As per rank and zero quantization AllReduce.
- All 25 deployed files verified. Two source files received docstring-only
  corrections after the probe; executable AST matches the tested payload.
- Targeted offline pre-commit/Ruff and syntax checks passed.
  Full hook environment initialization was unavailable earlier; offline checks
  do not claim that every configured repository hook ran.

The probe uses fixed small RF plans and a real process-group carrier, without
initializing the full H3 pipeline or loading weights. Copied omni metadata
warns about a vLLM version mismatch; full-model dependency compatibility,
complete video correctness, MD5 and performance are not established.
No device timeline was captured, so this validates route execution and
numerical parity, not a measured overlap ratio or speedup.

## Review draft

PR title: `[Feature][parallel]Reuse archived BF16 dual-stream head scheduling`

### Which issue(s) this PR fixes or accomplishes

Issue: pending. Add minimal dual-stream scheduling to the PR793-only BF16
branches without restoring the independent optimization framework.

### Purpose

Reuse existing state and Transport classes, preserve archived full-lookahead
ordering and buffer protection, and retain a serial mode for comparison.

### Test Plan

Check CPU contracts and real two-rank communication. Compare serial and
dual-stream native RF outputs on changing inputs and padded shards.

### Test Report

SD 11 CPU tests and omni 10 CPU tests passed. Real two-card serial/dual-stream
RF/HCCL probe passed bitwise. Full model/video/performance not run.
