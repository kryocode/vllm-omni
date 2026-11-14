# MiniMax-H3 MindIE-SD parallel attention: benchmark evidence

This bundle accompanies the opt-in Ascend parallel/RF-hoist PR. `summary.json`
pins the exact engine, SDK, observer and input fingerprint. `timings.csv`
contains every formal per-rank/per-step device-event sample. No measured sample
is discarded. The private reference video, prompt, generated media and local
server logs are excluded; these files are not ground-truth quality references.

## Timing scope

The observer registers model pre/post hooks around each actual DiT transformer
forward, before existing SP pre-hooks and after existing post-hooks. Timing
therefore includes DiT communication and layerwise offload waits. Input packing,
sampler updates, encoders and VAE work outside the transformer are excluded.
There is no observer synchronization between steps; the last event is joined
after the completed denoise loop before reading timestamps.

- First-step time is the slowest rank's forward-1 interval in an eight-step run.
- Mean step time averages the slowest-rank interval of each independently timed
  forward in a request, then averages request samples.
- Eight-forward time is the maximum over ranks of the sum of that rank's eight
  measured forward intervals. It is not the Diffuse wall time divided by eight.
- `diffuse_s` is the whole `MiniMaxH3Pipeline.diffuse` response-stage wall time,
  retained separately because it includes setup and sampler work.

Windows are native A1 → SDK B → native A2. Each has exactly two successful
complete warmups and three formal requests. All use the same engine and SDK
build, input, precision, sparsity, NFE and observer. The SDK arm adds overlap,
RF hoist/preparation caches and the AICPU Ulysses-group setting; the gain is for
that combined configuration rather than additive isolated ablations.

## Reproduction on an allocated eight-NPU host

Requirements: Python 3.12, vLLM 0.33, torch/torch-npu 2.10, CANN 9.0.1,
Triton-Ascend 3.6.0.post2, ffmpeg/ffprobe, a compatible MiniMax-H3 Ref2VA
checkpoint and the matching source-built MindIE-SD wheel/native plugin/OPP.
Use the SDK commit/wheel hash recorded in `summary.json`; package version alone
is insufficient. Original runs use the vLLM-Ascend Python installation at
`67da1c06c`; its full native extension was not built.

`server_ab.py` is a path-parameterized copy of the executed server driver;
`bootstrap.py` and `dit_probe.py` preserve the actual runtime/observer behavior.
Set paths for your environment and use only exclusively allocated empty cards:

```bash
export H3_BENCH_ROOT=/tmp/h3-dit-ab
export H3_BENCH_OMNI_REPO=/path/to/pr-checkout
export H3_BENCH_MODEL=/path/to/approved/Ref2VA
export H3_BENCH_PYTHON=/path/to/venv/bin/python
export H3_BENCH_CARDS=8,9,10,11,12,13,14,15
mkdir -p "$H3_BENCH_ROOT/python"
"$H3_BENCH_PYTHON" -m zipfile -e /path/to/matching/mindiesd.whl "$H3_BENCH_ROOT/python"
cp dit_probe.py "$H3_BENCH_ROOT/python/h3_dit_probe.py"
cp sitecustomize.py "$H3_BENCH_ROOT/python/sitecustomize.py"
cp bootstrap.py "$H3_BENCH_ROOT/bootstrap.py"
"$H3_BENCH_PYTHON" prepare_request.py --root "$H3_BENCH_ROOT" --prompt /path/to/prompt.txt --reference /path/to/reference.mp4
"$H3_BENCH_PYTHON" server_ab.py baseline --name baseline-pre --port 34181
"$H3_BENCH_PYTHON" server_ab.py candidate --name candidate --port 34182
"$H3_BENCH_PYTHON" server_ab.py baseline --name baseline-post --port 34183
```

The driver checks all selected cards before each server, declines occupied
cards, and cleans only its own process group. It will not overwrite existing
window directories. Both arms use BF16 RF-v3/CANN, INT8 transformer/text encoder,
USP8/ring1, text-encoder TP8, VAE patch parallel8/tile/tiling, layerwise offload,
eager execution, OMP4 and HCCL_BUFFSIZE=786. Sampling is 12 seconds, 480×864,
seed42, NFE8, flow_shift12, audio_flow_shift3 and preencode_batch_frames17.

Your own input can change token geometry, masks and latency. Exact replication
of the recorded workload requires its private input; the fingerprint does not
make that input publicly reproducible. Native/SDK generated media differ on the
recorded workload, and independent latent/pixel/audio quality parity remains
unqualified. CUDA/ROCm, multi-node, concurrency and other shapes are not claimed.

AI assistance: TRAE CLI assisted implementation, instrumentation, local checks
and evidence preparation. No private media or credentials are in this bundle.
