# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ["H3_BENCH_ROOT"])
REPO = Path(os.environ["H3_BENCH_OMNI_REPO"])
PYTHON = os.environ["H3_BENCH_PYTHON"]
PHYSICAL_CARDS = tuple(int(card) for card in os.environ.get("H3_BENCH_CARDS", "8,9,10,11,12,13,14,15").split(","))
assert len(PHYSICAL_CARDS) == len(set(PHYSICAL_CARDS)) == 8


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def snapshot(path):
    result = subprocess.run(
        ["npu-smi", "info"], check=False, capture_output=True, text=True, timeout=30
    )
    path.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("npu-smi unavailable")
    for card in PHYSICAL_CARDS:
        if not re.search(rf"No running processes found in NPU {card}\s", result.stdout):
            raise RuntimeError(f"test card occupied: physical NPU {card}")


def run(arm, name, port):
    output = ROOT / name
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output / "npu-before.txt")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port occupied: {port}")

    ipc_tmp = Path(tempfile.mkdtemp(prefix=f"20261113-h3-{arm}-", dir=str(ROOT.parent)))
    if len(str(ipc_tmp).encode()) + 37 > 107:
        raise RuntimeError("IPC temporary path exceeds Unix socket limit")

    block_sparse = {
        "sparsity": 0.8,
        "start_step": 0,
        "sparse_type": "rf_v3",
        "precision": "bf16",
    }
    if arm == "candidate":
        block_sparse["parallel"] = {"overlap_comm": True, "reorder_hoist": True}
    attention = json.dumps(
        {"default": {"backend": "RAINFUSION_ATTN", "block_sparse": block_sparse}}
    )
    quant = json.dumps(
        {"transformer": {"method": "int8"}, "text_encoder": {"method": "int8"}}
    )

    env = dict(
        os.environ,
        TMPDIR=str(ipc_tmp),
        TRITON_CACHE_DIR=str(output / "triton-cache"),
        TORCH_EXTENSIONS_DIR=str(output / "extensions"),
        ACL_OP_COMPILER_CACHE_DIR=str(output / "operator-cache"),
        ASCEND_RT_VISIBLE_DEVICES=",".join(str(card) for card in PHYSICAL_CARDS),
        VLLM_VERSION="0.33.0",
        VLLM_TARGET_DEVICE="npu",
        VLLM_OMNI_TARGET_DEVICE="npu",
        VLLM_PLUGINS="ascend",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(ROOT / "python") + ":" + str(REPO),
        HCCL_BUFFSIZE="786",
        HCCL_OP_EXPANSION_MODE="AIV",
        OMP_NUM_THREADS="4",
    )
    if arm == "candidate":
        env["VLLM_OMNI_ULYSSES_HCCL_EXPANSION_MODE"] = "2"
    else:
        env.pop("VLLM_OMNI_ULYSSES_HCCL_EXPANSION_MODE", None)
    argv = [
        PYTHON,
        str(ROOT / "bootstrap.py"),
        "serve",
        os.environ["H3_BENCH_MODEL"],
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--task-type",
        "ref2va",
        "--num-gpus",
        "8",
        "--usp",
        "8",
        "--ring",
        "1",
        "--text-encoder-tp-size",
        "8",
        "--vae-patch-parallel-size",
        "8",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--enable-layerwise-offload",
        "--enforce-eager",
        "--enable-diffusion-pipeline-profiler",
        "--diffusion-quantization-config",
        quant,
        "--diffusion-attention-config",
        attention,
    ]
    launch = [
        "bash",
        "-c",
        'source /usr/local/Ascend/cann-9.0.1/set_env.sh\nexec "$@"',
        "h3-parallel-only",
    ] + argv
    write(
        output / "launch.json",
        {
            "argv": argv,
            "arm": arm,
            "name": name,
            "port": port,
            "physical_cards": PHYSICAL_CARDS,
            "expected_nfe": 8,
            "omni_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "sd_source": str(ROOT / "python/mindiesd"),
        },
    )
    stream = (output / "server.log").open("w")
    process = subprocess.Popen(
        launch,
        cwd=output,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write(
        output / "ownership.json",
        {"pid": process.pid, "process_group": process.pid, "argv": launch},
    )
    records = []
    failure = None
    try:
        deadline = time.monotonic() + 900
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited before ready: {process.returncode}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("startup timed out")
            time.sleep(2)

        request = json.loads((ROOT / "request.json").read_text())
        body = (ROOT / "request.multipart").read_bytes()
        for number in range(5):
            phase = "warmup" if number < 2 else "formal"
            begin = (output / "server.log").stat().st_size
            started = time.time()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/videos/sync",
                data=body,
                headers={"Content-Type": request["content_type"]},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=900) as response:
                media = response.read()
                content_type = response.headers.get("content-type", "")
                status = response.status
                response_headers = dict(response.headers)
            elapsed = time.time() - started
            file = (
                output / f"{phase}{number + 1 if phase == 'warmup' else number - 1}.mp4"
            )
            file.write_bytes(media)
            if status != 200 or "video/mp4" not in content_type or len(media) < 100000:
                raise RuntimeError("incomplete media request")
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-show_format",
                        "-of",
                        "json",
                        str(file),
                    ],
                    text=True,
                )
            )
            if not any(
                item["codec_type"] == "video"
                and (item["width"], item["height"]) == (480, 864)
                for item in probe["streams"]
            ):
                raise RuntimeError("wrong output geometry")
            if not any(item["codec_type"] == "audio" for item in probe["streams"]):
                raise RuntimeError("missing audio")
            if any(
                abs(float(item.get("duration", probe["format"]["duration"])) - 12)
                > 0.35
                for item in probe["streams"]
                if item["codec_type"] in ("audio", "video")
            ):
                raise RuntimeError("incomplete media duration")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(file), "-f", "null", "-"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with (output / "server.log").open("rb") as log:
                log.seek(begin)
                request_log = log.read().decode(errors="replace")
            (output / f"request{number + 1}.log").write_text(request_log)
            if not re.findall(r"100%[^\r\n]*\b8/8\b", request_log):
                raise RuntimeError("NFE8 completion missing")
            stages = json.loads(
                next(
                    (
                        value
                        for key, value in response_headers.items()
                        if key.lower() == "x-stage-durations"
                    ),
                    "{}",
                )
            )
            diffuse = stages.get("MiniMaxH3Pipeline.diffuse")
            if not isinstance(diffuse, (float, int)) or diffuse <= 0:
                raise RuntimeError("diffuse stage timing missing")
            record = {
                "request": number + 1,
                "phase": phase,
                "http_status": status,
                "content_type": content_type,
                "e2e_s": elapsed,
                "diffuse_s": float(diffuse),
                "bytes": len(media),
                "sha256": hashlib.sha256(media).hexdigest(),
                "probe": probe,
                "response_headers": response_headers,
                "stage_durations": stages,
            }
            records.append(record)
            write(output / "records.json", records)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "request": number + 1,
                        "phase": phase,
                        "e2e_s": elapsed,
                    }
                ),
                flush=True,
            )
        if len({record["sha256"] for record in records}) != 1:
            raise RuntimeError("same-arm output is not byte deterministic")
    except Exception as error:  # noqa: BLE001 - preserve evidence while owned processes drain
        failure = error
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        stream.close()

    log_text = (output / "server.log").read_text(errors="replace")
    if failure is None and arm == "candidate":
        markers = len(
            re.findall(r"\[mindiesd\.parallel\].*FUSED PRODUCER ENGAGED", log_text)
        )
        completed = [
            int(value)
            for value in re.findall(
                r"MindIE-SD pipeline completed_calls=(\d+)", log_text
            )
        ]
        if markers != 8 or completed != [2000] * 8:
            failure = RuntimeError(
                f"SDK engagement evidence incomplete: markers={markers}, completed={completed}"
            )
    try:
        snapshot(output / "npu-after.txt")
    except Exception as cleanup_error:  # noqa: BLE001 - keep the primary inference failure
        if failure is None:
            failure = cleanup_error
    if failure is not None:
        write(
            output / "failure.json",
            {
                "type": type(failure).__name__,
                "message": str(failure),
                "successful_warmups": sum(
                    record["phase"] == "warmup" for record in records
                ),
                "successful_formal": sum(
                    record["phase"] == "formal" for record in records
                ),
                "records": records,
            },
        )
        raise failure
    write(
        output / "complete.json",
        {"status": "complete", "warmups": 2, "formal": 3, "records": records},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["baseline", "candidate"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=34181)
    args = parser.parse_args()
    run(args.arm, args.name, args.port)
