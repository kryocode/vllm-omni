# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prepare a Ref2VA request from user-supplied prompt/media; do not upload it."""

import argparse
import json
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--prompt", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
args = parser.parse_args()
boundary = "h3bench-" + uuid.uuid4().hex
fields = {
    "prompt": args.prompt.read_text(),
    "seconds": "12",
    "width": "480",
    "height": "864",
    "seed": "42",
    "num_inference_steps": "8",
    "flow_shift": "12",
    "extra_params": json.dumps({"audio_flow_shift": 3, "preencode_mp4": True, "preencode_batch_frames": 17}),
}
parts = []
for name, value in fields.items():
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode())
reference_header = (
    f"--{boundary}\r\nContent-Disposition: form-data; "
    'name="input_references"; filename="reference.mp4"\r\n'
    "Content-Type: video/mp4\r\n\r\n"
)
parts.append(reference_header.encode() + args.reference.read_bytes() + b"\r\n")
parts.append((f"--{boundary}--\r\n").encode())
args.root.mkdir(parents=True, exist_ok=True)
(args.root / "request.multipart").write_bytes(b"".join(parts))
metadata = {"content_type": "multipart/form-data; boundary=" + boundary}
(args.root / "request.json").write_text(json.dumps(metadata, indent=2) + "\n")
