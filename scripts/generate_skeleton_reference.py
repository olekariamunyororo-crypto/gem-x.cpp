#!/usr/bin/env python3
"""Generate exact NVIDIA SOMA/MHR skeleton positions for the parity fixture."""

import argparse
import hashlib
from pathlib import Path
import struct
import sys

import numpy as np
import torch

POSTPROCESS_SHA256 = "c92e0ed256e10cd6894b2c8c10094a529e5095f11381c3a31431c665bec1f427"
SOMA_SHA256 = "515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f"
MHR_SHA256 = "8072ec25449c95ea18b3f747b944d8d43377d9114bbd72ef34028610cc3d58e1"
BASE_SHA256 = "cd46620ec46e968658527f25e2b54b8f2292bfd486eac59c596f8b5d5229b9e0"
WRAP_SHA256 = "9ab7a2328d5b026c17ef1d09ed692ffeb68f96d3fc74970cba72e667dbe0f728"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_f32(stream, count: int) -> torch.Tensor:
    value = np.frombuffer(stream.read(count * 4), dtype="<f4")
    if value.size != count:
        raise ValueError("truncated postprocess fixture")
    return torch.from_numpy(value.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("postprocess_fixture", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--soma-source", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.postprocess_fixture) != POSTPROCESS_SHA256:
        raise ValueError("postprocess fixture is not the pinned upstream reference")
    required = {
        args.assets / "SOMA_neutral.npz": SOMA_SHA256,
        args.assets / "MHR/mhr_model_lod6.pt": MHR_SHA256,
        args.assets / "MHR/base_body_lod6.obj": BASE_SHA256,
        args.assets / "MHR/SOMA_wrap_lod1.obj": WRAP_SHA256,
    }
    for path, expected in required.items():
        if digest(path) != expected:
            raise ValueError(f"asset does not match pinned NVIDIA file: {path}")
    sys.path.insert(0, str(args.soma_source.resolve()))
    import soma.soma as soma_module  # pylint: disable=import-outside-toplevel

    # Joint transforms do not depend on pose correctives. Avoid loading the
    # 95 MB vertex-only correctives checkpoint for this skeleton reference.
    soma_module.CorrectivesMLP.load_checkpoint = staticmethod(lambda *args, **kwargs: None)
    model = soma_module.SOMALayer(
        args.assets, low_lod=True, device="cpu", identity_model_type="mhr", mode="torch"
    )
    with args.postprocess_fixture.open("rb") as stream:
        if stream.read(8) != b"GEMPOST1":
            raise ValueError("bad postprocess fixture")
        version, frames = struct.unpack("<2I", stream.read(8))
        if version != 1 or frames != 30:
            raise ValueError("unsupported postprocess fixture")
        body = read_f32(stream, frames * 76 * 3).reshape(frames, 76, 3)
        identity = read_f32(stream, frames * 45).reshape(frames, 45)
        scales = read_f32(stream, frames * 69).reshape(frames, 69)
        read_f32(stream, frames * 3)  # camera orientation
        read_f32(stream, frames * 3)  # camera translation
        orient = read_f32(stream, frames * 3).reshape(frames, 3)
        translation = read_f32(stream, frames * 3).reshape(frames, 3)
        if stream.read(1):
            raise ValueError("trailing postprocess fixture data")
    with torch.inference_mode():
        model.prepare_identity(
            identity.mean(0, keepdim=True), scales.mean(0, keepdim=True)[:, 1:],
            repose_to_bind_pose=False, global_scale=scales.mean(0, keepdim=True)[:, :1]
        )
        poses = torch.cat((orient[:, None], body), dim=1)
        result = model.pose(poses, transl=translation, apply_correctives=False)["joints"]
        bind_positions = model._cached_bind_transforms_world[:, 1:, :3, 3]
        neutral = model.pose(torch.zeros(1, 77, 3), transl=torch.zeros(1, 3),
                             apply_correctives=False)["joints"]
        print("neutral-vs-fitted-bind max", (neutral - bind_positions).abs().max().item())
    joints = result.detach().cpu().numpy().astype("<f4", copy=False)
    if joints.shape != (frames, 77, 3) or not np.isfinite(joints).all():
        raise ValueError("unexpected SOMA skeleton output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"GEMSKEL1")
        stream.write(struct.pack("<3I", 1, frames, 77))
        stream.write(joints.tobytes())
    print(f"wrote {args.output} ({digest(args.output)})")


if __name__ == "__main__":
    main()
