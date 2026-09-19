#!/usr/bin/env python3
"""Generate an upstream PyTorch reference for GEM-X SOMA-v2 decoding.

This instantiates NVIDIA's configured decoder and executes its camera/world
helpers over the official ONNX fixture. No checkpoint is deserialized.
"""

import argparse
import ast
import hashlib
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import torch

SOURCE_REVISION = "32992550dba114c62243fb55e361311972dce8f9"
STATS_SHA256 = "dafe4ef6a62e824b0b325f54e42d508015785509bc478ef76e4208ae1f95ba7c"
REFERENCE_SHA256 = "17cf39b9d0df6809e2c4c5db9c4b2ba0833ed24e5ba3eda2e9fa320a3ff5d073"
LENGTH = 30


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if digest(path) != STATS_SHA256:
        raise ValueError("GEM-X statistics do not match the pinned source")
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MM_V2_SOMA_METROSIM"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return np.asarray(value["mean"], np.float32), np.asarray(value["std"], np.float32)
    raise ValueError("pinned SOMA-v2 statistics are missing")


def read_f32(stream, count: int) -> np.ndarray:
    value = np.frombuffer(stream.read(count * 4), dtype="<f4")
    if value.size != count:
        raise ValueError("truncated inference fixture")
    return value.copy()


def read_fixture(path: Path):
    if digest(path) != REFERENCE_SHA256:
        raise ValueError("inference fixture does not match the pinned official fixture")
    with path.open("rb") as stream:
        if stream.read(8) != b"GEMXREF1":
            raise ValueError("invalid inference fixture")
        version, maximum, count, joints, features, motion_dim, camera_dim = struct.unpack(
            "<7I", stream.read(28)
        )
        if (version, maximum, count, joints, features, motion_dim, camera_dim) != (
            1, 120, 5, 77, 1024, 585, 3
        ):
            raise ValueError("unsupported inference fixture")
        lengths = struct.unpack("<5I", stream.read(20))
        keypoints = read_f32(stream, maximum * 77 * 3).reshape(maximum, 77, 3)
        boxes = read_f32(stream, maximum * 3).reshape(maximum, 3)
        intrinsics = read_f32(stream, maximum * 9).reshape(maximum, 3, 3)
        read_f32(stream, maximum * 1024)
        angular = read_f32(stream, maximum * 6).reshape(maximum, 6)
        selected = None
        for length in lengths:
            motion = read_f32(stream, length * 585).reshape(length, 585)
            camera = read_f32(stream, length * 3).reshape(length, 3)
            if length == LENGTH:
                selected = motion, camera
    if selected is None:
        raise ValueError("required length is absent from inference fixture")
    return keypoints[:LENGTH], boxes[:LENGTH], intrinsics[:LENGTH], angular[:LENGTH], *selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inference_fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.upstream_root.resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("GEM-X checkout is not the pinned source revision")
    load_stats(root / "gem/network/stats_compose.py")  # Verify the imported statistics hash.
    sys.path.insert(0, str(root))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from gem.utils.cam_utils import compute_transl_full_cam
    from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2

    _, boxes, intrinsics, angular, normalized, camera = read_fixture(args.inference_fixture)
    decoder = instantiate(OmegaConf.load(root / "configs/endecoder/v2_soma_local_cam.yaml"))
    decoder.build_obs_indices_dict()
    with torch.inference_mode():
        decoded = decoder.decode(torch.from_numpy(normalized)[None])
        body = decoded["body_pose"]
        identity = decoded["identity_coeffs"]
        scales = decoded["scale_params"].clone()
        scales[..., 0].clamp_(0.7, 1.0)
        orient_camera = decoded["global_orient"]
        translation_camera = compute_transl_full_cam(
            torch.from_numpy(camera)[None], torch.from_numpy(boxes)[None],
            torch.from_numpy(intrinsics)[None])
        world = get_body_params_w_Rt_v2(decoded["global_orient_gv"],
            decoded["local_transl_vel"], orient_camera, torch.from_numpy(angular)[None])
        orient_world, translation_world = world["global_orient"], world["transl"]

    values = (body, identity, scales, orient_camera, translation_camera,
              orient_world, translation_world)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"GEMPOST1")
        stream.write(struct.pack("<2I", 1, LENGTH))
        for value in values:
            array = value.detach().cpu().numpy().astype("<f4", copy=False)
            if not np.isfinite(array).all():
                raise ValueError("non-finite upstream postprocess reference")
            stream.write(array.tobytes())
    print(f"wrote {args.output} ({digest(args.output)})")


if __name__ == "__main__":
    main()
