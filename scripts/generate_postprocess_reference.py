#!/usr/bin/env python3
"""Generate an upstream PyTorch reference for GEM-X SOMA-v2 decoding.

This executes NVIDIA's pinned rotation and motion helpers over the official
ONNX fixture. It intentionally does not import or deserialize a checkpoint.
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
import torch.nn.functional as F
from scipy.ndimage._filters import _gaussian_kernel1d

SOURCE_REVISION = "32992550dba114c62243fb55e361311972dce8f9"
STATS_SHA256 = "dafe4ef6a62e824b0b325f54e42d508015785509bc478ef76e4208ae1f95ba7c"
REFERENCE_SHA256 = "728aba8c7832376f5f1adb05a01be862cbaefce1584ca3bb24153ec4ae15cf86"
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
            1, 120, 4, 77, 1024, 585, 3
        ):
            raise ValueError("unsupported inference fixture")
        lengths = struct.unpack("<4I", stream.read(16))
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


def gaussian_smooth(value: torch.Tensor) -> torch.Tensor:
    kernel = torch.from_numpy(_gaussian_kernel1d(3, 0, radius=12)).float()[None, None]
    x = value.transpose(-2, -1)
    shape = x.shape[:-1]
    x = F.pad(x.reshape(-1, 1, x.shape[-1])[None], (12, 12, 0, 0), mode="replicate")[0]
    return F.conv1d(x, kernel).squeeze(1).reshape(*shape, -1).transpose(-1, -2)


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
    sys.path.insert(0, str(root))
    from gem.utils.rotation_conversions import (  # pylint: disable=import-outside-toplevel
        axis_angle_to_matrix,
        matrix_to_axis_angle,
        rotation_6d_to_matrix,
    )
    from gem.utils.motion_utils import (  # pylint: disable=import-outside-toplevel
        get_tgtcoord_rootparam,
        rollout_local_transl_vel,
    )

    _, boxes, intrinsics, angular, normalized, camera = read_fixture(args.inference_fixture)
    mean, stddev = load_stats(root / "gem/network/stats_compose.py")
    x = torch.from_numpy(normalized * stddev + mean)[None]
    boxes_t = torch.from_numpy(boxes)[None]
    intrinsics_t = torch.from_numpy(intrinsics)[None]
    camera_t = torch.from_numpy(camera)[None]
    body = matrix_to_axis_angle(rotation_6d_to_matrix(x[..., :456].reshape(1, LENGTH, 76, 6)))
    identity = x[..., 456:501]
    scales = x[..., 501:570].clone()
    scales[..., 0].clamp_(0.7, 1.0)
    orient_camera = matrix_to_axis_angle(rotation_6d_to_matrix(x[..., 570:576]))
    orient_gravity = matrix_to_axis_angle(rotation_6d_to_matrix(x[..., 576:582]))
    local_velocity = x[..., 582:585]

    s, tx, ty = camera_t.unbind(-1)
    sb = s * boxes_t[..., 2]
    cx = 2 * (boxes_t[..., 0] - intrinsics_t[..., 0, 2]) / (sb + 1e-9)
    cy = 2 * (boxes_t[..., 1] - intrinsics_t[..., 1, 2]) / (sb + 1e-9)
    tz = 2 * intrinsics_t[..., 0, 0] / (sb + 1e-9)
    translation_camera = torch.stack((tx + cx, ty + cy, tz), -1)

    def as_identity(rotation):
        result = rotation.clone()
        mask = matrix_to_axis_angle(result).norm(dim=-1) < 1e-5
        result[mask] = torch.eye(3).expand(mask.sum(), -1, -1)
        return result

    camera_delta = as_identity(rotation_6d_to_matrix(torch.from_numpy(angular)[None]))
    rotation_gravity = axis_angle_to_matrix(orient_gravity)
    rotation_camera = axis_angle_to_matrix(orient_camera)
    camera_to_gravity = rotation_gravity @ rotation_camera.mT
    next_to_gravity = camera_to_gravity @ camera_delta.mT
    first = F.normalize(camera_to_gravity[..., 2].clone().index_fill(-1, torch.tensor([1]), 0), dim=-1)
    second = F.normalize(next_to_gravity[..., 2].clone().index_fill(-1, torch.tensor([1]), 0), dim=-1)
    yaw = F.normalize(second.cross(first, dim=-1), dim=-1)
    yaw *= torch.acos(torch.clamp((first * second).sum(-1, keepdim=True), -1, 1))
    yaw = gaussian_smooth(yaw)
    step = axis_angle_to_matrix(yaw).mT
    cumulative = [torch.eye(3)[None]]
    for frame in range(1, LENGTH):
        cumulative.append(cumulative[-1] @ step[:, frame])
    cumulative = as_identity(torch.stack(cumulative, 1))
    orient_world = matrix_to_axis_angle(cumulative @ rotation_gravity)
    translation_world = rollout_local_transl_vel(local_velocity, orient_world)
    orient_world, translation_world, _ = get_tgtcoord_rootparam(
        orient_world, translation_world, tsf="ay->ay"
    )

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
