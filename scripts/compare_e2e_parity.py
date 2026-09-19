#!/usr/bin/env python3
"""Compare native offline predictions with an upstream GEM-X result bundle."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np
import torch


def native_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 12 or data[:8] != b"GEMRAW01":
        raise ValueError(f"{path}: invalid native prediction file")
    frames = struct.unpack_from("<I", data, 8)[0]
    if not frames:
        raise ValueError(f"{path}: empty prediction sequence")
    expected = 12 + frames * (585 + 3) * 4
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(data)}")
    values = np.frombuffer(data, dtype="<f4", offset=12)
    split = frames * 585
    return values[:split].reshape(frames, 585), values[split:].reshape(frames, 3)


def upstream_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    result = torch.load(path, map_location="cpu", weights_only=False)
    output = result["net_outputs"]["model_output"]
    motion = output.get("pred_x", output.get("pred_x_start"))
    if motion is None:
        raise ValueError(f"{path}: model_output has no pred_x or pred_x_start")
    camera = output["pred_cam"]
    motion = motion.detach().float().cpu().numpy()
    camera = camera.detach().float().cpu().numpy()
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    if camera.ndim == 3 and camera.shape[0] == 1:
        camera = camera[0]
    return motion, camera


def statistics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: native {actual.shape}, upstream {expected.shape}")
    if not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("parity inputs must be nonempty and finite")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "maximum": float(difference.max(initial=0.0)),
        "mean": float(difference.mean()),
        "rms": float(np.sqrt(np.mean(difference * difference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path, help="native predictions.bin")
    parser.add_argument("upstream", type=Path, help="upstream hpe_results.pt")
    parser.add_argument("--motion-max", type=float, default=0.002)
    parser.add_argument("--motion-mean", type=float, default=0.0001)
    parser.add_argument("--camera-max", type=float, default=0.001)
    parser.add_argument("--camera-mean", type=float, default=0.0001)
    args = parser.parse_args()

    native_motion, native_camera = native_predictions(args.native)
    upstream_motion, upstream_camera = upstream_predictions(args.upstream)
    report = {
        "frames": int(native_motion.shape[0]),
        "motion": statistics(native_motion, upstream_motion),
        "camera": statistics(native_camera, upstream_camera),
    }
    print(json.dumps(report, indent=2))
    failed = (
        report["motion"]["maximum"] > args.motion_max
        or report["motion"]["mean"] > args.motion_mean
        or report["camera"]["maximum"] > args.camera_max
        or report["camera"]["mean"] > args.camera_mean
    )
    if failed:
        raise SystemExit("end-to-end parity thresholds exceeded")


if __name__ == "__main__":
    main()
