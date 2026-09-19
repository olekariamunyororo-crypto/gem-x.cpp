#!/usr/bin/env python3
"""Audit trusted local video captures through raw outputs and SOMA skeletons.

Requires the reference Python environment and the pinned SOMA source/assets.
This reports measurements, not a ground-truth motion quality score. Unlike
compare_e2e_parity.py, it does not apply raw-network acceptance thresholds.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

from compare_e2e_parity import native_predictions, statistics, upstream_predictions


def read_pose(path):
    data = path.read_bytes()
    if data[:8] != b"GEMPOSE2" or len(data) != 8 + (231 * 4 + 308 + 77 + 3) * 4:
        raise ValueError(f"Expected GEMPOSE2: {path}")
    at = 8
    result = {}
    for name, count in (("world", 231), ("camera_local", 231),
                        ("rotations", 308), ("local_translations", 231)):
        result[name] = np.frombuffer(data, dtype="<f4", count=count, offset=at).copy()
        at += count * 4
    result["parents"] = np.frombuffer(data, dtype="<i4", count=77, offset=at).copy()
    at += 77 * 4
    result["camera_translation"] = np.frombuffer(data, dtype="<f4", count=3, offset=at).copy()
    at += 12
    result["keypoints"] = np.frombuffer(data, dtype="<f4", count=231, offset=at).copy()
    for name in ("world", "camera_local", "keypoints"):
        result[name] = result[name].reshape(77, 3)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError(f"Nonfinite pose: {path}")
    return result


def distances(actual, expected, scale=1000):
    if actual.shape != expected.shape or not np.isfinite(expected).all():
        raise ValueError("Invalid reference geometry")
    d = np.linalg.norm(actual.astype(np.float64) - expected, axis=-1) * scale
    worst = np.unravel_index(d.argmax(), d.shape)
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)),
            "maximum": float(d.max()), "worst_index": list(map(int, worst)),
            "per_frame_mean": np.mean(d.reshape(len(d), -1), axis=1).tolist()}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_token(path):
    data = path.read_bytes()
    if data[:8] != b"S3DOUT01":
        raise ValueError("Invalid Body result")
    count = struct.unpack_from("<I", data, 8)[0]
    at = 12
    for _ in range(count):
        size = struct.unpack_from("<I", data, at)[0]
        at += 4
        name = data[at:at + size].decode()
        at += size
        typ, rank, elements = struct.unpack_from("<IIQ", data, at)
        at += 16
        shape = struct.unpack_from("<" + "Q" * rank, data, at)
        at += rank * 8
        if name == "pose_token":
            if typ != 1 or shape != (1, 1024) or elements != 1024:
                raise ValueError("Invalid Body pose token")
            return np.frombuffer(data, dtype="<f4", count=1024, offset=at).copy()
        at += elements * 4
    raise ValueError("Missing Body pose token")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--soma-source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--native", action="append", required=True, help="LABEL=output-directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, help="Optional camera XY skeleton comparison PNG")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.soma_source.resolve()))
    import soma.soma as soma_module

    # Vertex correctives have no effect on the joint transforms being audited.
    soma_module.CorrectivesMLP.load_checkpoint = staticmethod(lambda *a, **kw: None)
    soma = soma_module.SOMALayer(args.assets, low_lod=True, device="cpu",
                                 identity_model_type="mhr", mode="torch")
    reference = torch.load(args.upstream, map_location="cpu", weights_only=False)
    expected_motion, expected_camera = upstream_predictions(args.upstream)
    frames = len(expected_motion)
    expected = {}
    with torch.inference_mode():
        for key in ("body_params_global", "body_params_incam"):
            params = {k: v.detach().float().cpu() for k, v in reference[key].items()}
            if params["global_orient"].ndim == 3:
                params = {k: v[0] for k, v in params.items()}
            scales = params["scale_params"].mean(0, keepdim=True)
            soma.prepare_identity(params["identity_coeffs"].mean(0, keepdim=True),
                                  scales[:, 1:], repose_to_bind_pose=False,
                                  global_scale=scales[:, :1])
            poses = torch.cat((params["global_orient"][:, None],
                               params["body_pose"].reshape(frames, 76, 3)), dim=1)
            expected[key] = soma.pose(poses, transl=params["transl"],
                                      apply_correctives=False)["joints"].numpy()
    report = {"scope": "One video; numerical parity, not ground truth quality",
              "upstream": str(args.upstream), "upstream_sha256": sha(args.upstream),
              "frames": frames, "runs": {}}
    prep = args.upstream.parent / "preprocess/vitpose.pt"
    kp = None
    if prep.exists():
        kp = torch.load(prep, map_location="cpu", weights_only=False)
        if isinstance(kp, tuple):
            kp = kp[0]
        kp = kp.detach().cpu().numpy()
    plot_runs = []
    for spec in args.native:
        label, path = spec.split("=", 1)
        path = Path(path)
        motion, camera = native_predictions(path / "predictions.bin")
        if len(motion) != frames:
            raise ValueError("Frame counts differ")
        poses = [read_pose(path / f"{i:06d}.gpose") for i in range(frames)]
        world = np.stack([v["world"] for v in poses])
        incam = np.stack([v["camera_local"] + v["camera_translation"] for v in poses])
        target = expected["body_params_incam"]
        plot_runs.append((label, incam, poses[0]["parents"]))
        run = {"path": str(path), "predictions_sha256": sha(path / "predictions.bin"),
               "raw_motion": statistics(motion, expected_motion),
               "raw_camera": statistics(camera, expected_camera),
               "world_joints_mm": distances(world, expected["body_params_global"]),
               "pelvis_relative_world_joints_mm": distances(
                   world - world[:, :1], expected["body_params_global"] - expected["body_params_global"][:, :1]),
               "camera_joints_mm": distances(incam, target),
               "pelvis_relative_camera_joints_mm": distances(incam - incam[:, :1], target - target[:, :1])}
        if kp is not None:
            native_kp = np.stack([v["keypoints"] for v in poses])
            run["keypoint_xy_pixels"] = distances(native_kp[..., :2], kp[..., :2], scale=1)
            run["keypoint_confidence"] = statistics(native_kp[..., 2], kp[..., 2])
        body_dir = path.parent / "body"
        prep_dir = args.upstream.parent / "preprocess"
        if body_dir.is_dir() and (prep_dir / "vit_features.pt").exists():
            native_features = np.stack([body_token(body_dir / f"{i:06d}.bin") for i in range(frames)])
            features = torch.load(prep_dir / "vit_features.pt", map_location="cpu", weights_only=False)
            run["body_token"] = statistics(native_features, features["pose_tokens"].numpy())
        input_dir = path.parent.parent
        boxes_path = input_dir / "boxes.bin"
        if boxes_path.exists() and (prep_dir / "bbx.pt").exists():
            raw = boxes_path.read_bytes()
            if raw[:8] != b"GEMBOX01" or struct.unpack_from("<I", raw, 8)[0] != frames:
                raise ValueError("Invalid detector output")
            boxes = np.frombuffer(raw, dtype="<f4", offset=12).reshape(frames, 4)
            target_boxes = torch.load(prep_dir / "bbx.pt", map_location="cpu", weights_only=False)["bbx_xyxy"].numpy()
            run["detector_box_coordinates_pixels"] = statistics(boxes, target_boxes)
            rgb_hash = hashlib.sha256()
            for i in range(frames):
                rgb_hash.update((input_dir / "frames" / f"{i:06d}.input").read_bytes()[52:])
            run["input_rgb_sha256"] = rgb_hash.hexdigest()
        report["runs"][label] = run
        print(label, {k: v["mean"] for k, v in run.items() if isinstance(v, dict)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        selected = [0, frames // 3, 2 * frames // 3]
        fig, axes = plt.subplots(len(plot_runs), 3, figsize=(12, 4 * len(plot_runs)), squeeze=False)
        for row, (label, joints, parents) in enumerate(plot_runs):
            for col, frame in enumerate(selected):
                ax = axes[row, col]
                for values, color, name in ((expected["body_params_incam"], "#999999", "Upstream"),
                                            (joints, "#0073b7", label)):
                    points = values[frame] - values[frame, :1]
                    for joint, parent in enumerate(parents):
                        if parent >= 0:
                            ax.plot(points[[parent, joint], 0], -points[[parent, joint], 1],
                                    color=color, lw=1, alpha=.8)
                    ax.plot([], [], color=color, label=name)
                ax.set(xlim=(-1, 1), ylim=(-1.2, 1.2), title=f"{label}: frame {frame}",
                       xlabel="Camera X relative to pelvis (m)", ylabel="Camera -Y (m)")
                ax.set_aspect("equal")
                ax.legend(fontsize=8)
        fig.suptitle("SOMA skeleton parity: pelvis aligned, camera XY view")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
