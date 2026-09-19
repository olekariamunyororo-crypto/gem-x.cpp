#!/usr/bin/env python3
"""Replay RGB frames through resident native live models for parity testing.

No Body model is loaded. This mirrors upstream's per-frame detector selection
and rolling inference, including its two-frame warm-up. It is a test harness,
not a camera capture or SONIC publisher. Run under taskset -c 0-7.
"""
import argparse
import ctypes as c
import json
from pathlib import Path
import struct
import time

import numpy as np


class Config(c.Structure):
    _fields_ = [(n, c.c_char_p) for n in ("model", "module", "backend", "description")] + [
        (n, c.c_uint32) for n in ("device", "threads", "cache")]


class Frame(c.Structure):
    _fields_ = [("rgb", c.c_void_p), ("capacity", c.c_uint64), ("width", c.c_uint32),
                ("height", c.c_uint32), ("stride", c.c_uint64), ("box", c.c_float * 3)]


class Detection(c.Structure):
    _fields_ = [("box", c.c_float * 4), ("score", c.c_float)]


class Sequence(c.Structure):
    _fields_ = [("frames", c.c_uint32)] + [(n, c.c_void_p) for n in
        ("keypoints", "boxes", "intrinsics", "features", "angular")]


class Motion(c.Structure):
    _fields_ = [("frames", c.c_uint32)] + [(n, c.c_void_p) for n in
        ("body", "identity", "scales", "orient_camera", "translation_camera", "orient_world", "translation_world")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "module", "gem", "vitpose", "yolox", "frames", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--window", type=int, default=30, choices=range(2, 121))
    p.add_argument("--observations", type=Path, help="Controlled diagnostic: replace native boxes/keypoints with upstream NPZ")
    args = p.parse_args()
    lib = c.CDLL(str(args.library.resolve()))
    error = c.create_string_buffer(1024)

    def api(name, *values):
        status = getattr(lib, name)(*values, error, c.c_uint64(len(error)))
        if status:
            raise RuntimeError(name + ": " + error.value.decode())

    def ptr(a):
        return c.c_void_p(a.ctypes.data)

    handles = []
    def model(path, create, destroy):
        cfg = Config(str(path.resolve()).encode(), str(args.module.resolve()).encode(),
                     b"Vulkan", b"NVIDIA GeForce RTX 5070 Ti", 0, 8, 32)
        h = c.c_void_p()
        api(create, c.byref(cfg), c.byref(h))
        handles.append((destroy, h))
        return h

    results = {k: [] for k in ("boxes", "keypoints", "pred_x", "pred_cam", "body_pose",
                                "identity_coeffs", "scale_params", "global_orient", "frame_index")}
    supplied = np.load(args.observations) if args.observations else None
    paths = sorted(args.frames.glob("*.input"))
    if len(paths) < 2:
        raise ValueError("At least two packed frames required")
    times = []
    try:
        gem = model(args.gem, "gemx_session_create_live", "gemx_session_destroy")
        vit = model(args.vitpose, "gemx_vitpose_create", "gemx_vitpose_destroy") if supplied is None else None
        detector = model(args.yolox, "gemx_yolox_create", "gemx_yolox_destroy") if supplied is None else None
        for i, path in enumerate(paths):
            start_time = time.monotonic()
            data = path.read_bytes()
            if data[:8] != b"S3DIMG01":
                raise ValueError("Invalid RGB frame")
            w, h, stride = struct.unpack_from("<III", data, 8)
            rgb = np.frombuffer(data, np.uint8, offset=52).copy()
            if stride != w * 3 or len(rgb) != stride * h:
                raise ValueError("Invalid packed dimensions")
            if supplied is None:
                frame = Frame(ptr(rgb), len(rgb), w, h, stride, (c.c_float * 3)())
                detections = (Detection * 100)()
                count = c.c_uint32()
                api("gemx_yolox_detect", detector, c.byref(frame), c.c_float(.5), c.c_float(.45),
                    detections, 100, c.byref(count))
                # Upstream recreates ByteTrack for each one-frame call. Every
                # accepted detection starts a track; largest area wins.
                box = max((list(d.box) for d in detections[:count.value]),
                          key=lambda b: max(0, b[2]-b[0])*max(0, b[3]-b[1]), default=[0, 0, w-1, h-1])
                box = np.clip(np.array(box, np.float32), 0, [w-1, h-1, w-1, h-1]).astype(np.float32)
                size = np.float32(max(box[3]-box[1], (box[2]-box[0]) / np.float32(.75)) * np.float32(1.2))
                xys = np.array([(box[0]+box[2])*.5, (box[1]+box[3])*.5, size], np.float32)
                frame.box[:] = xys
                kp = np.zeros((77, 3), np.float32)
                api("gemx_vitpose_infer_rgb", vit, c.byref(frame), 1, ptr(kp), c.c_uint64(kp.size))
            else:
                xys, kp = supplied["boxes"][i], supplied["keypoints"][i]
            results["boxes"].append(xys.copy())
            results["keypoints"].append(kp.copy())
            if i == 0:
                continue
            length = min(i+1, args.window)
            boxes = np.array(results["boxes"][-length:], np.float32)
            keypoints = np.array(results["keypoints"][-length:], np.float32)
            K = np.tile(np.array([[max(w,h),0,w/2],[0,max(w,h),h/2],[0,0,1]], np.float32), (length,1,1))
            angular = np.tile(np.array([1,0,0,0,1,0], np.float32), (length,1))
            seq = Sequence(length, ptr(keypoints), ptr(boxes), ptr(K), None, ptr(angular))
            raw = np.zeros((length,585), np.float32); camera = np.zeros((length,3), np.float32)
            api("gemx_infer", gem, c.byref(seq), ptr(raw), c.c_uint64(raw.size), ptr(camera), c.c_uint64(camera.size))
            decoded = [np.zeros((length, width), np.float32) for width in (228,45,69,3,3,3,3)]
            motion = Motion(length, *(ptr(a) for a in decoded))
            api("gemx_decode_predictions", gem, c.byref(seq), ptr(raw), c.c_uint64(raw.size),
                ptr(camera), c.c_uint64(camera.size), c.byref(motion))
            for key, value in zip(("pred_x", "pred_cam", "body_pose", "identity_coeffs", "scale_params", "global_orient"),
                                  (raw, camera, decoded[0], decoded[1], decoded[2], decoded[5])):
                results[key].append(value[-1].copy())
            results["frame_index"].append(i)
            times.append(time.monotonic()-start_time)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **{k: np.array(v) for k,v in results.items()})
        print(json.dumps({"frames": len(paths), "window": args.window, "mean_step_seconds": float(np.mean(times)),
                          "note": "Includes graph warm-up; excludes model load. No camera/transport."}))
    finally:
        for destroy, handle in reversed(handles):
            getattr(lib, destroy)(handle)


if __name__ == "__main__":
    main()
