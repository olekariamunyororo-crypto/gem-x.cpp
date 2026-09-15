#!/usr/bin/env python3
"""Generate deterministic parity fixtures with the official ONNX model."""

import argparse
import hashlib
from pathlib import Path
import struct

import numpy as np
import onnxruntime as ort

EXPECTED_ONNX_SHA256 = "20aab83c01bbd909a258ad0fa465458eda382c63dc80d56952b2ec952d6e192c"
EXPECTED_DATA_SHA256 = "720c2400281b730574bf2a446c33ecd77bf494ac8161b081b09bb122b9fad2f8"
LENGTHS = (1, 2, 16, 30, 120)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inputs(frames: int) -> dict[str, np.ndarray]:
    f = np.arange(frames, dtype=np.float32)[:, None]
    j = np.arange(77, dtype=np.float32)[None, :]
    cx = np.float32(320.0) + np.float32(22.0) * np.sin(f * np.float32(0.071))
    cy = np.float32(240.0) + np.float32(17.0) * np.cos(f * np.float32(0.053))
    size = np.float32(360.0) + np.float32(11.0) * np.sin(f * np.float32(0.037))
    x = cx + size * np.float32(0.34) * np.sin(f * np.float32(0.029) + j * np.float32(0.131))
    y = cy + size * np.float32(0.31) * np.cos(f * np.float32(0.023) + j * np.float32(0.113))
    confidence = np.full((frames, 77), np.float32(0.93), dtype=np.float32)
    confidence[((np.arange(frames)[:, None] * 77 + np.arange(77)) % 13) == 0] = np.float32(0.2)
    outside = ((np.arange(frames)[:, None] * 77 + np.arange(77)) % 41) == 0
    x[outside] = np.broadcast_to(cx + size, (frames, 77))[outside]
    obs = np.stack((x, y, confidence), axis=-1).astype(np.float32)
    boxes = np.concatenate((cx, cy, size), axis=1).astype(np.float32)

    intrinsics = np.zeros((frames, 3, 3), dtype=np.float32)
    intrinsics[:, 0, 0] = np.float32(900.0)
    intrinsics[:, 1, 1] = np.float32(900.0)
    intrinsics[:, 0, 2] = np.float32(320.0)
    intrinsics[:, 1, 2] = np.float32(240.0)
    intrinsics[:, 2, 2] = np.float32(1.0)

    d = np.arange(1024, dtype=np.float32)[None, :]
    feature = (np.float32(0.25) * np.sin(f * np.float32(0.067) + d * np.float32(0.0031))
               + np.float32(0.1) * np.cos(d * np.float32(0.0127))).astype(np.float32)
    angular = np.empty((frames, 6), dtype=np.float32)
    angular[:, 0] = np.float32(1.0) + np.float32(0.0002) * np.sin(f[:, 0] * np.float32(0.1))
    angular[:, 1] = np.float32(0.02) * np.sin(f[:, 0] * np.float32(0.05))
    angular[:, 2] = np.float32(0.01) * np.cos(f[:, 0] * np.float32(0.04))
    angular[:, 3] = np.float32(0.015) * np.sin(f[:, 0] * np.float32(0.03))
    angular[:, 4] = np.float32(1.0) + np.float32(0.00015) * np.cos(f[:, 0] * np.float32(0.09))
    angular[:, 5] = np.float32(0.012) * np.cos(f[:, 0] * np.float32(0.06))
    return {"obs": obs[None], "bbx_xys": boxes[None], "K_fullimg": intrinsics[None],
            "f_imgseq": feature[None], "f_cam_angvel": angular[None]}


def write_f32(stream, value: np.ndarray) -> None:
    stream.write(np.ascontiguousarray(value, dtype="<f4").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = args.model.resolve()
    data = model.with_name(model.name + ".data")
    if digest(model) != EXPECTED_ONNX_SHA256 or digest(data) != EXPECTED_DATA_SHA256:
        raise ValueError("reference model does not match the pinned NVIDIA artifact")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    maximum = inputs(max(LENGTHS))
    outputs = []
    for length in LENGTHS:
        feed = {name: value[:, :length].copy() for name, value in maximum.items()}
        pred_x, pred_cam = session.run(["pred_x", "pred_cam"], feed)
        if not np.isfinite(pred_x).all() or not np.isfinite(pred_cam).all():
            raise ValueError(f"non-finite reference output for length {length}")
        outputs.append((pred_x, pred_cam))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"GEMXREF1")
        stream.write(struct.pack("<7I", 1, max(LENGTHS), len(LENGTHS), 77, 1024, 585, 3))
        stream.write(struct.pack(f"<{len(LENGTHS)}I", *LENGTHS))
        for name in ("obs", "bbx_xys", "K_fullimg", "f_imgseq", "f_cam_angvel"):
            write_f32(stream, maximum[name])
        for pred_x, pred_cam in outputs:
            write_f32(stream, pred_x)
            write_f32(stream, pred_cam)
    print(f"wrote {args.output} ({digest(args.output)})")


if __name__ == "__main__":
    main()
