#!/usr/bin/env python3
"""Regenerate RGB->ViTPose fixtures through the actual pinned upstream helpers.

Use the reference environment with GEM-X and SOMA on PYTHONPATH. CPU ONNX
Runtime supplies independent network outputs; no native outputs are used.
"""

import argparse
import hashlib
from pathlib import Path
import struct

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("reference", type=Path)
    args = p.parse_args()
    import cv2
    if cv2.__version__ != "4.11.0":
        raise RuntimeError("Use upstream's pinned opencv-python(-headless)==4.11.0.86")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    import onnxruntime as ort
    from scripts.demo.demo_soma_onnx import vitpose_preprocess, vitpose_postprocess
    from gem.utils.vitpose_extractor import flip_heatmap_soma77
    prep_path = args.reference / "vitpose_preprocess_reference.bin"
    old = prep_path.read_bytes()
    if old[:8] != b"VITPREP1":
        raise ValueError("Missing input dimensions/box fixture")
    width, height, cx, cy, size = struct.unpack_from("<II3f", old, 8)
    y, x = np.mgrid[:height, :width]
    frame = np.stack(((x * 17 + y * 3 + 11) % 256, (x * 5 + y * 13 + 29) % 256,
                      (x * 7 + y * 19 + 47) % 256), -1).astype(np.uint8)
    boxes = torch.tensor([[cx, cy, size]])
    normalized, _ = vitpose_preprocess(frame[None], boxes)
    cropped = normalized[:, :, :, 32:224].contiguous()
    batch = torch.cat((cropped, cropped.flip(3)), dim=0).numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    heatmaps = session.run(["heatmaps"], {"imgs": batch})[0]
    fused = (torch.from_numpy(heatmaps[:1]) + flip_heatmap_soma77(torch.from_numpy(heatmaps[1:]))) * .5
    keypoints = vitpose_postprocess(fused.numpy(), boxes).numpy()
    values = {
        prep_path: old[:28] + cropped.numpy().astype("<f4").tobytes(),
        args.reference / "vitpose_heatmap_reference.bin": b"VITHEAT1" + struct.pack("<I", 2) + heatmaps.astype("<f4").tobytes(),
        args.reference / "vitpose_keypoint_reference.bin": b"VITKEYP1" + keypoints.astype("<f4").tobytes(),
    }
    for path, data in values.items():
        path.write_bytes(data)
        print(path, hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
