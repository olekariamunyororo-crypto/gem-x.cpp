#!/usr/bin/env python3
"""Pack independent upstream checkpoint outputs into the contact test fixture.

Inputs are trusted local hpe_results_nopost.pt and hpe_results.pt captures
produced by capture_video_reference.py. No contact equations are duplicated.
"""
import argparse
import hashlib
from pathlib import Path
import struct
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path); p.add_argument("after", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args()
    torch.set_num_threads(8)
    before = torch.load(a.before, weights_only=False, map_location="cpu")
    after = torch.load(a.after, weights_only=False, map_location="cpu")
    src, dst = before["body_params_global"], after["body_params_global"]
    frames = src["body_pose"].reshape(-1,228).shape[0]
    arrays = [src[k] for k in ("body_pose", "identity_coeffs", "scale_params", "global_orient", "transl")]
    arrays += [before["net_outputs"]["static_conf_logits"], dst["body_pose"], dst["transl"]]
    data = b"GEMCONT1" + struct.pack("<I", frames)
    for value, width in zip(arrays, (228,45,69,3,3,6,228,3)):
        value = value.numpy().reshape(frames,width).astype("<f4")
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite reference")
        data += value.tobytes()
    a.output.write_bytes(data)
    print(a.output, hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
