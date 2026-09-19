#!/usr/bin/env python3
"""Execute upstream GemWebcamStreamer on stored video frames, without a robot.

Supply a local upstream webcam_stream.py, models and checkpoint. Only model
loading is redirected to local paths; its process_frame method runs unchanged.
Use upstream OpenCV 4.11.0.86 and limit CPU affinity to eight cores.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "gem-source", "video", "yolox", "vitpose", "denoiser", "checkpoint", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--strict-f32", action="store_true", help="Disable CUDA TF32 for numerical parity")
    args = p.parse_args()
    if not 2 <= args.window <= 120:
        raise ValueError("Window must be in 2..120")
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    import cv2
    if cv2.__version__ != "4.11.0":
        raise ValueError("Reference requires OpenCV 4.11.0.86")
    import onnxruntime as ort
    original_options = ort.SessionOptions
    def options():
        value = original_options(); value.intra_op_num_threads = 8; value.inter_op_num_threads = 1
        return value
    ort.SessionOptions = options
    if args.strict_f32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        original_session = ort.InferenceSession
        def strict_session(*a, **kw):
            kw["providers"] = [("CUDAExecutionProvider", {**(p[1] if isinstance(p, tuple) else {}), "use_tf32": 0})
                               if (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" else p
                               for p in kw.get("providers", [])]
            return original_session(*a, **kw)
        ort.InferenceSession = strict_session
    sys.path.insert(0, str(args.gem_source.resolve()))
    spec = importlib.util.spec_from_file_location("upstream_webcam", args.source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from scripts.demo.demo_soma_onnx import OnnxRunner
    from gem.utils.video_io_utils import read_video_np
    import gem.utils.yolox_detector as detector_module
    import gem.utils.hf_utils as hf
    original_detector = detector_module.YOLOXDetector
    detector_module.YOLOXDetector = lambda **kw: original_detector(str(args.yolox), **kw)
    hf.download_checkpoint = lambda *a, **kw: str(args.checkpoint)
    module.load_vitpose = lambda: (OnnxRunner(str(args.vitpose), device="cuda"), "onnx")
    module.load_denoiser = lambda **kw: (OnnxRunner(str(args.denoiser), device="cuda"), "onnx")
    outputs = {key: [] for key in ("boxes", "keypoints", "pred_x", "pred_cam", "body_pose",
                                  "identity_coeffs", "scale_params", "global_orient", "frame_index")}
    run = module.run_denoiser_onnx
    def record(*a, **kw):
        x, camera = run(*a, **kw)
        outputs["pred_x"].append(x[0,-1].cpu().numpy())
        outputs["pred_cam"].append(camera[0,-1].cpu().numpy())
        return x, camera
    module.run_denoiser_onnx = record
    streamer = module.GemWebcamStreamer(str(args.gem_source), window=args.window, no_imgfeat=True)
    frames = read_video_np(str(args.video))
    for index, frame in enumerate(frames):
        kp, soma = streamer.process_frame(frame, frame.shape[1], frame.shape[0])
        outputs["boxes"].append(streamer.buf_bbx[-1].cpu().numpy())
        outputs["keypoints"].append(kp.cpu().numpy())
        if soma is not None:
            for key, value in soma.items():
                outputs[key].append(value.cpu().numpy().reshape(-1))
            outputs["frame_index"].append(index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{key: np.array(value) for key, value in outputs.items()})
    manifest = {"window": args.window, "frames": len(frames), "strict_f32": args.strict_f32, "opencv": cv2.__version__,
                "rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "denoiser_sha256": hashlib.sha256(args.denoiser.read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
