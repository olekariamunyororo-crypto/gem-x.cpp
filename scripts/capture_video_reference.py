#!/usr/bin/env python3
"""Capture independent upstream video stages using local, trusted model assets.

Run in the pinned GEM-X reference environment. Requires GEM-X, SOMA and SAM3D
on PYTHONPATH, and inputs/soma_assets in the working directory for contact IK.
No native detections, keypoints or Body features are reused.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("video", "output", "gem-source", "yolox", "vitpose", "body-checkpoint", "mhr", "checkpoint"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--strict-f32", action="store_true", help="Disable CUDA TF32 in the independent reference")
    args = p.parse_args()
    import cv2
    if cv2.__version__ != "4.11.0":
        raise RuntimeError("Use upstream's pinned opencv-python(-headless)==4.11.0.86")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    # Restrict ORT's own pools as well as the enclosing taskset/container.
    import onnxruntime as ort
    original_options = ort.SessionOptions
    def bounded_options():
        options = original_options()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        return options
    ort.SessionOptions = bounded_options
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
    from gem.utils.video_io_utils import read_video_np
    from gem.utils.yolox_detector import YOLOXDetector, detect_and_track
    from gem.utils.kp2d_utils import smooth_bbx_xyxy
    from gem.utils.geo_transform import get_bbx_xys_from_xyxy, compute_cam_angvel
    from gem.utils.cam_utils import estimate_K
    from scripts.demo.demo_soma_onnx import OnnxRunner, run_vitpose_onnx
    from gem.utils.sam3db_extractor import SAM3DBExtractor
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    args.output.mkdir(parents=True, exist_ok=True)
    prep = args.output / "preprocess"
    prep.mkdir(exist_ok=True)
    frames = read_video_np(str(args.video))
    length, height, width, _ = frames.shape
    # Export only source pixels and generic camera metadata for the native
    # runner. It will independently detect boxes and extract observations.
    packed = args.output / "frames"
    packed.mkdir(exist_ok=True)
    focal = math.hypot(width, height)
    header = b"S3DIMG01" + struct.pack("<III8f", width, height, width*3,
        0,0,width-1,height-1,focal,focal,width/2,height/2)
    for index, frame in enumerate(frames):
        (packed / f"{index:06d}.input").write_bytes(header + frame.tobytes())
    timings = {}
    start = time.monotonic()
    detector = YOLOXDetector(str(args.yolox), device="cuda")
    boxes, ids = detect_and_track(frames, detector)
    boxes = smooth_bbx_xyxy(torch.from_numpy(boxes), window=5)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height - 1)
    xys = get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()
    torch.save({"bbx_xyxy": boxes, "bbx_xys": xys, "track_ids": ids}, prep / "bbx.pt")
    timings["detector_seconds"] = time.monotonic() - start
    del detector
    gc.collect()
    start = time.monotonic()
    vitpose = OnnxRunner(str(args.vitpose), device="cuda")
    with torch.inference_mode():
        kp = run_vitpose_onnx(vitpose, "onnx", frames, xys, batch_size=1).cpu()
    torch.save(kp, prep / "vitpose.pt")
    timings["vitpose_seconds"] = time.monotonic() - start
    del vitpose
    gc.collect()
    torch.cuda.empty_cache()
    start = time.monotonic()
    extractor = SAM3DBExtractor(checkpoint_path=str(args.body_checkpoint),
                                mhr_path=str(args.mhr), device="cuda:0")
    with torch.inference_mode():
        body = extractor.extract_video_features(str(args.video), xys, batch_size=1)
    features = body["pose_tokens"].detach().cpu()
    torch.save({"pose_tokens": features}, prep / "vit_features.pt")
    timings["body_seconds"] = time.monotonic() - start
    del extractor, body
    gc.collect()
    torch.cuda.empty_cache()
    K = estimate_K(width, height).repeat(length, 1, 1)
    data = {"meta": [{"vid": args.video.stem}], "length": torch.tensor(length),
            "bbx_xys": xys, "kp2d": kp, "K_fullimg": K,
            "cam_angvel": compute_cam_angvel(torch.eye(3).repeat(length, 1, 1)),
            "cam_tvel": torch.zeros(length, 3), "R_w2c": torch.eye(3).repeat(length, 1, 1),
            "f_imgseq": features, "has_text": torch.tensor([False]),
            "mask": {name: torch.full((length,), value, dtype=torch.bool) for name, value in
                     (("valid", True), ("has_img_mask", True), ("has_2d_mask", True),
                      ("has_cam_mask", True), ("has_audio_mask", False), ("has_music_mask", False))}}
    with initialize_config_dir(version_base="1.3", config_dir=str(args.gem_source / "configs")):
        cfg = compose(config_name="demo_soma", overrides=["exp=gem_soma_regression",
                      "video_name=audit", "video_path=" + str(args.video), "use_wandb=false", "task=test"])
    model = instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(str(args.checkpoint))
    model = model.eval().cuda()

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(v) for v in value)
        return value

    with torch.inference_mode():
        for postproc, filename in ((False, "hpe_results_nopost.pt"), (True, "hpe_results.pt")):
            start = time.monotonic()
            torch.save(cpu(model.predict(data, static_cam=True, postproc=postproc)), args.output / filename)
            timings[filename] = time.monotonic() - start
    manifest = {"frames": length, "strict_f32": args.strict_f32, "rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                "opencv": cv2.__version__, "onnxruntime": ort.__version__, "torch": torch.__version__,
                "body_batch": 1, "vitpose_batch": 1, "detector_score_threshold": 0.5,
                "detector_nms_threshold": 0.45, "timings_including_load": timings}
    (args.output / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
