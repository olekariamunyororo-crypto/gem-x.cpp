#!/usr/bin/env python3
"""Compare live replay observations, newest-frame outputs and SOMA joints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

from audit_video_parity import distances
from compare_e2e_parity import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "native", "soma-source", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    a,b = np.load(args.native),np.load(args.upstream)
    if not np.array_equal(a["frame_index"],b["frame_index"]):
        raise ValueError("Live output frame indices differ")
    sys.path.insert(0,str(args.soma_source.resolve()))
    import soma.soma as sm
    sm.CorrectivesMLP.load_checkpoint = staticmethod(lambda *a, **kw: None)
    soma = sm.SOMALayer(args.assets, low_lod=True, device="cpu", identity_model_type="mhr", mode="torch")
    def joints(data):
        identity=torch.from_numpy(data["identity_coeffs"])
        scales=torch.from_numpy(data["scale_params"])
        # The live publisher prepares the current frame's shape, not a mean
        # over the emitted stream (each prediction has its own past window).
        soma.prepare_identity(identity,scales[:,1:],repose_to_bind_pose=False,global_scale=scales[:,:1])
        pose=torch.cat((torch.from_numpy(data["global_orient"])[:,None],
                        torch.from_numpy(data["body_pose"]).reshape(-1,76,3)),1)
        return soma.pose(pose,transl=torch.zeros(len(pose),3),apply_correctives=False)["joints"].numpy()
    with torch.inference_mode():
        ja,jb=joints(a),joints(b)
    report={"frames":len(a["frame_index"]),"native":str(args.native),"upstream":str(args.upstream),
            "native_sha256":hashlib.sha256(args.native.read_bytes()).hexdigest(),
            "upstream_sha256":hashlib.sha256(args.upstream.read_bytes()).hexdigest(),
            "metrics":{}}
    for key in ("boxes","pred_x","pred_cam","body_pose","identity_coeffs","scale_params","global_orient"):
        report["metrics"][key]=statistics(a[key],b[key])
    report["metrics"]["keypoint_xy_pixels"]=distances(a["keypoints"][...,:2],b["keypoints"][...,:2],1)
    report["metrics"]["keypoint_confidence"]=statistics(a["keypoints"][...,2],b["keypoints"][...,2])
    report["metrics"]["pelvis_relative_joints_mm"]=distances(ja-ja[:,:1],jb-jb[:,:1])
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print({k:{n:v for n,v in metric.items() if n!='per_frame_mean'} for k,metric in report["metrics"].items()})


if __name__ == "__main__":
    main()
