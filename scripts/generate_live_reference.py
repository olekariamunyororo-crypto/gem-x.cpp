#!/usr/bin/env python3
"""Pack a capture_live_reference.py NPZ into the native rolling regression fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--width",type=int,required=True);p.add_argument("--height",type=int,required=True)
    a=p.parse_args();x=np.load(a.capture);meta=json.loads(a.capture.with_suffix('.json').read_text())
    frames=len(x['boxes'])
    if not np.array_equal(x['frame_index'],np.arange(1,frames)):
        raise ValueError("Expected a two-frame warm-up and every subsequent endpoint")
    data=b'GEMLIVE1'+struct.pack('<4I',frames,meta['window'],a.width,a.height)
    for key in ('keypoints','boxes','pred_x','pred_cam','body_pose','identity_coeffs','scale_params','global_orient'):
        v=x[key].astype('<f4')
        if not np.isfinite(v).all():raise ValueError('Nonfinite reference')
        data+=v.tobytes()
    a.output.write_bytes(data);print(a.output,hashlib.sha256(data).hexdigest())


if __name__=='__main__':main()
