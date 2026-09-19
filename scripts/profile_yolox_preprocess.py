#!/usr/bin/env python3
"""Compare YOLOX resize CPU cost and exact output bytes between two libraries.

Requires NumPy. Run with taskset -c 0-7, separately from GPU timing runs.
"""
import argparse
import ctypes as c
import json
from pathlib import Path
import statistics
import time

import numpy as np
from replay_live_native import Frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    libraries = [c.CDLL(str(path.resolve())) for path in (a.baseline,a.candidate)]
    functions = [lib.gemx_yolox_prepare_rgb for lib in libraries]
    for fn in functions:
        fn.argtypes = [c.POINTER(Frame),c.c_void_p,c.c_uint64,c.POINTER(c.c_float),c.c_void_p,c.c_uint64]
        fn.restype = c.c_int
    rng = np.random.default_rng(917)
    sizes = [(480,270),(960,540),(1280,720),(317,191),(641,359),(13,7),(1,1),(1,640),(640,1)]
    sizes += [tuple(map(int,rng.integers(1,1500,2))) for _ in range(16)]
    report = []
    for width,height in sizes:
        # Non-packed stride also checks that axis reuse respects row padding.
        stride=width*3+7
        rgb=rng.integers(0,256,height*stride,dtype=np.uint8)
        frame=Frame(rgb.ctypes.data,rgb.size,width,height,stride,(c.c_float*3)())
        outputs=[np.empty(12*320*320,np.float32) for _ in functions]
        ratios=[c.c_float(),c.c_float()];error=c.create_string_buffer(512)
        times=[[],[]]
        for repeat in range(16):
            # Alternate the order to reduce CPU warm-up bias.
            for i in ([0,1] if repeat%2 else [1,0]):
                start=time.perf_counter()
                status=functions[i](c.byref(frame),outputs[i].ctypes.data,outputs[i].size,c.byref(ratios[i]),error,len(error))
                assert status==0,error.value
                times[i].append((time.perf_counter()-start)*1000)
        assert outputs[0].tobytes()==outputs[1].tobytes(),(width,height)
        assert ratios[0].value==ratios[1].value
        report.append(dict(width=width,height=height,baseline_median_ms=statistics.median(times[0][2:]),
                           candidate_median_ms=statistics.median(times[1][2:]),byte_identical=True))
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report[:3],indent=2))
    print(f'All {len(sizes)} sizes, including padded rows and one-pixel dimensions, are byte-identical.')

if __name__ == '__main__':
    main()
