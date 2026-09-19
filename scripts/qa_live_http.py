#!/usr/bin/env python3
"""Exercise a running demo's real resident worker with lossless video frames.

Requires NumPy/OpenCV, packed source frames, and a previously accepted native
live replay NPZ. Tests HTTP transport plus identical observations and newest-
frame skeletons, including rolling-window eviction. No camera is required.
"""
import argparse
import ctypes as c
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from replay_live_native import Config, Motion

class Skeleton(c.Structure):
    _fields_ = [("frames", c.c_uint32)] + [(n, c.c_void_p) for n in
        ("positions", "rotations", "parents", "translations")]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8099')
    for name in ('frames', 'reference', 'library', 'module', 'gem', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    reference = np.load(a.reference)
    lib = c.CDLL(str(a.library.resolve()))
    error = c.create_string_buffer(1024)
    def api(name, *args):
        if getattr(lib, name)(*args, error, c.c_uint64(len(error))):
            raise RuntimeError(error.value.decode())
    cfg = Config(str(a.gem.resolve()).encode(), str(a.module.resolve()).encode(), b'CPU', b'', 0, 8, 1)
    handle = c.c_void_p()
    api('gemx_session_create_live', c.byref(cfg), c.byref(handle))
    def ptr(v): return c.c_void_p(v.ctypes.data)
    def request(method, path, data=None):
        return urllib.request.urlopen(urllib.request.Request(a.url+path, data=data, method=method), timeout=120)
    path = None
    errors, timings, native_timings = [], [], []
    try:
        with request('POST', '/api/live') as response:
            path = '/api/live/'+json.load(response)['id']
        try:
            request('POST', '/api/live')
            raise AssertionError('second live session was accepted')
        except urllib.error.HTTPError as e:
            assert e.code == 409
        for i, source in enumerate(sorted(a.frames.glob('*.input'))):
            data = source.read_bytes()
            width, height, stride = struct.unpack_from('<III', data, 8)
            assert stride == width*3
            rgb = np.frombuffer(data, np.uint8, offset=52).reshape(height,width,3)
            ok, png = cv2.imencode('.png', rgb[:,:,::-1]); assert ok
            start = time.monotonic()
            with request('PUT', path+'/frame', png.tobytes()) as response:
                payload = response.read()
                assert int(response.headers['X-GEMX-Sequence']) == i+1
                if i == 0:
                    assert response.status == 204
                    continue
                assert response.status == 200 and payload[:8] == b'GEMPOSE2'
                native_timings.append(float(response.headers['X-GEMX-Inference-Ms'])/1000)
            timings.append(time.monotonic()-start)
            values = np.frombuffer(payload, '<f4', offset=8)
            assert np.isfinite(values[:1001]).all()
            observed = values[-231:].reshape(77,3)
            kp_error = np.max(np.abs(observed-reference['keypoints'][i]))
            zero = np.zeros(3,np.float32)
            body, identity, scale, orient = [np.ascontiguousarray(reference[key][i-1]) for key in
                ('body_pose','identity_coeffs','scale_params','global_orient')]
            motion = Motion(1,ptr(body),ptr(identity),ptr(scale),ptr(orient),ptr(zero),ptr(orient),ptr(zero))
            positions = np.zeros((77,3),np.float32)
            rotations = np.zeros((77,4),np.float32)
            skeleton = Skeleton(1,ptr(positions),ptr(rotations),None,None)
            api('gemx_build_skeleton',handle,c.byref(motion),c.byref(skeleton))
            joint_error = np.max(np.linalg.norm(values[:231].reshape(77,3)-positions,axis=1))
            errors.append((float(kp_error),float(joint_error)))
        assert len(errors) >= 30, 'need eviction coverage'
        maxima = np.max(errors,axis=0)
        assert maxima[0] < 1e-4, maxima
        assert maxima[1] < 1e-4, maxima
        report = dict(poses=len(errors),keypoint_component_max=float(maxima[0]),
            joint_max_metres=float(maxima[1]),mean_http_seconds=float(np.mean(timings)),
            p95_http_seconds=float(np.percentile(timings,95)),
            mean_native_seconds=float(np.mean(native_timings)),
            mean_other_http_seconds=float(np.mean(np.array(timings)-native_timings)),
            note='Lossless frame HTTP replay, not physical camera latency; strict F32.')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    finally:
        if path:
            with request('DELETE',path): pass
        lib.gemx_session_destroy(handle)

if __name__ == '__main__':
    main()
