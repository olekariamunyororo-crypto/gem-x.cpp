#!/usr/bin/env python3
"""Compare every pose byte from serial and two-frame HTTP live replay.

Uses lossless PNGs generated with the standard library. No inference backend
or precision changes. Run under taskset -c 0-7 against an idle demo server.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request
import zlib


def png(source):
    data = source.read_bytes()
    width, height, stride = struct.unpack_from('<III', data, 8)
    assert stride == width * 3 and len(data) == 52 + stride * height
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    rows = b''.join(b'\0' + data[52+y*stride:52+(y+1)*stride] for y in range(height))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8099')
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--detect-interval', type=int, default=5)
    p.add_argument('--repeats', type=int, default=2)
    a = p.parse_args()
    if len(os.sched_getaffinity(0)) > 8: p.error('restrict CPU affinity to eight cores')
    frames = [png(f) for f in sorted(a.frames.glob('*.input'))] * a.repeats
    assert len(frames) >= 32
    def request(method, path, data=None, headers=None):
        return urllib.request.urlopen(urllib.request.Request(a.url+path, data=data, method=method, headers=headers or {}), timeout=120)
    reports = {}
    for mode in ('serial', 'pipeline'):
        with request('POST', f'/api/live?detect_interval={a.detect_interval}'+('&pipeline=2' if mode=='pipeline' else '')) as response:
            path = '/api/live/'+json.load(response)['id']
        def frame(index):
            start = time.monotonic()
            with request('PUT', path+'/frame', frames[index], {'X-GEMX-Frame':str(index+1)}) as response:
                payload = response.read()
                assert int(response.headers['X-GEMX-Sequence']) == index+1
                assert response.status == (204 if index==0 else 200)
                return dict(index=index, sha256=hashlib.sha256(payload).hexdigest(), seconds=time.monotonic()-start, server_timing=response.headers['Server-Timing'])
        start = time.monotonic()
        try:
            if mode == 'serial':
                results = [frame(i) for i in range(len(frames))]
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    pending = [pool.submit(frame, i) for i in range(2)]
                    results = []
                    for i in range(len(frames)):
                        results.append(pending.pop(0).result())
                        if i+2 < len(frames): pending.append(pool.submit(frame, i+2))
            reports[mode] = dict(seconds=time.monotonic()-start, frames=results)
        finally:
            with request('DELETE', path): pass
    assert [r['sha256'] for r in reports['serial']['frames']] == [r['sha256'] for r in reports['pipeline']['frames']], 'pose bytes changed'
    reports.update(byte_identical=True, poses=len(frames)-1, detect_interval=a.detect_interval,
        note='Fixed lossless input sequence; includes rolling-window eviction. Full lookahead stress test, not browser latency.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps({k:v for k,v in reports.items() if k not in ('serial','pipeline')}))

if __name__ == '__main__': main()
