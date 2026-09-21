#!/usr/bin/env python3
"""Hash all ViTPose heatmaps for real normalized crops, including their flips.

Run under taskset -c 0-7 with strict F32 flags. Requires NumPy. Compare the
same inputs and batch sizes between processes/backends with --compare.
"""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from replay_live_native import Config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','module','model','crops','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[2,8],choices=[2,4,6,8])
    p.add_argument('--compare',type=Path)
    p.add_argument('--save-heatmaps',type=Path,help='Directory for per-batch NPY heatmaps for numerical comparison')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--device-name',default='',help='Optional exact device description; otherwise select by index')
    a=p.parse_args()
    if len(os.sched_getaffinity(0))>8:p.error('restrict affinity to at most eight cores')
    for key in ('GGML_VK_DISABLE_F16','GGML_VK_DISABLE_COOPMAT','GGML_VK_DISABLE_COOPMAT2'):
        os.environ[key]='1'
    if a.save_heatmaps:a.save_heatmaps.mkdir(parents=True,exist_ok=False)
    crops=np.load(a.crops)
    assert crops.dtype==np.float32 and crops.shape[1:]==(3,256,192)
    lib=c.CDLL(str(a.library.resolve()));handle=c.c_void_p();error=c.create_string_buffer(1024)
    config=Config(str(a.model.resolve()).encode(),str(a.module.resolve()).encode(),
                  b'Vulkan',a.device_name.encode(),a.device,8,2)
    def api(name,*args):
        if getattr(lib,name)(*args,error,c.c_uint64(len(error))):raise RuntimeError(error.value.decode())
    api('gemx_vitpose_create',c.byref(config),c.byref(handle))
    report=dict(crops_sha256=hashlib.sha256(a.crops.read_bytes()).hexdigest(),frames=len(crops),batches={})
    try:
        for batch in a.batches:
            assert len(crops)%(batch//2)==0
            hashes=[];times=[]
            for start in range(0,len(crops),batch//2):
                images=np.empty((batch,3,256,192),np.float32)
                images[::2]=crops[start:start+batch//2]
                images[1::2]=images[::2,:,:,::-1]
                heatmaps=np.empty((batch,77,64,48),np.float32)
                begin=time.monotonic()
                api('gemx_vitpose_infer_normalized',handle,c.c_void_p(images.ctypes.data),c.c_uint32(batch),
                    c.c_void_p(heatmaps.ctypes.data),c.c_uint64(heatmaps.size))
                times.append((time.monotonic()-begin)*1000)
                hashes.append(hashlib.sha256(heatmaps.tobytes()).hexdigest())
                if a.save_heatmaps:np.save(a.save_heatmaps/f"batch{batch}-{start:04d}.npy",heatmaps)
            report['batches'][str(batch)]=dict(hashes=hashes,wall_ms=times)
    finally:
        lib.gemx_vitpose_destroy(handle)
    if a.compare:
        reference=json.loads(a.compare.read_text())
        assert report['crops_sha256']==reference['crops_sha256']
        for batch,values in report['batches'].items():
            assert values['hashes']==reference['batches'][batch]['hashes'], f'batch {batch} heatmaps changed'
        report['byte_identical_to']=str(a.compare)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='batches'}))


if __name__=='__main__':main()
