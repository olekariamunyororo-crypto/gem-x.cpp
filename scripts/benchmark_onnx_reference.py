#!/usr/bin/env python3
"""Benchmark existing ONNX models with CUDA, independently of the native backend.

Run in the existing CUDA reference environment with affinity limited to eight
cores. Timings include host input/output transfers but exclude preprocessing.
This script is an experiment only; the application does not depend on ORT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import time

import numpy as np
import torch  # Load the reference environment's CUDA/cuDNN dependencies first.
import onnxruntime as ort


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--kind',choices=['vitpose','yolox'],required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--expected',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tf32',type=int,choices=[0,1],default=0)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--warmups',type=int,default=20)
    p.add_argument('--profile',action='store_true',help='Export ORT node execution trace; timings are instrumented')
    p.add_argument('--cuda-trace',type=Path,help='Export CUPTI kernel timeline through torch.profiler')
    p.add_argument('--cuda-profile',action='store_true',help='Bracket timed iterations with CUDA profiler API calls')
    p.add_argument('--save-output',type=Path,help='Save first output as raw F32')
    p.add_argument('--ready-file',type=Path,help='Signal completion of warm-up to an external GPU sampler')
    a=p.parse_args()
    if len(os.sched_getaffinity(0))>8 or a.iterations<1 or a.warmups<0:p.error('use at most eight cores, positive iterations and nonnegative warmups')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if a.kind=='vitpose':
        data=a.input.read_bytes();assert data[:8]==b'VITPREP1'
        crop=np.frombuffer(data,dtype='<f4',offset=28).reshape(3,256,192)
        images=np.ascontiguousarray(np.stack([crop,crop[:,:,::-1]]))
    else:
        import cv2
        data=a.input.read_bytes();assert data[:8]==b'S3DIMG01'
        w,h,stride=struct.unpack_from('<III',data,8);assert stride==w*3
        rgb=np.frombuffer(data,np.uint8,offset=52).reshape(h,w,3)
        ratio=min(640/h,640/w)
        padded=np.full((640,640,3),114,np.uint8)
        resized=cv2.resize(rgb[:,:,::-1],(int(w*ratio),int(h*ratio)))
        padded[:resized.shape[0],:resized.shape[1]]=resized
        images=np.ascontiguousarray(padded.transpose(2,0,1)[None],dtype=np.float32)
    options=ort.SessionOptions();options.intra_op_num_threads=8;options.inter_op_num_threads=1
    options.log_severity_level=3
    if a.profile:
        options.enable_profiling=True
        options.profile_file_prefix=str(a.output.with_suffix(''))
    begin=time.perf_counter()
    session=ort.InferenceSession(str(a.model),sess_options=options,
        providers=[('CUDAExecutionProvider',{'use_tf32':a.tf32}),'CPUExecutionProvider'])
    if session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CUDA provider unavailable; refusing CPU fallback benchmark')
    load=time.perf_counter()-begin
    inputs={session.get_inputs()[0].name:images}
    for _ in range(a.warmups):session.run(None,inputs)
    if a.ready_file:a.ready_file.write_text('ready\n')
    profiler=None
    if a.cuda_trace:
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
        profiler.__enter__()
    if a.cuda_profile:torch.cuda.cudart().cudaProfilerStart()
    timings=[]
    for _ in range(a.iterations):
        begin=time.perf_counter();outputs=session.run(None,inputs)
        timings.append((time.perf_counter()-begin)*1000)
    if a.cuda_profile:torch.cuda.cudart().cudaProfilerStop()
    if profiler:
        profiler.__exit__(None,None,None)
        profiler.export_chrome_trace(str(a.cuda_trace))
    if a.save_output:outputs[0].tofile(a.save_output)
    r={'model':str(a.model),'kind':a.kind,'ort_version':ort.__version__,
       'providers':session.get_providers(),'provider_options':session.get_provider_options(),
       'tf32':a.tf32,'cpu_affinity':sorted(os.sched_getaffinity(0)),
       'input_shape':list(images.shape),'input_sha256':hashlib.sha256(images.tobytes()).hexdigest(),
       'scope':'ORT session.run, including host transfers; excludes preprocess and native postprocess',
       'load_seconds':load,'warmups':a.warmups,'iterations':a.iterations,
       'instrumented':bool(a.profile or a.cuda_trace or a.cuda_profile),
       'mean_ms':statistics.mean(timings),'median_ms':statistics.median(timings),
       'p95_ms':sorted(timings)[min(len(timings)-1,int(.95*len(timings)))],
       'output_shapes':[list(x.shape) for x in outputs],
       'output_sha256':[hashlib.sha256(x.tobytes()).hexdigest() for x in outputs]}
    if a.expected:
        expected=np.frombuffer(a.expected.read_bytes(),dtype='<f4',offset=12).reshape(outputs[0].shape)
        diff=np.abs(outputs[0]-expected)
        peaks=outputs[0].reshape(2,77,-1).argmax(-1);ref=expected.reshape(2,77,-1).argmax(-1)
        r['fixture_error']={'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
                            'changed_raw_heatmap_peaks':int(np.count_nonzero(peaks!=ref))}
    if a.profile:r['profile_path']=session.end_profiling()
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)


if __name__=='__main__':main()
