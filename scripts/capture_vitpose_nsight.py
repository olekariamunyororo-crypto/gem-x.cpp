#!/usr/bin/env python3
"""Capture Vulkan counters/warp states with a locally installed Nsight Graphics.

Run under taskset -c 0-7. --sudo uses administrator counter access without
changing driver permissions. GPU clocks are explicitly left unaltered.
Requires Nsight Graphics with Blackwell GPU Trace support.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ngfx',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library-path',help='Extra runtime library path, e.g. on NixOS')
    p.add_argument('--sudo',action='store_true')
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--optimized',action='store_true',help='Enable the validated combined matrices, fusions and expansion tiles')
    p.add_argument('--qkv-tile',choices=['128x64','64x64'],help='Diagnostic QKV tile comparison; requires --qkv-only --optimized')
    p.add_argument('--qkv-only',action='store_true',help='Replay only the first real QKV matmul using profiling preload')
    p.add_argument('--input',type=Path,help='Raw normalized NCHW F32 input, batch two')
    p.add_argument('--duration-ms',type=int,default=250)
    p.add_argument('--no-pc-sampling',action='store_true',help='Collect additional SM/cache metrics instead')
    a=p.parse_args()
    if len(os.sched_getaffinity(0))>8:p.error('restrict affinity to at most eight cores')
    if a.baseline and a.optimized:p.error('baseline and optimized are mutually exclusive')
    if a.qkv_tile and not (a.qkv_only and a.optimized):p.error('--qkv-tile requires --qkv-only --optimized')
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    envs=['NV_AGORA_FORCE_BREAKPAD=-1','XDG_CACHE_HOME=/tmp/gemx-nsight-cache',
          'XDG_CONFIG_HOME=/tmp/gemx-nsight-config']
    if a.library_path:envs.append('LD_LIBRARY_PATH='+a.library_path)
    command=(['sudo','-n'] if a.sudo else [])+['taskset','-c',','.join(map(str,sorted(os.sched_getaffinity(0)))),
             'env',*envs,str(a.ngfx.resolve()),'--activity','GPU Trace Profiler',
             '--exe',str(root/'build/vulkan/gemx-vitpose-benchmark'),'--dir',str(root),
             '--args',shlex.join([str(root/'generated/reference/vitpose-f32.gguf'),
                                 str(root/'build/vulkan/bin/libggml-vulkan.so'),'Vulkan','8','2','500']),
             '--env','GGML_VK_DISABLE_F16=1;GGML_VK_DISABLE_COOPMAT=1;GGML_VK_DISABLE_COOPMAT2=1;'
                     'GGML_VK_DEBUG_MARKERS=1;GEMX_BENCHMARK_WARMUP=20;GEMX_VITPOSE_TILES='+('0' if a.baseline else '1')+';',
             '--output-dir',str(out),'--start-after-ms','4000','--max-duration-ms',str(a.duration_ms),
             '--architecture','Blackwell GB20x','--metric-set-id','0','--auto-export',
             '--set-gpu-clocks','unaltered','--collect-screenshot','0','--trace-timeout','60']
    if not a.no_pc_sampling:command.append('--real-time-shader-profiler')
    if a.optimized:
        command[command.index('--env')+1]+='GEMX_VITPOSE_FLATTEN=1;GEMX_VITPOSE_NORM=1;GEMX_VITPOSE_SWIGLU=1;GEMX_VITPOSE_RECT_GROUP=up;GEMX_VITPOSE_RECT=64x64;'
    if a.qkv_only:command[command.index('--env')+1]+='LD_PRELOAD='+str(root/'build/vulkan/libgemx-profile-qkv.so')+';'
    if a.qkv_tile:
        index=command.index('--env')+1
        command[index]=command[index].replace('GEMX_VITPOSE_RECT_GROUP=up;GEMX_VITPOSE_RECT=64x64;', 'GEMX_VITPOSE_RECT_GROUP=qkv;GEMX_VITPOSE_RECT='+a.qkv_tile+';')
    if a.input:command[command.index('--env')+1]+='GEMX_BENCHMARK_INPUT='+str(a.input.resolve())+';'
    with (out/'capture.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=log,timeout=120)
    if result.returncode or not (out/'BASE_UNLOCKED/GPUTRACE_FRAME.xls').is_file():
        raise RuntimeError('capture/export failed; see '+str(out/'capture.log'))
    print('Captured '+str(out),flush=True)


if __name__=='__main__':main()
