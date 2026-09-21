#!/usr/bin/env python3
"""Screen QKV microtiles and packed FFN against the accepted strict-F32 configuration."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--layouts',nargs='+',choices=['128x64-t256','64x64-t128','64x32-t128','128x32-w1','128x128-t256','128x64-w1','packed-ffn','packed-ffn-qkv'],default=['128x64-t256','64x64-t128','64x32-t128','128x32-w1','128x128-t256','128x64-w1'])
    a=p.parse_args()
    if len(os.sched_getaffinity(0))>8 or a.iterations<1:p.error('use at most eight cores and positive iterations')
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_VK_','GEMX_VITPOSE_','GEMX_BENCHMARK_'))}
    env.pop('LD_PRELOAD',None)
    env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1',
        GEMX_VITPOSE_QKV_LAYOUT='0',GEMX_VITPOSE_FLATTEN='1',GEMX_VITPOSE_NORM='1',GEMX_VITPOSE_SWIGLU='1',GEMX_VITPOSE_RECT_GROUP='up',GEMX_VITPOSE_RECT='64x64',
        GEMX_BENCHMARK_WARMUP='10',GEMX_BENCHMARK_INPUT=str(a.input.resolve()),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    report={'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'iterations':a.iterations,'warmups':10,
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'env':{k:v for k,v in env.items() if k.startswith(('GEMX_','GGML_VK_'))},'runs':{}}
    for layout in ['baseline',*a.layouts,'baseline-repeat']:
        folder=out/layout;folder.mkdir()
        extra={} if layout.startswith('baseline') else {'GEMX_VITPOSE_QKV_LAYOUT':layout}
        if layout=='packed-ffn':extra={'GEMX_VITPOSE_PACK_FFN':'1'}
        if layout=='packed-ffn-qkv':extra={'GEMX_VITPOSE_PACK_FFN':'1','GEMX_VITPOSE_QKV_LAYOUT':'64x64-t128'}
        local=dict(env,**extra,GEMX_BENCHMARK_PROFILE=str(folder/'iterations.csv'),GEMX_BENCHMARK_OUTPUT=str(folder/'heatmaps.f32'),GGML_VK_PIPELINE_STATS='gemx_qkv_microtile')
        cmd=[str(root/'build/vulkan/gemx-vitpose-benchmark'),str(root/'generated/reference/vitpose-f32.gguf'),str(root/'build/vulkan/bin/libggml-vulkan.so'),'Vulkan','8','2',str(a.iterations)]
        with (folder/'run.log').open('w') as f:subprocess.run(cmd,cwd=root,env=local,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
        times=[float(r['wall_ms']) for r in csv.DictReader((folder/'iterations.csv').open())]
        r={'env':extra,'mean_ms':statistics.mean(times),'median_ms':statistics.median(times),'heatmaps_sha256':hashlib.sha256((folder/'heatmaps.f32').read_bytes()).hexdigest()}
        report['runs'][layout]=r;r['byte_identical']=r['heatmaps_sha256']==report['runs']['baseline']['heatmaps_sha256']
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(layout,json.dumps(r),flush=True)


if __name__=='__main__':main()
