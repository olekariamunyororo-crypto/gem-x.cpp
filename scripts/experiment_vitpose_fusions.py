#!/usr/bin/env python3
"""Screen opt-in strict-F32 ViTPose fusions and projection tiles on eight cores."""
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
    p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--cases',nargs='+')
    a=p.parse_args()
    if len(os.sched_getaffinity(0))>8 or a.iterations<1:p.error('use at most eight cores and positive iterations')
    root=Path(__file__).resolve().parents[1];a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_VK_','GEMX_VITPOSE_','GEMX_BENCHMARK_'))}
    env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1',
               GEMX_VITPOSE_NORM='0',GEMX_VITPOSE_SWIGLU='0',GEMX_VITPOSE_RECT='0',GEMX_VITPOSE_QKV_LAYOUT='0',
               GEMX_VITPOSE_FLATTEN='1',GEMX_BENCHMARK_WARMUP='10',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    plans=[('baseline',{}),('swiglu',{'GEMX_VITPOSE_SWIGLU':'1'}),('norm',{'GEMX_VITPOSE_NORM':'1'}),
           ('fusions',{'GEMX_VITPOSE_SWIGLU':'1','GEMX_VITPOSE_NORM':'1'})]
    plans.append(('combined',{'GEMX_VITPOSE_SWIGLU':'1','GEMX_VITPOSE_NORM':'1',
                             'GEMX_VITPOSE_RECT_GROUP':'up','GEMX_VITPOSE_RECT':'64x64'}))
    for group in ['up','qkv','proj','down']:
        for tile in ['128x32','64x32','32x64','64x64','128x64','32x128','64x128']:
            plans.append((group+'-'+tile,{'GEMX_VITPOSE_RECT_GROUP':group,'GEMX_VITPOSE_RECT':tile}))
    for group in ['up','qkv','down']:
        for tile,bk in [('32x32','32'),('32x32','64'),('64x32','32'),('128x32','32'),('64x64','32')]:
            plans.append((group+'-'+tile+'-bk'+bk,{'GEMX_VITPOSE_RECT_GROUP':group,
                          'GEMX_VITPOSE_RECT':tile,'GEMX_VITPOSE_RECT_BK':bk}))
    plans.append(('baseline-repeat',{}))
    if a.cases:
        if set(a.cases)-{n for n,_ in plans}:p.error('unknown case')
        plans=[(n,e) for n,e in plans if n in a.cases or n=='baseline']
    report={'cpu_affinity':sorted(os.sched_getaffinity(0)),'iterations':a.iterations,'warmups':10,
            'base_env':{k:v for k,v in env.items() if k.startswith(('GGML_VK_','GEMX_'))},'runs':{}}
    for name,extra in plans:
        folder=a.output/name;folder.mkdir()
        local=dict(env,**extra,GEMX_BENCHMARK_PROFILE=str(folder/'iterations.csv'),GEMX_BENCHMARK_OUTPUT=str(folder/'heatmaps.f32'))
        cmd=[str(root/'build/vulkan/gemx-vitpose-benchmark'),str(root/'generated/reference/vitpose-f32.gguf'),
             str(root/'build/vulkan/bin/libggml-vulkan.so'),'Vulkan','8','2',str(a.iterations)]
        with (folder/'log.txt').open('w') as f:
            subprocess.run(cmd,cwd=root,env=local,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
        times=[float(r['wall_ms']) for r in csv.DictReader((folder/'iterations.csv').open())]
        result={'env':extra,'mean_ms':statistics.mean(times),'median_ms':statistics.median(times),
                'heatmaps_sha256':hashlib.sha256((folder/'heatmaps.f32').read_bytes()).hexdigest()}
        report['runs'][name]=result
        result['byte_identical']=result['heatmaps_sha256']==report['runs']['baseline']['heatmaps_sha256']
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,json.dumps(result),flush=True)


if __name__=='__main__':main()
