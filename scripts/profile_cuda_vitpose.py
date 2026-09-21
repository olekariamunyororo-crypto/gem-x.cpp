#!/usr/bin/env python3
"""Reference-only CUDA profiling in the existing Docker environment; no backend integration.

Run with taskset -c 0-7. Counters require container SYS_ADMIN but do not change
host driver permissions. Profiler tools, dependencies and model artifacts are external.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['latency','timeline','kernels','counters'],required=True)
    p.add_argument('--ncu',type=Path,help='Standalone Nsight Compute directory; use 2026.3+ for counters')
    p.add_argument('--model',type=Path,default=root/'generated/reference/onnx/vitpose.onnx')
    p.add_argument('--weights',type=Path,default=root/'generated/reference/onnx/vitpose.onnx.data')
    p.add_argument('--dependencies',type=Path,default=root/'generated/reference/e2e/audit-20260918/python')
    p.add_argument('--image',default='gemx-reference:e2e')
    a=p.parse_args()
    if len(os.sched_getaffinity(0))>8:p.error('restrict affinity to eight cores')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    try:mountout='/work/'+str(out.relative_to(root))
    except ValueError:p.error('output must be within this repository')
    base=['docker','run','--rm','--network','none','--cpuset-cpus','0-7','--device','nvidia.com/gpu=all',
          '-e','PYTHONPATH=/deps','-e','OMP_NUM_THREADS=8','-e','OPENBLAS_NUM_THREADS=8',
          '-v',str(root)+':/work','-v',str(a.dependencies.resolve())+':/deps:ro',
          '-v',str(a.model.resolve())+':/models/vitpose.onnx:ro',
          '-v',str(a.weights.resolve())+':/models/vitpose.onnx.data:ro']
    args=['/work/scripts/benchmark_onnx_reference.py','--model','/models/vitpose.onnx','--kind','vitpose',
          '--input','/work/reference/vitpose_preprocess_reference.bin','--tf32','0','--warmups','20',
          '--output',mountout+'/'+a.mode+'.json']
    if a.mode in ['latency','timeline']:
        cmd=base+['--entrypoint','python',a.image]+args
        if a.mode=='latency':cmd+=['--iterations','200','--expected','/work/reference/vitpose_heatmap_reference.bin','--save-output',mountout+'/cuda-output.f32']
        else:cmd+=['--iterations','3','--cuda-trace',mountout+'/cuda-trace.json']
    else:
        base+=['--cap-add','SYS_ADMIN'];exe='ncu'
        if a.ncu:base+=['-v',str(a.ncu.resolve())+':/ncu:ro'];exe='/ncu/ncu'
        flags=['--clock-control','none','--cache-control','none','--export',mountout+'/cuda-'+a.mode]
        if a.mode=='counters':
            metrics=['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
                     'sm__warps_active.avg.pct_of_peak_sustained_elapsed','dram__throughput.avg.pct_of_peak_sustained_elapsed',
                     'lts__throughput.avg.pct_of_peak_sustained_elapsed','l1tex__throughput.avg.pct_of_peak_sustained_elapsed']
            flags+=['--replay-mode','app-range','--metrics',','.join(metrics)]
        else:
            flags+=['--profile-from-start','off','--kernel-name-base','demangled',
                    '--kernel-name','regex:.*(cutlass::Kernel2|magma_sgemmEx_kernel).*','--launch-count','7']
            for section in ['SpeedOfLight','Occupancy','SchedulerStats','WarpStateStats','LaunchStats','SourceCounters']:
                flags+=['--section',section]
        cmd=base+['--entrypoint',exe,a.image]+flags+['python']+args+['--iterations','1','--cuda-profile']
    (out/(a.mode+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    with (out/(a.mode+'.log')).open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
    if a.mode in ['kernels','counters'] and not any(out.glob('cuda-'+a.mode+'.ncu-rep*')):
        raise RuntimeError('profiler did not export a counter report; inspect the log')
    if a.mode=='timeline':
        events=json.loads((out/'cuda-trace.json').read_text())['traceEvents']
        if not any(e.get('cat')=='kernel' for e in events):raise RuntimeError('CUDA timeline contains no kernels')
    print(a.mode+' complete')


if __name__=='__main__':main()
