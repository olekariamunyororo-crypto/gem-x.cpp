#!/usr/bin/env python3
"""Sample both workloads with the same external, non-injecting GPU collector.

Requires profile_cuda_vitpose.py --mode latency in --output and a matching
raw input.f32 file there. Run with taskset -c 0-7. No clocks are changed.
"""
import argparse,json,os,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--nsys',type=pathlib.Path,required=True,help='Directory containing target-linux-x64/nsys')
a=p.parse_args()
if len(os.sched_getaffinity(0))>8:p.error('restrict affinity to eight cores')
out=a.output.resolve()
try:mountout='/work/'+str(out.relative_to(root))
except ValueError:p.error('output must be inside this repository')
for mode in ['vulkan','cuda']:
 ready=out/(mode+'-sampling-ready');log=out/(mode+'-sampling-workload.log')
 if ready.exists():raise RuntimeError('output already exists')
 if mode=='vulkan':
  env=dict(os.environ,GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1',GEMX_VITPOSE_QKV_LAYOUT='0',GEMX_VITPOSE_FLATTEN='1',GEMX_VITPOSE_NORM='1',GEMX_VITPOSE_SWIGLU='1',GEMX_VITPOSE_RECT_GROUP='up',GEMX_VITPOSE_RECT='64x64',GEMX_BENCHMARK_INPUT=str(out/'input.f32'),GEMX_BENCHMARK_PROFILE=str(ready),GEMX_BENCHMARK_WARMUP='20')
  cmd=[str(root/'build/vulkan/gemx-vitpose-benchmark'),str(root/'generated/reference/vitpose-f32.gguf'),str(root/'build/vulkan/bin/libggml-vulkan.so'),'Vulkan','8','2','600']
 else:
  env=os.environ.copy();cmd=json.loads((out/'latency-command.json').read_text());cmd[cmd.index('--iterations')+1]='750';cmd[cmd.index('--output')+1]=mountout+'/cuda-sampling-workload.json';cmd[cmd.index('--save-output')+1]=mountout+'/cuda-sampling-output.f32';cmd[2:2]=['--name','gemx-sampler-'+str(os.getpid())];cmd+=['--ready-file',mountout+'/'+ready.name]
 with log.open('w') as f:
  work=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  try:
   deadline=time.monotonic()+180
   while not ready.exists():
    if work.poll() is not None or time.monotonic()>deadline:raise RuntimeError('workload did not reach warm-up')
    time.sleep(.1)
   time.sleep(2)
   prof=['docker','run','--rm','--network','none','--cpuset-cpus','0-7','--cap-add','SYS_ADMIN','--device','nvidia.com/gpu=all','-v',str(a.nsys.resolve())+':/nsys:ro','-v',str(root)+':/work','--entrypoint','/nsys/target-linux-x64/nsys','gemx-reference:e2e','profile','--trace=none','--sample=none','--cpuctxsw=none','--gpu-metrics-devices=0','--gpu-metrics-set=gb20x-top','--gpu-metrics-frequency=10000','--duration=8','--export=sqlite','--output',mountout+'/'+mode+'-sampling','sleep','8']
   (out/(mode+'-sampling-command.json')).write_text(json.dumps(prof,indent=2))
   with (out/(mode+'-sampling.log')).open('w') as g:subprocess.run(prof,stdout=g,stderr=subprocess.STDOUT,check=True,timeout=60)
   if work.wait(timeout=60)!=0:raise RuntimeError('workload failed; inspect '+str(log))
  finally:
   if work.poll() is None:
    if mode=='cuda':subprocess.run(['docker','rm','-f','gemx-sampler-'+str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    work.terminate();work.wait(timeout=10)
 print(mode+' sampling complete',flush=True)
