#!/usr/bin/env python3
"""Build an isolated ViTPose tile-selection experiment using existing shader objects.

No production backend or GGML source is overwritten. Run under taskset -c 0-7.
Requires a completed Release Vulkan build and its compile_commands.json.
"""
import argparse
import array
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess

HOOK = r'''
    // Experiment only: select existing F32 kernels for ViTPose's four projections.
    if (!ctx->device->coopmat2 && !ctx->device->coopmat_support &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && ne11 == 197) {
        const char *key = ne01 == 5120 && ne10 == 1280 ? "GEMX_TILE_UP" :
                          ne01 == 1280 && ne10 == 5120 ? "GEMX_TILE_DOWN" :
                          ne01 == 3840 && ne10 == 1280 ? "GEMX_TILE_QKV" :
                          ne01 == 1280 && ne10 == 1280 ? "GEMX_TILE_PROJ" : nullptr;
        const char *tile = key ? getenv(key) : nullptr;
        if (tile && tile[0] == 'm') pipeline = aligned ? mmp->a_m : mmp->m;
        if (tile && tile[0] == 's') pipeline = aligned ? mmp->a_s : mmp->s;
    }
'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=60)
    p.add_argument('--skip-build', action='store_true')
    p.add_argument('--cases', nargs='+', help='Select named cases from the sweep')
    a = p.parse_args()
    if len(os.sched_getaffinity(0))>8 or a.iterations<1:
        p.error('use at most eight CPU cores and a positive iteration count')
    root=Path(__file__).resolve().parents[1]; build=root/'build/vulkan'
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    module=out/'libggml-vulkan.so'
    if not a.skip_build:
        source=(root/'ggml/src/ggml-vulkan/ggml-vulkan.cpp').read_text()
        anchor='    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, ne11, aligned, qx_needs_dequant ? f16_type : src0->type, effective_src1_type);'
        assert source.count(anchor)==1
        (out/'ggml-vulkan.cpp').write_text(source.replace(anchor,anchor+'\n'+HOOK))
        entry=next(e for e in json.loads((build/'compile_commands.json').read_text())
                   if e['file'].endswith('/ggml-vulkan.cpp'))
        cmd=shlex.split(entry['command']);obj=out/'ggml-vulkan.cpp.o'
        cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-c')+1]=str(out/'ggml-vulkan.cpp')
        cmd+=['-I'+str(root/'ggml/src/ggml-vulkan')]
        cache=(build/'CMakeCache.txt').read_text().splitlines()
        for line in cache:
            if line.startswith('Vulkan_INCLUDE_DIR:'):
                cmd+=['-I'+line.split('=',1)[1]]
            if line.startswith('SPIRV-Headers_DIR:'):
                cmd+=['-I'+str(Path(line.split('=',1)[1]).parents[2]/'include')]
        subprocess.run(cmd,cwd=build,check=True)
        # Reuse CMake's linker invocation and immutable generated shader objects.
        ninja=next(line.split('=',1)[1] for line in cache if line.startswith('CMAKE_MAKE_PROGRAM:'))
        line=subprocess.check_output([ninja,'-t','commands','bin/libggml-vulkan.so'],cwd=build,text=True).splitlines()[-1]
        assert line.startswith(': && ') and line.endswith(' && :')
        cmd=shlex.split(line[5:-5]);cmd[cmd.index('-o')+1]=str(module)
        cmd=[str(obj) if v.endswith('/ggml-vulkan.cpp.o') else v for v in cmd
             if not v.startswith('-Wl,--dependency-file=')]
        subprocess.run(cmd,cwd=build,check=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_VK_','GEMX_TILE_','GEMX_BENCHMARK_'))}
    env.pop('LD_PRELOAD',None)
    env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1',
               GEMX_BENCHMARK_WARMUP='10',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    plans=[('baseline',{})]
    plans += [(name.lower()+'-'+tile,{'GEMX_TILE_'+name:tile})
              for name in ('UP','DOWN','QKV','PROJ') for tile in ('m','s')]
    plans += [('all-m',{'GEMX_TILE_'+name:'m' for name in ('UP','DOWN','QKV','PROJ')}),
              ('up-proj-s',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s'}),
              ('up-proj-s-qkv-m',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s','GEMX_TILE_QKV':'m'}),
              ('up-proj-s-qkv-s',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s','GEMX_TILE_QKV':'s'}),
              ('baseline-repeat',{})]
    if a.cases:
        assert set(a.cases)<=set(name for name,_ in plans), 'unknown case'
        plans=[(name,env) for name,env in plans if name in a.cases or name=='baseline']
    report={};reference=None
    for name,override in plans:
        folder=out/name;folder.mkdir(exist_ok=False)
        local=dict(env,**override,GEMX_BENCHMARK_OUTPUT=str(folder/'heatmaps.f32'),
                   GEMX_BENCHMARK_PROFILE=str(folder/'iterations.csv'))
        cmd=[str(build/'gemx-vitpose-benchmark'),str(root/'generated/reference/vitpose-f32.gguf'),
             str(module),'Vulkan','8','2',str(a.iterations)]
        with (folder/'stdout.txt').open('w') as stdout,(folder/'stderr.txt').open('w') as stderr:
            subprocess.run(cmd,env=local,stdout=stdout,stderr=stderr,check=True,timeout=120)
        data=(folder/'heatmaps.f32').read_bytes()
        if reference is None:reference=data
        values=array.array('f');values.frombytes(data)
        refs=array.array('f');refs.frombytes(reference)
        errors=[abs(x-y) for x,y in zip(values,refs)]
        rows=list(csv.DictReader((folder/'iterations.csv').open()))
        times=sorted(float(row['wall_ms']) for row in rows)
        report[name]=dict(mean_ms=statistics.mean(times),median_ms=statistics.median(times),
                          p95_ms=times[min(len(times)-1,int(len(times)*.95))],
                          exact=data==reference,max_abs=max(errors),mean_abs=statistics.mean(errors),
                          sha256=hashlib.sha256(data).hexdigest(),env=override)
        print(name,json.dumps(report[name]),flush=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
