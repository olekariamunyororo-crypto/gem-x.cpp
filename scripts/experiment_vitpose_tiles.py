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

# BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP.
# Reuse the existing scalar shader with alternate specialization constants.
DOWN_TILES = {
    '128x64': [128,128,64,16,64,32,2,4,2,1,32],
    '64x128': [128,64,128,16,32,64,2,4,4,1,32],
    '64x32': [64,64,32,16,32,32,2,4,2,1,32],
    '128x32': [128,128,32,16,32,32,2,4,2,1,32],
    '32x64': [64,32,64,16,32,32,2,4,2,1,32],
    '64x64-64threads': [64,64,64,16,32,64,2,4,2,1,32],
}


def custom_down_source(source):
    anchor='    vk_matmul_pipeline pipeline_matmul_f32 {};'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'\n    vk_pipeline gemx_down_pipeline;')
    anchor='    // mul mat vec\n'
    assert source.count(anchor)==1
    code='''
    if (!device->fp16 && !device->coopmat2 && !device->coopmat_support) {
        const char *tile = getenv("GEMX_TILE_DOWN_CUSTOM");
        std::vector<uint32_t> spec;
'''
    for name, values in DOWN_TILES.items():
        block,bm,bn,_,wm,wn,wmi,tm,tn,_,warp=values
        assert block//warp==(bm//wm)*(bn//wn)
        assert wm*wn%(warp*tm*tn*wmi)==0
        wni=wm*wn//(warp*tm*tn*wmi)
        assert wm%wmi==0 and wn%wni==0 and wm//wmi%tm==0 and wn//wni%tn==0
        code+='        if (tile && strcmp(tile, "'+name+'") == 0) spec = {'+','.join(map(str,values))+'};\n'
    code+='''
        if (!spec.empty()) {
            ggml_vk_create_pipeline(device, device->gemx_down_pipeline, "gemx_down_custom",
                matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants),
                {spec[1], spec[2], 1}, ggml_vk_mul_mm_spec(spec, true), 32);
        }
    }
'''
    source=source.replace(anchor,code+anchor)
    anchor='    const vk_pipeline original_pipeline = pipeline;'
    assert source.count(anchor)==1
    return source.replace(anchor,anchor+'''
    if (ctx->device->gemx_down_pipeline && aligned &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
        ne01 == 1280 && ne11 == 197 && ne10 == 5120 && ne12 * ne13 == 2) {
        pipeline = ctx->device->gemx_down_pipeline;
    }
''')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=60)
    p.add_argument('--skip-build', action='store_true')
    p.add_argument('--preserve-split-k', action='store_true',
                   help='Choose reduction partitions using the original pipeline, independently of output tiles')
    p.add_argument('--custom-down-tiles', action='store_true', help='Test rectangular down tiles with original split-K')
    p.add_argument('--cases', nargs='+', help='Select named cases from the sweep')
    a = p.parse_args()
    if a.custom_down_tiles:a.preserve_split_k=True
    if len(os.sched_getaffinity(0))>8 or a.iterations<1:
        p.error('use at most eight CPU cores and a positive iteration count')
    root=Path(__file__).resolve().parents[1]; build=root/'build/vulkan'
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    module=out/'libggml-vulkan.so'
    manifest=out/'build-options.json'
    options={'preserve_split_k':a.preserve_split_k, 'custom_down_tiles':a.custom_down_tiles,
             'custom_down_shader':'matmul_f32_f32_fp32' if a.custom_down_tiles else None,
             'custom_down_specs':DOWN_TILES if a.custom_down_tiles else None}
    if a.skip_build and (not manifest.exists() or json.loads(manifest.read_text())!=options):
        p.error('--skip-build requires matching recorded build options')
    if not a.skip_build:
        source=(root/'ggml/src/ggml-vulkan/ggml-vulkan.cpp').read_text()
        anchor='    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, ne11, aligned, qx_needs_dequant ? f16_type : src0->type, effective_src1_type);'
        assert source.count(anchor)==1
        saved_pipeline='\n    const vk_pipeline original_pipeline = pipeline;\n' if a.preserve_split_k else '\n'
        source=source.replace(anchor,anchor+saved_pipeline+HOOK)
        if a.preserve_split_k:
            split_anchor='const uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);'
            assert source.count(split_anchor)==1
            source=source.replace(split_anchor,split_anchor.replace('disable_split_k, pipeline','disable_split_k, original_pipeline'))
        if a.custom_down_tiles:source=custom_down_source(source)
        (out/'ggml-vulkan.cpp').write_text(source)
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
        manifest.write_text(json.dumps(options,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_VK_','GEMX_TILE_','GEMX_BENCHMARK_'))}
    env.pop('LD_PRELOAD',None)
    env.update(GEMX_VITPOSE_FLATTEN='0',GEMX_VITPOSE_NORM='0',GEMX_VITPOSE_SWIGLU='0',GEMX_VITPOSE_RECT='0',GEMX_VITPOSE_QKV_LAYOUT='0',GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1',
               GEMX_BENCHMARK_WARMUP='10',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    plans=[('baseline',{})]
    plans += [(name.lower()+'-'+tile,{'GEMX_TILE_'+name:tile})
              for name in ('UP','DOWN','QKV','PROJ') for tile in ('m','s')]
    plans += [('all-m',{'GEMX_TILE_'+name:'m' for name in ('UP','DOWN','QKV','PROJ')}),
              ('up-proj-s',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s'}),
              ('up-proj-s-down-m',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s','GEMX_TILE_DOWN':'m'}),
              ('up-proj-s-down-s',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s','GEMX_TILE_DOWN':'s'}),
              ('up-proj-s-repeat',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s'}),
              ('up-proj-s-qkv-m',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s','GEMX_TILE_QKV':'m'}),
              ('up-proj-s-qkv-s',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s','GEMX_TILE_QKV':'s'}),
              ('baseline-repeat',{})]
    if a.custom_down_tiles:
        plans=[('baseline',{}),('up-proj-s',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s'})]
        plans += [('down-'+name,{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s','GEMX_TILE_DOWN_CUSTOM':name})
                  for name in DOWN_TILES]
        plans += [('up-proj-s-repeat',{'GEMX_TILE_UP':'s','GEMX_TILE_PROJ':'s'})]
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
