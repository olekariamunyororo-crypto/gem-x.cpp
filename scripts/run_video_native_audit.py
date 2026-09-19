#!/usr/bin/env python3
"""Run an independent native offline video audit from packed RGB frames.

Uses the same float32 boxes and GEMSEQ02 manifest as the Go demo. All child
processes inherit an eight-core affinity limit. Models remain resident across
Body frames; detector, Body and GEM/ViTPose stages run sequentially.
"""
from pathlib import Path
import argparse, json, os, struct, subprocess, time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--frames',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--gem-root',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--sam-root',type=Path)
p.add_argument('--bf16',action='store_true')
p.add_argument('--fast',action='store_true',help='Use approximate Vulkan arithmetic instead of the strict parity baseline')
p.add_argument('--fps',type=float,default=10)
a=p.parse_args()
allowed=sorted(os.sched_getaffinity(0))[:8]
os.sched_setaffinity(0,allowed)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='8'
for k in list(os.environ):
    if k.startswith(('GGML_VK_','SAM3D_')):os.environ.pop(k)
if not a.fast:
    for k in ('GGML_VK_DISABLE_F16','GGML_VK_DISABLE_COOPMAT','GGML_VK_DISABLE_COOPMAT2'):os.environ[k]='1'
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]


gem=a.gem_root.resolve()
sam=(a.sam_root or gem.parent/'sam3d.cpp').resolve()
source=a.frames.resolve()
out=a.output.resolve()
if out.exists() and any(out.iterdir()):raise ValueError('Use an empty audit output directory')
out.mkdir(parents=True,exist_ok=True)
images = out / 'frames'; images.mkdir(exist_ok=True)
for p in sorted(source.glob('*.input')):
    (images / p.name).write_bytes(p.read_bytes())
def manifest(path, magic, rows):
    path.write_bytes(magic + struct.pack('<I',len(rows)) + b''.join(struct.pack('<I',len(str(p).encode()))+str(p).encode() for row in rows for p in row))
paths = sorted(images.glob('*.input'))
if not 1<=len(paths)<=120:raise ValueError('Expected 1..120 packed frames')
manifest(out/'images.manifest', b'GEMIMGS1', [(p,) for p in paths])
pipeline = gem/'build/vulkan/gemx-pipeline'
backend = gem/'build/vulkan/bin/libggml-vulkan.so'
start=time.monotonic()
subprocess.run([pipeline,'--detect',gem/'generated/reference/yolox-f32.gguf',backend,'Vulkan','0','NVIDIA GeForce RTX 5070 Ti','8',out/'images.manifest',out/'boxes.bin'],check=True)
boxes=struct.unpack('<'+str(len(paths)*4)+'f',(out/'boxes.bin').read_bytes()[12:])
xys=[]
for i,p in enumerate(paths):
    x0,y0,x1,y1=boxes[i*4:i*4+4]; s=f32(max(f32(y1-y0),f32(f32(x1-x0)/.75))*f32(1.2)); cx=f32(f32(x0+x1)*.5);cy=f32(f32(y0+y1)*.5)
    xys.extend((cx,cy,s))
    data=bytearray(p.read_bytes());struct.pack_into('<4f',data,20,cx-s/2,cy-s/2,cx+s/2,cy+s/2);p.write_bytes(data)
times={'detector':time.monotonic()-start}
for mode in ['bf16' if a.bf16 else 'f32']:
    target=out/mode;target.mkdir(exist_ok=True)
    body=target/'body';body.mkdir(exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_VK_','SAM3D_'))}
    env['GGML_VK_DISABLE_F16']='1'
    if mode=='bf16':
        for k in ['SAM3D_BF16_COOPMAT2','SAM3D_BF16_FLASH_ATTENTION','SAM3D_BF16_PRECISE_PREFIX','GGML_VK_FUSE_BF16_ROUND','GGML_VK_FUSE_BF16_BINARY','GGML_VK_BF16_BINARY_LINEAR','SAM3D_BATCHED_TRANSFERS','SAM3D_SIMD_SKINNING','GGML_VK_F32_NARROW_MATMUL','GGML_VK_FUSE_BF16_SILU_GATE','GGML_VK_FUSE_BF16_AFFINE','GGML_VK_FUSE_BF16_NORM_AFFINE','SAM3D_IMAGE_GATHER']:env[k]='1'
        env.update(GGML_VK_BF16_MATMUL_TILE='small',GGML_VK_F32_NARROW_TILE='tiny32')
    else:env.update(GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
    build=sam/'build/vulkan-bf16-performance/bin'
    args=[build/'sam3d-body-infer','--worker',build/'libggml-vulkan.so','Vulkan','0','NVIDIA GeForce RTX 5070 Ti',sam/'generated/models/sam-3d-body-dinov3/body-dinov3-f32.gguf',sam/'generated/models/sam-3d-body-dinov3/body-pose-branch-f32.gguf',sam/'generated/models/mhr-public/mhr-lod1-f32.gguf','8','--gem-features']
    if mode=='bf16':args.append('--bf16')
    start=time.monotonic()
    with (target/'body.log').open('w') as log:
        proc=subprocess.Popen(args,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
        try:
            if proc.stdout.readline().strip()!=b'READY':raise RuntimeError('Body worker did not become ready; see body.log')
            for i,p in enumerate(paths):
                dest=body/f'{i:06d}.bin'
                packet=b''.join(struct.pack('<I',len(str(x).encode()))+str(x).encode() for x in [p,dest])
                proc.stdin.write(packet);proc.stdin.flush()
                while True:
                    line=proc.stdout.readline().strip()
                    if line==b'DONE':break
                    if not line.startswith(b'TIMING '):raise RuntimeError(line)
                if i%12==0:print(mode,i,flush=True)
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Body worker failed; see body.log')
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
    times[mode+'_body_with_load']=time.monotonic()-start
    manifest(target/'sequence.manifest',b'GEMSEQ02',[(p,body/f'{i:06d}.bin') for i,p in enumerate(paths)])
    with (target/'sequence.manifest').open('ab') as stream:stream.write(struct.pack('<'+str(len(xys))+'f',*xys))
    subprocess.run([pipeline,'--offline',gem/'generated/reference/gem-x-contact-f32.gguf',gem/'generated/reference/vitpose-f32.gguf',backend,'Vulkan','0','NVIDIA GeForce RTX 5070 Ti','8',target/'sequence.manifest',target/'unrefined-output',str(a.fps)],check=True)
    subprocess.run([pipeline,'--offline-contact',gem/'generated/reference/gem-x-contact-f32.gguf',gem/'generated/reference/vitpose-f32.gguf',backend,'Vulkan','0','NVIDIA GeForce RTX 5070 Ti','8',target/'sequence.manifest',target/'output',str(a.fps)],check=True)
(out/'timings.json').write_text(json.dumps(times,indent=2)+'\n')
