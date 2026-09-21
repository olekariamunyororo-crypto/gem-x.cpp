#!/usr/bin/env python3
"""Create small valid non-GGUF seeds; no models or external assets required."""
import argparse
from pathlib import Path
import struct
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
inputs=a.output/'inputs';api=a.output/'api'
inputs.mkdir(parents=True,exist_ok=True);api.mkdir(parents=True,exist_ok=True)
u32=lambda x:struct.pack('<I',x)
path=u32(5)+b'a.bin'
def seed(name,kind,data): (inputs/name).write_bytes(bytes([kind])+data)
seed('image',0,b'S3DIMG01'+struct.pack('<III8f',8,8,24,0,0,8,8,8,8,4,4)+bytes(8*8*3))
seed('token',1,b'S3DOUT01'+u32(1)+u32(10)+b'pose_token'+struct.pack('<IIQQQ',1,2,1024,1,1024)+bytes(4096))
for version,size in [(1,4332),(2,5256)]:seed('pose'+str(version),2,f'GEMPOSE{version}'.encode()+bytes(size-8))
seed('images',3,b'GEMIMGS1'+u32(1)+path)
seed('sequence1',4,b'GEMSEQ01'+u32(1)+path+path)
seed('sequence2',4,b'GEMSEQ02'+u32(1)+path+path+struct.pack('<3f',4,4,8))
seed('export',5,b'GEMMAN01'+u32(1)+path)
seed('integer',6,b'8');seed('float',7,b'0.5');seed('empty-float',7,b'')
values=[0.0]*(231+3+9+6+1024)
values[231:234]=[4,4,8];values[234:243]=[8,0,4,0,8,4,0,0,1];values[243:249]=[1,0,0,0,1,0]
values[0:3]=[4,4,8]
for mode in (0,4): (api/f'crop-{mode}').write_bytes(bytes([mode,8,8,24])+struct.pack('<'+'f'*len(values),*values))
