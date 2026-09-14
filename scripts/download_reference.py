#!/usr/bin/env python3
"""Download only pinned GEM-X denoiser reference assets with SHA-256 checks."""
import argparse,hashlib,json,os,tempfile,urllib.request
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('generated/reference'))
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'reference/sources.json').read_text());revision=manifest['models']['revision']
    base=f"https://huggingface.co/nvidia/GEM-X/resolve/{revision}/"
    for name,entry in manifest['models'].items():
        if not isinstance(entry,dict) or not name.startswith('onnx/'):continue
        dst=a.output/name
        if dst.exists():
            if dst.stat().st_size!=entry['bytes'] or digest(dst)!=entry['sha256']:raise ValueError(f'existing asset identity mismatch: {dst}')
            continue
        dst.parent.mkdir(parents=True,exist_ok=True);temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=dst.parent,prefix='.download-',delete=False) as f:
                temporary=Path(f.name)
                with urllib.request.urlopen(base+name,timeout=60) as response:
                    while block:=response.read(1024*1024):f.write(block)
                f.flush();os.fsync(f.fileno())
            if temporary.stat().st_size!=entry['bytes'] or digest(temporary)!=entry['sha256']:raise ValueError(f'download identity mismatch: {name}')
            os.link(temporary,dst)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
    print(a.output)
if __name__=='__main__':main()

