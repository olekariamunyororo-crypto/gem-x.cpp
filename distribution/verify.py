#!/usr/bin/env python3
"""Offline verification of the tested GGUF bundle; never uploads or downloads."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--models',type=Path,required=True)
a=p.parse_args()
manifest=json.loads((Path(__file__).parent/'manifest.json').read_text())
for name,expected in manifest.items():
    path=a.models/name
    if path.stat().st_size!=expected['bytes']:raise SystemExit(f'{name}: size mismatch')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected['sha256']:raise SystemExit(f'{name}: SHA-256 mismatch')
    print(f'{name}: verified')
