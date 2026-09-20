#!/usr/bin/env python3
"""Compare saved crop heatmaps and native flip/quarter-pixel decoding.

Inputs are directories from check_vitpose_crops.py --save-heatmaps. Coordinate
errors are heatmap pixels, before applying the common crop-to-image transform.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def decode(h):
    flip=np.arange(77)
    pairs=[(9,10),*[(i,i+28) for i in range(11,39)],(67,72),(68,73),(69,74),(70,75),(71,76)]
    for a,b in pairs:flip[a],flip[b]=b,a
    merged=np.float32(.5)*(h[::2]+h[1::2,flip,:,::-1])
    flat=merged.reshape(-1,64*48);best=flat.argmax(-1)
    ix=best%48;iy=best//48;r=np.arange(len(best))
    xy=np.stack([ix,iy],axis=-1).astype(np.float32)
    for axis,step,valid in [(0,1,(ix>1)&(ix<47)),(1,48,(iy>1)&(iy<63))]:
        rr=r[valid];bb=best[valid]
        xy[valid,axis]+=np.float32(.25)*np.sign(flat[rr,bb+step]-flat[rr,bb-step])
    return xy,flat[r,best]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();names=sorted(x.name for x in a.baseline.glob('*.npy'))
    if not names or names!=sorted(x.name for x in a.candidate.glob('*.npy')):raise ValueError('mismatched or empty files')
    report={}
    for name in names:
        x=np.load(a.baseline/name);y=np.load(a.candidate/name)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError(name)
        key=name.split('-')[0]
        r=report.setdefault(key,dict(files=0,elements=0,absolute_error_sum=0.,max_abs_error=0.,raw_peaks=0,changed_raw_peaks=0,decoded_joints=0,changed_decoded_xy=0,max_xy_heatmap_pixels=0.,max_confidence_error=0.))
        d=np.abs(x-y);r['files']+=1;r['elements']+=d.size;r['absolute_error_sum']+=float(d.sum(dtype=np.float64));r['max_abs_error']=max(r['max_abs_error'],float(d.max()))
        px=x.reshape(-1,64*48).argmax(-1);py=y.reshape(-1,64*48).argmax(-1)
        r['raw_peaks']+=len(px);r['changed_raw_peaks']+=int(np.count_nonzero(px!=py))
        xx,xc=decode(x);yx,yc=decode(y)
        r['decoded_joints']+=len(xx);r['changed_decoded_xy']+=int(np.count_nonzero(np.any(xx!=yx,axis=-1)))
        r['max_xy_heatmap_pixels']=max(r['max_xy_heatmap_pixels'],float(np.abs(xx-yx).max()))
        r['max_confidence_error']=max(r['max_confidence_error'],float(np.abs(xc-yc).max()))
    for r in report.values():r['mean_abs_error']=r.pop('absolute_error_sum')/r['elements']
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
