#!/usr/bin/env python3
"""Summarize isolated Vulkan QKV traces alongside the first CUDA QKV kernel.

CUDA CSV is the raw wide Nsight Compute export with a units row. Vulkan traces
must come from capture_vitpose_nsight.py --optimized --qkv-only. Samples are
warp-state shares, not fractions of wall time or guaranteed recoverable time.
"""
import argparse
import csv
import json
from pathlib import Path
from summarize_vitpose_nsight import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda-csv',type=Path,required=True)
    p.add_argument('--vulkan',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.cuda_csv.open() as f:
        reader=csv.DictReader(f);units=next(reader);qkv=next(reader)
    if 'cutlass_80_simt_sgemm_128x64_8x5_nn_align1' not in qkv['Kernel Name']:
        raise ValueError('unexpected first CUDA kernel; verify QKV launch mapping')
    prefix='smsp__pcsamp_warps_issue_stalled_'
    samples={k[len(prefix):]:float(v) for k,v in qkv.items()
             if k.startswith(prefix) and not k.endswith('_not_issued')}
    total=sum(samples.values())
    result={'scope':'First-block QKV: CUDA original-graph kernel replay; Vulkan resident-operand isolated replay. Different profilers and cache histories.',
            'normalization':'All sampled active warp states, including selected and not-selected; not wall-time shares.',
            'cuda':{'kernel':qkv['Kernel Name'],'block':qkv['Block Size'],'grid':qkv['Grid Size'],
                    'sample_count':total,'warp_state_percent':{k:100*v/total for k,v in samples.items()},
                    'metrics':{k:{'value':v,'unit':units[k]} for k,v in qkv.items()
                               if k.startswith(('launch__','sm__','smsp__','lts__','l1tex__','gpu__dram','memory_l1'))}},
            'vulkan':{str(d):summarize(d) for d in a.vulkan}}
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
