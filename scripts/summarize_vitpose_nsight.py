#!/usr/bin/env python3
"""Summarize Nsight Graphics auto-export TSVs; warp shares are not time shares."""
import argparse
import csv
import json
from pathlib import Path


def summarize(directory):
    export = directory / 'BASE_UNLOCKED'
    with (export / 'GPUTRACE_FRAME.xls').open() as stream:
        metrics = {key: float(value) for key, value in csv.reader(stream, delimiter='\t')}
    prefix = 'GPUTrace.PCSampler.tpc__warps_issue_stalled_'
    suffix = '.avg.per_cycle_elapsed'
    states = {key[len(prefix):-len(suffix)]: value for key, value in metrics.items()
              if key.startswith(prefix) and key.endswith(suffix)}
    total = sum(states.values())
    active = metrics['GPUTrace.PCSampler.tpc__warps_active_shader_compute.avg.per_cycle_elapsed']
    if not total or abs(total - active) > 0.001 * active:
        raise ValueError(f'{directory}: incomplete warp-state accounting')
    with (export / 'REPRO_INFO.xls').open() as stream:
        metadata = dict(csv.reader(stream, delimiter='\t'))
    return {'trace_directory': str(directory), 'metadata': metadata,
            'metrics': metrics, 'warp_state_total': total,
            'warp_state_percent': {key: 100 * value / total for key, value in states.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {
        'scope': 'Complete repeated strict-F32 batch-2 ViTPose graph, not individual matmuls',
        'warp_state_normalization': 'Percent of all sampled active warp states, including selected and not_selected; not wall time or recoverable time',
        'traces': {directory.name: summarize(directory) for directory in args.traces},
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
