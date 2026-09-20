#!/usr/bin/env python3
"""Strict-F32 ViTPose profiling; run under taskset -c 0-7 on Linux.

Writes raw GPU operator logs, host API spans, per-inference timings, NVIDIA
telemetry and output hashes. Batch two represents a live crop plus its flip.
No camera, network service or production model code is modified.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time


def stats(values):
    values = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p95=values[min(len(values)-1, int(len(values)*.95))],
                minimum=values[0], maximum=values[-1]) if values else None


def operators(path, warmups):
    blocks = path.read_text().split('Vulkan Timings:\n')[1:][warmups:]
    groups = {}
    totals = []
    for block in blocks:
        match = re.search(r'Total time: ([\d.]+) us', block)
        if not match:
            continue
        totals.append(float(match[1])/1000)
        for line in block.splitlines():
            m = re.match(r'(.+): (\d+) x ([\d.]+) us = ([\d.]+) us(?: \(([\d.]+) GFLOPS/s\))?', line)
            if m:
                name, count, average, total, gflops = m.groups()
                entry = groups.setdefault(name, dict(count=int(count), total_ms=[], gflops=[]))
                entry['total_ms'].append(float(total)/1000)
                if gflops:
                    entry['gflops'].append(float(gflops))
    return dict(graph_ms=stats(totals), samples=len(totals), operations={
        name: dict(count=g['count'], total_ms=stats(g['total_ms']), gflops=stats(g['gflops']))
        for name, g in groups.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', type=Path, help='Isolated Vulkan backend for kernel experiments')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--input',type=Path,help='Raw normalized NCHW F32 input; use batch-two runs only')
    parser.add_argument('--runs', nargs='+', choices=['baseline', 'host', 'operators', 'concurrent',
                                                     'batch1', 'batch4', 'batch8', 'baseline-repeat', 'sync'])
    a = parser.parse_args()
    if a.iterations < 30:
        parser.error('at least 30 iterations required')
    if len(os.sched_getaffinity(0)) > 8:
        parser.error('restrict CPU affinity to at most eight cores')
    root, output = a.root.resolve(), a.output.resolve()
    module=(a.module or root/'build/vulkan/bin/libggml-vulkan.so').resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_VK_', 'GEMX_BENCHMARK_', 'GEMX_GGML_'))}
    env.pop('LD_PRELOAD', None)
    env.update(GGML_VK_DISABLE_F16='1', GGML_VK_DISABLE_COOPMAT='1', GGML_VK_DISABLE_COOPMAT2='1',
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', GEMX_BENCHMARK_WARMUP='10')
    if a.input:
        if not a.runs or any(n in a.runs for n in ('batch1','batch4','batch8')):
            parser.error('--input requires explicit batch-two --runs')
        env['GEMX_BENCHMARK_INPUT']=str(a.input.resolve())
    fields = ['utilization.gpu', 'utilization.memory', 'power.draw', 'clocks.sm',
              'clocks.mem', 'temperature.gpu', 'memory.used', 'pstate',
              'clocks_event_reasons.sw_power_cap', 'clocks_event_reasons.hw_thermal_slowdown']
    report = dict(cpu_affinity=sorted(os.sched_getaffinity(0)), warmups=10, module=str(module),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  ggml_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root/'ggml', text=True).strip(),
                  input='deterministic normalized synthetic RGB; batch 2 = crop + flip workload shape',
                  device=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,power.limit',
                      '--format=csv'], text=True).strip(), runs={})
    if a.input:
        report['input']=str(a.input.resolve())
        report['input_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    plans = [('baseline', 2, a.iterations, {}),
             ('host', 2, a.iterations, {'LD_PRELOAD': str(root/'build/vulkan/libgemx-profile-ggml.so')}),
             ('operators', 2, 30, {'GGML_VK_PERF_LOGGER': '1'}),
             ('concurrent', 2, 30, {'GGML_VK_PERF_LOGGER': '1', 'GGML_VK_PERF_LOGGER_CONCURRENT': '1'}),
             ('batch1', 1, a.iterations, {}), ('batch4', 4, a.iterations//2, {}),
             ('batch8', 8, a.iterations//4, {}),
             ('baseline-repeat', 2, a.iterations, {}),
             ('sync', 2, 1, {'GGML_VK_SYNC_LOGGER': '1', 'GEMX_BENCHMARK_WARMUP': '1'})]
    for name, batch, count, extra in plans:
        if a.runs and name not in a.runs:
            continue
        folder = output/name
        folder.mkdir()
        local_env = dict(env, **extra, GEMX_BENCHMARK_PROFILE=str(folder/'iterations.csv'),
                         GEMX_BENCHMARK_OUTPUT=str(folder/'heatmaps.f32'))
        if name == 'host':
            local_env['GEMX_GGML_TRACE'] = str(folder/'host.csv')
        telemetry = []
        monitor = subprocess.Popen(['nvidia-smi', '--id=0', '--query-gpu='+','.join(fields),
                                    '--format=csv,noheader,nounits', '-lms', '100'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def collect():
            for line in monitor.stdout:
                telemetry.append([time.monotonic(), *next(csv.reader([line]))])
        thread = threading.Thread(target=collect)
        thread.start()
        try:
            command = [str(root/'build/vulkan/gemx-vitpose-benchmark'),
                       str(root/'generated/reference/vitpose-f32.gguf'),
                       str(module), 'Vulkan', '8', str(batch), str(count)]
            with (folder/'stdout.txt').open('w') as stdout, (folder/'stderr.txt').open('w') as stderr:
                subprocess.run(command, env=local_env, stdout=stdout, stderr=stderr, check=True, timeout=300)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            thread.join(timeout=10)
        if not telemetry:
            raise RuntimeError('NVIDIA telemetry unavailable: '+monitor.stderr.read())
        with (folder/'telemetry.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['monotonic_seconds', *fields]);writer.writerows(telemetry)
        rows = list(csv.DictReader((folder/'iterations.csv').open()))
        start, end = float(rows[0]['start_seconds']), float(rows[-1]['end_seconds'])
        # NVIDIA utilization uses a rolling sampling window: omit the first 2 s.
        samples = [r for r in telemetry if start+2 <= r[0] <= end]
        run = dict(batch=batch, iterations=count, wall_ms=stats([float(r['wall_ms']) for r in rows]),
                   process_cpu_ms=stats([float(r['cpu_ms']) for r in rows]), telemetry_samples=len(samples),
                   telemetry={}, heatmaps_sha256=hashlib.sha256((folder/'heatmaps.f32').read_bytes()).hexdigest(),
                   stdout=(folder/'stdout.txt').read_text().strip())
        for index, field in enumerate(fields, 1):
            vals = [r[index].strip() for r in samples]
            try:
                run['telemetry'][field] = stats([float(v) for v in vals])
            except ValueError:
                run['telemetry'][field] = sorted(set(vals))
        if 'GGML_VK_PERF_LOGGER' in extra:
            run['gpu'] = operators(folder/'stderr.txt', 10)
            assert run['gpu']['samples']==count, 'missing Vulkan timestamp records'
        if name == 'host':
            traces = [r for r in csv.DictReader((folder/'host.csv').open())
                      if float(r['start_seconds']) >= start and float(r['end_seconds']) <= end]
            run['host'] = {op: {key: stats([float(r[key]) for r in traces if r['operation']==op])
                               for key in ('wall_ms', 'cpu_ms', 'bytes')}
                           for op in sorted({r['operation'] for r in traces})}
            run['host_counts'] = {op: sum(r['operation']==op for r in traces) for op in run['host']}
            assert set(run['host']) == {'upload', 'download', 'record_submit', 'synchronize'}, 'incomplete host interception'
            assert all(n==count for n in run['host_counts'].values()), 'unexpected nested or missing host events'
            # Perfetto/Chrome trace viewer: these are HOST ranges, not GPU events.
            events = [dict(name=r['operation'], cat='host_api', ph='X', pid=1, tid=1,
                           ts=(float(r['start_seconds'])-start)*1e6, dur=float(r['wall_ms'])*1000,
                           args=dict(cpu_ms=float(r['cpu_ms']), bytes=int(r['bytes']))) for r in traces]
            events += [dict(name='inference', cat='host_total', ph='X', pid=1, tid=2,
                            ts=(float(r['start_seconds'])-start)*1e6, dur=float(r['wall_ms'])*1000,
                            args=dict(iteration=int(r['iteration']))) for r in rows]
            (folder/'host-trace.json').write_text(json.dumps(dict(traceEvents=events)))
        if name == 'sync':
            lines = (folder/'stderr.txt').read_text().splitlines()
            calls = int(local_env['GEMX_BENCHMARK_WARMUP'])+count
            run['dependency_barriers_per_inference'] = lines.count('sync')/calls
            run['logged_operations_per_inference'] = sum(bool(re.match(r'^\d+ ',line)) for line in lines)/calls
        report['runs'][name] = run
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, run['stdout'], flush=True)
    hashes = {run['heatmaps_sha256'] for run in report['runs'].values() if run['batch']==2}
    assert len(hashes)<=1, 'profiler changed output'


if __name__ == '__main__':
    main()
