#!/usr/bin/env python3
"""Summarize qa_live_browser.py A/B runs and live-worker API transfer tracing."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import statistics


def browser(path):
    report = json.loads((path/'browser.json').read_text())
    # Discard context warm-up; HTTP result i corresponds to encoded source i.
    rows = report['timings'][30:]
    encodes = report['encodes'][30:30+len(rows)]
    age = [r['end']-e['start'] for r,e in zip(rows,encodes)]
    result = dict(frames=len(rows), fps=1000*(len(rows)-1)/(rows[-1]['end']-rows[0]['end']),
        native_ms=statistics.mean(r['native'] for r in rows),
        encode_to_headers_ms=statistics.mean(age), encode_to_headers_p95_ms=sorted(age)[int(.95*len(age))],
        interframe_overhead_ms=statistics.mean(rows[i]['end']-rows[i-1]['end']-rows[i]['native'] for i in range(1,len(rows))),
        max_pending=report['maxPending'])
    if rows[0]['timing']:
        result['server_ms'] = {key:statistics.mean(float(part.split('=')[1]) for r in rows for part in r['timing'].split(', ') if part.startswith(key+';')) for key in ('prepare','queue','write','infer')}
    with sqlite3.connect(path/'gpu.sqlite') as db:
        metrics = dict(db.execute('select metricId,metricName from TARGET_INFO_GPU_METRICS'))
        wanted = {'Sync Compute in Flight [Throughput %]', 'SM Throughput [Throughput %]', 'SM FMA Pipe Throughput [Throughput %]', 'VRAM Total Bandwidth [Throughput %]', 'PCIe Throughput [Throughput %]', 'Async CS SM Warps [Occupancy %]'}
        ids = [i for i,n in metrics.items() if n in wanted]
        result['gpu_mean'] = {metrics[i]:v for i,v in db.execute('select metricId,avg(value) from GPU_METRICS where metricId in ('+','.join(map(str,ids))+') group by metricId')}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result={'serial':browser(a.directory/'browser-serial-matched'),'lookahead':browser(a.directory/'browser-adaptive')}
    trace=list(csv.DictReader((a.directory/'transfers.csv').open()))
    # Each ViTPose image upload identifies one frame. Select complete intervals
    # between uploads after warm-up, avoiding initialization/model transfers.
    starts=[i for i,r in enumerate(trace) if r['operation']=='upload' and int(r['bytes'])==1179648]
    hot=trace[starts[30]:starts[-1]]
    frames=len(starts)-31
    result['native_api_per_frame']={op:dict(wall_ms=sum(float(r['wall_ms']) for r in hot if r['operation']==op)/frames,
        bytes=sum(int(r['bytes']) for r in hot if r['operation']==op)/frames) for op in ('upload','download','record_submit','synchronize')}
    result['native_api_frames']=frames
    result['notes']=['RTX 5070 Ti; strict F32; eight CPU cores; browser detector interval 5 and 20 fps cap.',
        'Headless simulated webcam and localhost HTTP; encoder start to response headers is a latency proxy, not physical camera-to-display latency.',
        'External GPU sampling at 10 kHz for eight seconds after warm-up; system-wide counters. Browser input samples differ between A/B runs.',
        'API synchronize duration includes GPU execution; it is not GPU idle time. Upload/download durations are CPU API wall time, not isolated DMA timings.']
    for name in ('parity','parity-upstream-cadence'):
        r=json.loads((a.directory/(name+'.json')).read_text())
        result[name]={key:r[key] for key in ('byte_identical','poses','detect_interval')}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
