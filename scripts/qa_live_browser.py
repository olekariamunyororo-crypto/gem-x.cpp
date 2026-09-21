#!/usr/bin/env python3
"""Real browser camera -> HTTP -> native live inference smoke test.

Uses Chromium's simulated webcam with a local Y4M video. The inference service
must already be running. Uses the bundled DevTools helper.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

from devtools import CDP

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8099')
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chrome',default='chromium')
    p.add_argument('--detect-interval',type=int,default=7)
    p.add_argument('--poses',type=int,default=72)
    p.add_argument('--serial',action='store_true',help='Check the saved serial baseline binary')
    p.add_argument('--nsys',type=Path,help='Sample GPU counters while the browser runs; directory containing target-linux-x64/nsys')
    p.add_argument('--image',default='gemx-reference:e2e',help='Reference/profiler container image')
    p.add_argument('--gpu-metrics-set',default='gb20x-top',help='Nsight metric set for the selected GPU architecture')
    p.add_argument('--gpu-metrics-device',default='0')
    a=p.parse_args()
    if len(os.sched_getaffinity(0))>8:p.error('restrict CPU affinity to eight cores')
    a.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gemx-camera-') as profile, (a.output/'chrome.log').open('w') as log:
        browser=subprocess.Popen([a.chrome,'--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-gpu',
            '--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',
            '--use-file-for-fake-video-capture='+str(a.video.resolve()),
            '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
            '--user-data-dir='+profile,'--window-size=1440,1100','about:blank'],stdout=log,stderr=log)
        try:
            portfile=Path(profile)/'DevToolsActivePort';deadline=time.monotonic()+20
            while not portfile.exists():
                if browser.poll() is not None or time.monotonic()>deadline:raise RuntimeError('browser failed to start')
                time.sleep(.1)
            port=int(portfile.read_text().splitlines()[0])
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as response:targets=json.load(response)
            c=CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'))
            c.call('Page.enable');c.call('Runtime.enable')
            c.call('Page.navigate',dict(url=a.url))
            c.wait('typeof document.querySelector("#live-start")?.onclick === "function"')
            c.evaluate(f'document.querySelector("#detect-interval").value={a.detect_interval}')
            c.evaluate('''window.qa={poses:0,pending:0,maxPending:0,sessionURL:null,timings:[],encodes:[]};const originalEncode=HTMLCanvasElement.prototype.toBlob;HTMLCanvasElement.prototype.toBlob=function(callback,...args){const start=performance.now();return originalEncode.call(this,blob=>{qa.encodes.push({start,end:performance.now()});callback(blob)},...args)};window.realFetch=window.fetch;window.fetch=async(...args)=>{
              if(String(args[0]).startsWith('/api/live?'))qa.sessionURL=String(args[0]);
              const frame=String(args[0]).endsWith('/frame');if(frame){qa.pending++;qa.maxPending=Math.max(qa.maxPending,qa.pending)}
              const start=performance.now();try{const r=await realFetch(...args);if(frame){qa.timings.push({start,end:performance.now(),native:Number(r.headers.get('X-GEMX-Inference-Ms')),timing:r.headers.get('Server-Timing')});if(r.status===200)qa.poses++}return r}finally{if(frame)qa.pending--}
            };document.querySelector('#live-start').click()''')
            if a.nsys:
                c.wait('window.qa.poses>=32',timeout=120)
                root=Path(__file__).resolve().parents[1]
                mountout='/work/'+str(a.output.resolve().relative_to(root))
                command=['docker','run','--rm','--network','none','--cpuset-cpus','0-7','--cap-add','SYS_ADMIN','--device','nvidia.com/gpu=all','-v',str(a.nsys.resolve())+':/nsys:ro','-v',str(root)+':/work','--entrypoint','/nsys/target-linux-x64/nsys',a.image,'profile','--trace=none','--sample=none','--cpuctxsw=none','--gpu-metrics-devices='+a.gpu_metrics_device,'--gpu-metrics-set='+a.gpu_metrics_set,'--gpu-metrics-frequency=10000','--duration=8','--export=sqlite','--output',mountout+'/gpu','sleep','8']
                (a.output/'sampling-command.json').write_text(json.dumps(command,indent=2)+'\n')
                with (a.output/'sampling.log').open('w') as samplelog:
                    subprocess.run(command,stdout=samplelog,stderr=subprocess.STDOUT,check=True,timeout=60)
            c.wait(f'window.qa.poses>={a.poses}',timeout=120)
            if not a.serial:
                c.wait('document.querySelector(".brand img")?.naturalWidth>0')
            c.screenshot(a.output/'live.png')
            report=c.evaluate('({encodes:qa.encodes,timings:qa.timings,sharedPreview:!document.querySelector("#skeleton").hidden&&document.querySelector(".stage").classList.contains("show-pose"),poses:qa.poses,maxPending:qa.maxPending,sessionURL:qa.sessionURL,intervalLocked:document.querySelector("#detect-interval").disabled,status:document.querySelector("#status").textContent,metrics:document.querySelector("#live-metrics").textContent,hasStream:!!document.querySelector("#video").srcObject})')
            assert report['maxPending']<=2 and report['hasStream'] and report['sharedPreview'],report
            assert report['sessionURL']==f'/api/live?detect_interval={a.detect_interval}'+('' if a.serial else '&pipeline=2') and report['intervalLocked'],report
            c.evaluate('document.querySelector("#live-stop").click()')
            c.wait('document.querySelector("#video").srcObject!==null && document.querySelector("#skeleton").hidden && !document.querySelector(".stage").classList.contains("show-pose") && !document.querySelector("#live-start").disabled')
            c.screenshot(a.output/'stopped-camera.png')
            if not a.serial:
                c.call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
                assert c.evaluate('document.documentElement.scrollWidth<=window.innerWidth'), 'mobile layout overflows'
                c.screenshot(a.output/'mobile.png')
                c.call('Emulation.clearDeviceMetricsOverride')
            c.evaluate('document.querySelector("#live-stop").click()')
            c.wait('document.querySelector("#video").srcObject===null && !document.querySelector("#live-start").disabled')
            # Stop waits for the worker to release GPU ownership before restart.
            time.sleep(.5)
            before=c.evaluate('qa.poses')
            c.evaluate('document.querySelector("#live-start").click()')
            c.wait(f'qa.poses>{before+2}',timeout=120)
            c.evaluate('document.querySelector("#live-stop").click()')
            c.wait('!document.querySelector("#live-start").disabled')
            time.sleep(.5)
            c.evaluate('var inputMode=document.querySelector("#mode");inputMode.value="offline";inputMode.dispatchEvent(new Event("change"))')
            c.wait('document.querySelector("#live-start").hidden && !document.querySelector("#process").hidden')
            c.screenshot(a.output/'offline.png')
            # Simulate a denied permission through the browser API boundary.
            c.evaluate('navigator.mediaDevices.getUserMedia=()=>Promise.reject(new DOMException("Permission denied","NotAllowedError"));var inputMode=document.querySelector("#mode");inputMode.value="live";inputMode.dispatchEvent(new Event("change"))')
            c.wait('!document.querySelector("#live-start").hidden')
            c.evaluate('document.querySelector("#live-start").click()')
            c.wait('document.querySelector("#status").textContent.includes("Permission denied") && !document.querySelector("#live-start").disabled')
            failures=[e for e in c.events if e.get('method')=='Runtime.exceptionThrown']
            assert not failures,failures
            report.update(restart=True,offline_switch=True,permission_error_recovery=True,browser_exceptions=failures)
            (a.output/'browser.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k not in ('timings','encodes')}))
        finally:
            browser.terminate()
            try:browser.wait(timeout=10)
            except subprocess.TimeoutExpired:browser.kill();browser.wait()

if __name__=='__main__':main()
