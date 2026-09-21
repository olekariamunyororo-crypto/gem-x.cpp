#!/usr/bin/env python3
"""Real browser camera -> HTTP -> native live inference smoke test.

Uses Chromium's simulated webcam with a local Y4M video. The inference service
must already be running. Requires the sibling sam3d.cpp DevTools test helper.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'sam3d.cpp/scripts'))
from devtools import CDP

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8099')
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chrome',default='chromium')
    p.add_argument('--detect-interval',type=int,default=7)
    p.add_argument('--poses',type=int,default=72)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
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
            c.evaluate('''window.qa={poses:0,pending:0,maxPending:0,sessionURL:null,timings:[]};window.realFetch=window.fetch;window.fetch=async(...args)=>{
              if(String(args[0]).startsWith('/api/live?'))qa.sessionURL=String(args[0]);
              const frame=String(args[0]).endsWith('/frame');if(frame){qa.pending++;qa.maxPending=Math.max(qa.maxPending,qa.pending)}
              const start=performance.now();try{const r=await realFetch(...args);if(frame){qa.timings.push({start,end:performance.now(),native:Number(r.headers.get('X-GEMX-Inference-Ms')),timing:r.headers.get('Server-Timing')});if(r.status===200)qa.poses++}return r}finally{if(frame)qa.pending--}
            };document.querySelector('#live-start').click()''')
            c.wait(f'window.qa.poses>={a.poses}',timeout=120)
            c.screenshot(a.output/'live.png')
            report=c.evaluate('({timings:qa.timings,sharedPreview:!document.querySelector("#skeleton").hidden&&document.querySelector(".stage").classList.contains("show-pose"),poses:qa.poses,maxPending:qa.maxPending,sessionURL:qa.sessionURL,intervalLocked:document.querySelector("#detect-interval").disabled,status:document.querySelector("#status").textContent,metrics:document.querySelector("#live-metrics").textContent,hasStream:!!document.querySelector("#video").srcObject})')
            assert report['maxPending']<=2 and report['hasStream'] and report['sharedPreview'],report
            assert report['sessionURL']==f'/api/live?detect_interval={a.detect_interval}' and report['intervalLocked'],report
            c.evaluate('document.querySelector("#live-stop").click()')
            c.wait('document.querySelector("#video").srcObject!==null && document.querySelector("#skeleton").hidden && !document.querySelector(".stage").classList.contains("show-pose") && !document.querySelector("#live-start").disabled')
            c.screenshot(a.output/'stopped-camera.png')
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
            print(json.dumps(report))
        finally:
            browser.terminate()
            try:browser.wait(timeout=10)
            except subprocess.TimeoutExpired:browser.kill();browser.wait()

if __name__=='__main__':main()
