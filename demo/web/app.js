const $=id=>document.getElementById(id);
const video=$('video'),empty=$('empty'),file=$('file'),camera=$('camera'),record=$('record'),processButton=$('process'),status=$('status'),bar=$('bar'),download=$('download'),canvas=$('skeleton'),ctx=canvas.getContext('2d');
let sourceURL=null,stream=null,recorder=null,chunks=[],job=null,playing=false;

function message(text,progress){status.textContent=text;if(progress!==undefined)bar.style.width=`${Math.max(0,Math.min(100,progress))}%`}
async function ready(){
  if(!Number.isFinite(video.duration)){video.currentTime=Number.MAX_SAFE_INTEGER;await new Promise(resolve=>video.addEventListener('seeked',resolve,{once:true}));video.currentTime=0}
  empty.hidden=true;video.classList.add('ready');processButton.disabled=false;message(`Ready · ${video.videoWidth}×${video.videoHeight} · ${Number.isFinite(video.duration)?video.duration.toFixed(1):'?'} s`,0)
}
function useBlob(blob){if(sourceURL)URL.revokeObjectURL(sourceURL);sourceURL=URL.createObjectURL(blob);video.srcObject=null;video.src=sourceURL;video.load();video.onloadedmetadata=()=>ready().catch(e=>message(e.message,0))}

file.onchange=()=>{if(file.files[0])useBlob(file.files[0])};
camera.onclick=async()=>{try{stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});video.removeAttribute('src');video.srcObject=stream;video.controls=false;await video.play();empty.hidden=true;video.classList.add('ready');record.disabled=false;processButton.disabled=true;message('Camera ready. Record a short clip, then process it.',0)}catch(e){message(`Camera: ${e.message}`,0)}};
record.onclick=()=>{if(!recorder||recorder.state==='inactive'){
  chunks=[];const type=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(value=>MediaRecorder.isTypeSupported(value))||'';recorder=new MediaRecorder(stream,type?{mimeType:type}:undefined);
  recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};recorder.onstop=()=>{const blob=new Blob(chunks,{type:recorder.mimeType});stream.getTracks().forEach(t=>t.stop());stream=null;video.controls=true;record.disabled=true;record.textContent='Start recording';useBlob(blob)};
  recorder.start(250);record.textContent='Stop recording';message('Recording… inference remains idle.',0)
}else recorder.stop()};

function seek(time){return new Promise((resolve,reject)=>{const done=()=>{cleanup();resolve()},bad=()=>{cleanup();reject(new Error('could not decode this point in the clip'))},cleanup=()=>{video.removeEventListener('seeked',done);video.removeEventListener('error',bad)};video.addEventListener('seeked',done,{once:true});video.addEventListener('error',bad,{once:true});video.currentTime=Math.min(time,Math.max(0,video.duration-.001))})}
function jpeg(canvas){return new Promise((resolve,reject)=>canvas.toBlob(v=>v?resolve(v):reject(new Error('could not encode frame')),'image/jpeg',.88))}
async function api(path,options={}){const response=await fetch(path,options);let value;try{value=await response.json()}catch{value={error:await response.text()}}if(!response.ok)throw new Error(value.error||`HTTP ${response.status}`);return value}
async function waitForJob(id){for(;;){const value=await api(`/api/jobs/${id}`);message(value.error||value.stage,value.state==='complete'?100:74);if(value.state==='complete')return value;if(value.state==='failed')throw new Error(value.error);await new Promise(r=>setTimeout(r,700))}}

processButton.onclick=async()=>{try{
  if(!Number.isFinite(video.duration)||video.duration<=0)throw new Error('wait for the recorded video duration to become available');
  processButton.disabled=true;download.classList.add('disabled');playing=false;
  const fps=Math.max(1,Math.min(30,Number($('fps').value)||10)),frames=Math.max(1,Math.min(120,Math.floor(video.duration*fps)));
  const ratio=Math.min(1,960/Math.max(video.videoWidth,video.videoHeight)),width=Math.max(8,Math.round(video.videoWidth*ratio)),height=Math.max(8,Math.round(video.videoHeight*ratio));
  const capture=document.createElement('canvas');capture.width=width;capture.height=height;const captureContext=capture.getContext('2d',{alpha:false});
  job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:file.files[0]?.name||'webcam recording',width,height,frames,fps})});
  video.pause();for(let i=0;i<frames;i++){await seek(i/fps);captureContext.drawImage(video,0,0,width,height);const blob=await jpeg(capture);await api(`/api/jobs/${job.id}/frames/${i}`,{method:'PUT',headers:{'Content-Type':'image/jpeg'},body:blob});message(`Collecting complete clip · ${i+1}/${frames}`,Math.round((i+1)/frames*65))}
  await api(`/api/jobs/${job.id}/process`,{method:'POST'});job=await waitForJob(job.id);download.href=`/api/jobs/${job.id}/motion.glb`;download.classList.remove('disabled');playing=true;animate(job)
}catch(e){message(e.message,0)}finally{processButton.disabled=false}};

function parsePose(buffer){const view=new DataView(buffer);let magic='';for(let i=0;i<8;i++)magic+=String.fromCharCode(view.getUint8(i));if(magic!=='GEMPOSE1')throw new Error('invalid pose result');let at=8;const positions=[];for(let i=0;i<77;i++){positions.push([view.getFloat32(at,true),view.getFloat32(at+4,true),view.getFloat32(at+8,true)]);at+=12}at+=77*4*4+77*3*4;const parents=[];for(let i=0;i<77;i++){parents.push(view.getInt32(at,true));at+=4}return {positions,parents}}
function draw(pose){const {positions,parents}=pose,W=canvas.width,H=canvas.height;ctx.fillStyle='#080908';ctx.fillRect(0,0,W,H);let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const p of positions){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1])}const scale=.78*Math.min(W/Math.max(.2,maxX-minX),H/Math.max(.2,maxY-minY)),cx=(minX+maxX)/2,cy=(minY+maxY)/2,point=p=>[W/2+(p[0]-cx)*scale,H/2-(p[1]-cy)*scale];ctx.lineWidth=3;ctx.strokeStyle='#c6ff3d';ctx.beginPath();for(let i=0;i<77;i++){if(parents[i]<0||parents[i]>=77)continue;const a=point(positions[i]),b=point(positions[parents[i]]);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1])}ctx.stroke();ctx.fillStyle='#ece9e2';for(const p of positions){const q=point(p);ctx.beginPath();ctx.arc(q[0],q[1],3.3,0,Math.PI*2);ctx.fill()}}
async function animate(value){const cache=new Map();let frame=0,last=performance.now();while(playing&&job?.id===value.id){const now=performance.now();if(now-last>=1000/value.fps){last=now;const index=frame%value.frames;try{let pose=cache.get(index);if(!pose){const response=await fetch(`/api/jobs/${value.id}/poses/${String(index).padStart(6,'0')}`);if(!response.ok)throw new Error('pose unavailable');pose=parsePose(await response.arrayBuffer());cache.set(index,pose)}draw(pose);frame++}catch(e){message(e.message);playing=false}}await new Promise(requestAnimationFrame)}}

ctx.fillStyle='#080908';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#747770';ctx.font='24px system-ui';ctx.textAlign='center';ctx.fillText('Skeleton preview',canvas.width/2,canvas.height/2);
