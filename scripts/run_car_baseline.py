"""Fixed four-run reference ablation; no model tuning on the held-out video."""
import datetime,html,json,os,subprocess,sys,time,traceback,shutil
from pathlib import Path
from scenedit.common import ROOT,read,save,sha,snapshot
BASE=Path('D:/career-projects/runs/SceneEdit/20261022-car-background')
MODEL=Path('D:/career-projects/models/SceneEdit/574e6a744642ce3bee319afc31496b88bde8aac4')
SMOKE=Path('D:/career-projects/runs/SceneEdit/20261022-vace-demo/generated-smoke-v3/status.json')
def main():
 out=BASE/'baseline-v1';out.mkdir(parents=True,exist_ok=False)
 state={'status':'waiting','pid':os.getpid(),'started':datetime.datetime.now().astimezone().isoformat(),'purpose':'Automotive advertising background editing; two research benchmark sequences, not a commercial deployment validation.'};save(out/'status.json',state)
 try:
  deadline=time.monotonic()+3600
  while True:
   smoke=read(SMOKE) if SMOKE.exists() else {}
   if smoke.get('status') in ('failed','interrupted'):raise RuntimeError('Integration generation did not succeed')
   if smoke.get('status')=='complete' and all((BASE/case/'prepared-pilot/prepared.json').exists() for case in ('car-roundabout','car-turn')):break
   if time.monotonic()>deadline:raise TimeoutError('Prerequisites did not become ready in one hour')
   time.sleep(15)
  protocol={'profile':read(ROOT/'configs/profiles.json')['pilot'],'cases':[{'sequence':'car-roundabout','role':'development'},{'sequence':'car-turn','role':'heldout_case'}],'conditions':['with-reference','without-reference'],'controls':'Identical prompt, input frames, edit masks, seed and scheduler steps; only reference-image input changes.','metrics':'Protected-region pixel MAE and background pixel change are diagnostics, not semantic or commercial acceptance metrics. Human review required.','prepared_sha256':{case:sha(BASE/case/'prepared-pilot/prepared.json') for case in ('car-roundabout','car-turn')},'model_identity_sha256':sha(MODEL/'identity.json'),'converted_identity_sha256':sha(MODEL/'derived-safetensors/v1/identity.json'),'code_sha256':snapshot(out)};save(out/'protocol.json',protocol);shutil.copy2(__file__,out/'run_car_baseline.py');results=[]
  for case in ('car-roundabout','car-turn'):
   for mode in protocol['conditions']:
    if sha(BASE/case/'prepared-pilot/prepared.json')!=protocol['prepared_sha256'][case]:raise ValueError('Prepared protocol changed')
    name=case+'-'+mode;target=out/name;state.update(status='running',stage=name,finished_runs=len(results));save(out/'status.json',state)
    command=[sys.executable,'-u','-X','utf8','-m','scenedit.infer','--prepared',str(BASE/case/'prepared-pilot'),'--model-dir',str(MODEL),'--output-dir',str(target)]
    if mode=='without-reference':command.append('--without-reference')
    with (out/(name+'.log')).open('w',encoding='utf-8') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    session=read(target/'status.json');metrics=read(target/'metrics.json');results.append({'case':case,'mode':mode,'metrics':metrics['mean'],'seconds':session['seconds'],'peak_allocated_gib':session['peak_allocated_gib'],'manual_review':'pending'});save(out/'results.json',results)
  cards=[]
  for result in results:
   name=result['case']+'-'+result['mode'];cards.append('<section><h2>'+html.escape(name)+'</h2><video controls preload="metadata" src="'+name+'/edited.mp4"></video><p>检查车身颜色/形状、轮廓、海岸背景及运动连贯性；人工结论待填写。</p></section>')
  page='<!doctype html><meta charset="utf-8"><title>SceneEdit 汽车素材换景对照</title><style>body{font-family:system-ui;max-width:1100px;margin:32px auto;background:#f5f5f2}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;padding:20px}video{width:100%}</style><h1>汽车素材换景：参考图对照</h1><p>目标：海岸公路、蓝色海面、远处峭壁与傍晚自然光。两段公开基准素材，仅用于技术验证。</p><main>'+''.join(cards)+'</main>'
  (out/'review.html').write_text(page,encoding='utf-8');state.update(status='complete',finished=datetime.datetime.now().astimezone().isoformat(),finished_runs=len(results),human_review='pending');save(out/'status.json',state);print('CAR_BASELINE_COMPLETE',out,flush=True)
 except Exception as error:
  state.update(status='needs_attention',error=repr(error),traceback=traceback.format_exc());save(out/'status.json',state);raise
if __name__=='__main__':main()
