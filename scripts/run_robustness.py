"""Frozen six-run paired validation; no tuning, compositing or cherry-picking."""
import argparse,datetime,shutil,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from scenedit.common import ROOT,read,save,sha,snapshot
from scenedit.infer import verify_inputs,gpu_guard
from scenedit.preservation import prepare_variant,evaluate_fixed,valid_pixels
from scenedit.robustness import paired_results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();config=read(a.config);gpu_guard();torch.set_num_threads(4)
 pairs=config['pairs'];assert [(p['case'],p['seed']) for p in pairs]==[('car-roundabout',7),('car-roundabout',123),('car-turn',42)];assert config['radii']==[0,3] and config['scale']==1.0
 out=Path(config['output']);out.mkdir(parents=True,exist_ok=False);state={'status':'preparing','started':datetime.datetime.now().astimezone().isoformat(),'finished_runs':0,'planned_runs':6};save(out/'status.json',state)
 try:
  protocol=dict(config,config_sha256=sha(a.config),code=snapshot(out),runner_sha256=sha(Path(__file__)),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),cpu_preprocessing_threads=4,full_video_review='pending');save(out/'protocol.json',protocol);shutil.copyfile(__file__,out/'runner.py');shutil.copyfile(a.config,out/'config.json')
  jobs=[];fingerprints={}
  for pair in pairs:
   base=Path(pair['prepared']);meta=verify_inputs(base)
   for radius in config['radii']:
    name=pair['case']+'-seed'+str(pair['seed'])+'-radius'+str(radius);prepared=out/'prepared'/name;pmeta=prepare_variant(base,prepared,radius,config['scale']);pmeta['profile']['seed']=pair['seed'];save(prepared/'prepared.json',pmeta);verify_inputs(prepared)
    assert sha(prepared/'reference.png')==sha(base/'reference.png') and pmeta['prompt']==meta['prompt']
    frame_fingerprints=[]
    for i in range(meta['profile']['num_frames']):
     filename=str(i).zfill(4)+'.png';assert sha(base/'source'/filename)==sha(prepared/'source'/filename)
     edit=np.array(Image.open(prepared/'edit-mask'/filename).convert('L'));valid=valid_pixels(edit.shape,meta['letterbox'][i]).numpy();assert not edit[~valid].any()
     frame_fingerprints.append({'source':sha(prepared/'source'/filename),'control':sha(prepared/'control'/filename),'mask':sha(prepared/'edit-mask'/filename)})
    group=pair['case']+'-radius'+str(radius)
    if group in fingerprints:assert fingerprints[group]==frame_fingerprints,'Conditioning changed across seeds'
    else:fingerprints[group]=frame_fingerprints
    jobs.append(dict(pair,radius=radius,name=name,prepared_variant=str(prepared),prepared_sha256=sha(prepared/'prepared.json')))
  save(out/'input-consistency.json',{'conditioning_identical_across_seeds':True,'sources_and_references_unchanged':True,'padding_noneditable_for_all_conditions':True,'fingerprints':fingerprints,'jobs':jobs})
  rows=[]
  for job in jobs:
   name=job['name'];prepared=Path(job['prepared_variant']);assert sha(prepared/'prepared.json')==job['prepared_sha256'];verify_inputs(prepared);state.update(status='running',stage=name);save(out/'status.json',state);run=out/name
   command=[sys.executable,'-u','-X','utf8','-m','scenedit.infer','--prepared',str(prepared),'--model-dir',config['model_dir'],'--output-dir',str(run)]
   with (out/(name+'.log')).open('w',encoding='utf-8') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   session=read(run/'status.json');assert session['use_reference'] and session['profile']==read(prepared/'prepared.json')['profile'];fixed=evaluate_fixed(Path(job['prepared']),run);save(out/(name+'-fixed.json'),fixed)
   row=dict(job,metrics=fixed['mean'],seconds=session['seconds'],peak_allocated_gib=session['peak_allocated_gib'],output_sha256=session['output_sha256'],model_identity_sha256=session['model_identity_sha256'],run=str(run));rows.append(row);save(out/'results.json',rows);state['finished_runs']=len(rows);save(out/'status.json',state);print('ROBUSTNESS_RUN_COMPLETE',name,row['metrics'],flush=True)
  effects=paired_results(rows,[(p['case'],p['seed']) for p in pairs]);save(out/'paired-effects.json',effects)
  for case in ('car-roundabout','car-turn'):
   selected=[row for row in rows if row['case']==case];base=Path(selected[0]['prepared']);p=read(base/'prepared.json')['profile'];w,h=p['width'],p['height'];sheet=Image.new('RGB',(w*3,(h+24)*(len(selected)+1)),(35,35,35));draw=ImageDraw.Draw(sheet)
   for r,(label,folder) in enumerate([('original',base/'source')]+[(row['name'],Path(row['run'])/'frames') for row in selected]):
    for c,i in enumerate((0,8,16)):
     sheet.paste(Image.open(folder/(str(i).zfill(4)+'.png')).convert('RGB'),(c*w,r*(h+24)+24));draw.text((c*w+5,r*(h+24)+4),label+' frame '+str(i),fill='white')
   sheet.save(out/(case+'-review.jpg'),quality=95)
  state.update(status='complete',finished=datetime.datetime.now().astimezone().isoformat(),full_video_review='pending');save(out/'status.json',state);print('ROBUSTNESS_COMPLETE',effects,flush=True)
 except Exception as error:
  state.update(status='failed',error=str(error),traceback=traceback.format_exc());save(out/'status.json',state);raise
if __name__=='__main__':main()
