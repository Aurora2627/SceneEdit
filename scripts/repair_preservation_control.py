"""Repair a historical padding mismatch using one matched baseline, retaining prior runs."""
import argparse,datetime,subprocess,sys,shutil,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scenedit.common import ROOT,read,save,sha,snapshot
from scenedit.infer import verify_inputs,gpu_guard
from scenedit.preservation import prepare_variant,evaluate_fixed,valid_pixels

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);a=ap.parse_args();config=read(a.config);base=Path(config['base_prepared']);out=Path(config['output']);gpu_guard();meta=verify_inputs(base)
 assert read(a.prior/'status.json')['status']=='complete'
 out.mkdir(parents=True,exist_ok=False);state={'status':'running','started':datetime.datetime.now().astimezone().isoformat(),'stage':'matched-baseline','finished_runs':0};save(out/'status.json',state)
 try:
  protocol=dict(config,base_prepared_sha256=sha(base/'prepared.json'),code=snapshot(out),scope='car-roundabout only; matching noneditable letterbox policy for all four conditions; no output compositing',correction={'excluded_baseline':config['baseline_run'],'reason':'Historical reference baseline had editable letterbox; variants had noneditable letterbox. Historical baseline cannot isolate the scale effect.','prior_run':str(a.prior),'old_results_sha256':sha(a.prior/'results.json')});save(out/'protocol.json',protocol);shutil.copyfile(__file__,out/'repair_runner.py')
  item=config['conditions'][0];assert item['scale']==1 and item['radius']==0 and not item.get('reuse_baseline');prepared=out/'prepared'/item['name'];prepare_variant(base,prepared,0,1.0);run=out/item['name']
  with (out/(item['name']+'.log')).open('w',encoding='utf-8') as log:subprocess.run([sys.executable,'-u','-X','utf8','-m','scenedit.infer','--prepared',str(prepared),'--model-dir',config['model_dir'],'--output-dir',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  prior={row['name']:row for row in read(a.prior/'results.json')};rows=[];input_fingerprints={}
  for condition in config['conditions']:
   name=condition['name'];target=run if name==item['name'] else Path(prior[name]['run']);session=read(target/'status.json');prepared_path=Path(session['prepared']);pmeta=verify_inputs(prepared_path)
   assert session['use_reference'] and pmeta['variant']['protection_radius_px']==condition['radius'] and pmeta['profile']['vace_scale']==condition['scale']
   expected=dict(meta['profile'],vace_scale=float(condition['scale']));assert session['profile']==expected and pmeta['prompt']==meta['prompt'] and sha(prepared_path/'reference.png')==sha(base/'reference.png')
   fingerprints=[]
   for i in range(meta['profile']['num_frames']):
    filename=str(i).zfill(4)+'.png';assert sha(prepared_path/'source'/filename)==sha(base/'source'/filename)
    edit=np.array(Image.open(prepared_path/'edit-mask'/filename).convert('L'));valid=valid_pixels(edit.shape,meta['letterbox'][i]).numpy();assert not edit[~valid].any()
    fingerprints.append({'control':sha(prepared_path/'control'/filename),'edit_mask':sha(prepared_path/'edit-mask'/filename)})
   radius=str(condition['radius'])
   if radius in input_fingerprints:assert fingerprints==input_fingerprints[radius],'Scale-only pair changed mask/control'
   else:input_fingerprints[radius]=fingerprints
   fixed=evaluate_fixed(base,target);row=dict(condition,run=str(target),metrics=fixed['mean'],seconds=session['seconds'],peak_allocated_gib=session['peak_allocated_gib'],output_sha256=session['output_sha256'],prepared_sha256=sha(prepared_path/'prepared.json'),fixed_definition=fixed['definition']);rows.append(row);save(out/(name+'-fixed.json'),fixed);save(out/'results.json',rows)
  save(out/'input-consistency.json',{'all_sources_and_references_unchanged':True,'all_letterbox_noneditable':True,'scale_only_pairs_have_identical_controls_and_masks':True,'fingerprints_by_radius':input_fingerprints})
  w,h=meta['profile']['width'],meta['profile']['height'];sheet=Image.new('RGB',(w*3,(h+24)*5),(35,35,35));draw=ImageDraw.Draw(sheet)
  for r,(label,folder) in enumerate([('original',base/'source')]+[(row['name'],Path(row['run'])/'frames') for row in rows]):
   for c,i in enumerate((0,8,16)):
    sheet.paste(Image.open(folder/(str(i).zfill(4)+'.png')).convert('RGB'),(c*w,r*(h+24)+24));draw.text((c*w+5,r*(h+24)+4),label+' frame '+str(i),fill='white')
  sheet.save(out/'review.jpg',quality=95);state.update(status='complete',finished_runs=4,finished=datetime.datetime.now().astimezone().isoformat(),review='static review pending');save(out/'status.json',state);print('MATCHED_COMPARISON_COMPLETE',flush=True)
 except Exception as error:
  state.update(status='failed',error=str(error),traceback=traceback.format_exc());save(out/'status.json',state);raise
if __name__=='__main__':main()
