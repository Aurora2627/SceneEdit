"""Predeclared development-only 2x2 VACE conditioning ablation."""
import argparse,datetime,shutil,subprocess,sys,traceback
from pathlib import Path
from PIL import Image,ImageDraw
from scenedit.common import ROOT,read,save,sha,snapshot
from scenedit.infer import gpu_guard,verify_inputs
from scenedit.preservation import prepare_variant,evaluate_fixed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();config=read(a.config)
    gpu_guard();base=Path(config['base_prepared']);baseline=Path(config['baseline_run']);out=Path(config['output']);meta=verify_inputs(base)
    assert 'car-roundabout' in str(base)
    if any(item.get('reuse_baseline') for item in config['conditions']):raise ValueError('Historical baseline reuse disabled: all conditions must share the padding policy')
    out.mkdir(parents=True,exist_ok=False);state={'status':'preparing','started':datetime.datetime.now().astimezone().isoformat(),'finished_runs':0};save(out/'status.json',state)
    try:
        protocol=dict(config,config_sha256=sha(a.config),base_prepared_sha256=sha(base/'prepared.json'),code=snapshot(out),scope='Development sequence only; car-turn not used for tuning. No trained weights, no compositing. All reference images, prompts, seeds and scheduler settings fixed.');save(out/'protocol.json',protocol);shutil.copyfile(__file__,out/'runner.py')
        prepared={}
        for item in config['conditions']:
            if item.get('reuse_baseline'):continue
            target=out/'prepared'/item['name'];prepare_variant(base,target,item['radius'],item['scale']);prepared[item['name']]=target
        rows=[]
        for item in config['conditions']:
            name=item['name'];state.update(status='running',stage=name);save(out/'status.json',state)
            if item.get('reuse_baseline'):
                run=baseline;previous=read(run/'status.json');assert previous['profile']==meta['profile'] and previous['use_reference'] is True
            else:
                run=out/name
                command=[sys.executable,'-u','-X','utf8','-m','scenedit.infer','--prepared',str(prepared[name]),'--model-dir',config['model_dir'],'--output-dir',str(run)]
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            fixed=evaluate_fixed(base,run);session=read(run/'status.json')
            row=dict(item,metrics=fixed['mean'],seconds=session['seconds'],peak_allocated_gib=session['peak_allocated_gib'],output_sha256=session['output_sha256'],fixed_definition=fixed['definition'],run=str(run));rows.append(row);save(out/'results.json',rows);save(out/(name+'-fixed.json'),fixed)
            state['finished_runs']=len(rows);save(out/'status.json',state);print('CONDITION_COMPLETE',name,row['metrics'],flush=True)
        p=meta['profile'];w,h=p['width'],p['height'];indices=[0,8,16];sheet=Image.new('RGB',(3*w,(h+24)*5),(35,35,35));draw=ImageDraw.Draw(sheet)
        for r,(name,folder) in enumerate([('original',base/'source')]+[(row['name'],Path(row['run'])/'frames') for row in rows]):
            for c,i in enumerate(indices):
                sheet.paste(Image.open(folder/(str(i).zfill(4)+'.png')).convert('RGB'),(c*w,r*(h+24)+24));draw.text((c*w+5,r*(h+24)+4),name+' | frame '+str(i),fill='white')
        sheet.save(out/'review.jpg',quality=95)
        state.update(status='complete',finished=datetime.datetime.now().astimezone().isoformat(),review='Static comparison pending; no automatic quality winner');save(out/'status.json',state)
    except Exception as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc());save(out/'status.json',state);raise
if __name__=='__main__':main()
