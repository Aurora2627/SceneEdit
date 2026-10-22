"""Pinned, resumable downloads; weights stay outside the Git checkout."""
import os,json,time,hashlib,datetime,argparse
from pathlib import Path
os.environ.setdefault('HTTP_PROXY','http://127.0.0.1:7890')
os.environ.setdefault('HTTPS_PROXY','http://127.0.0.1:7890')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='180'
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('D:/career-projects/data/SceneEdit')
LOCK=json.loads((ROOT/'configs/sources.lock.json').read_text(encoding='utf-8-sig'))
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def metadata(kind):
 import requests
 path=DATA/(kind+'-api.json');expected=LOCK[kind]['revision']
 if path.exists():
  result=json.loads(path.read_text())
  if result.get('sha')==expected:return result
 family='models' if kind=='model' else 'datasets'
 url='https://huggingface.co/api/'+family+'/'+LOCK[kind]['repository']+'/revision/'+expected+'?blobs=true'
 response=requests.get(url,timeout=90);response.raise_for_status();result=response.json()
 if result.get('sha')!=expected:raise ValueError('Metadata revision mismatch')
 DATA.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result),encoding='utf-8');return result

def get_file(name):
 import requests
 revision=LOCK['benchmark']['revision'];dest=DATA/'benchmark'/revision/name;dest.parent.mkdir(parents=True,exist_ok=True)
 meta=metadata('benchmark')
 info=next(x for x in meta['siblings'] if x['rfilename']==name)
 def valid():return dest.exists() and dest.stat().st_size==info['size'] and (not info.get('lfs') or digest(dest)==info['lfs']['sha256'])
 if not valid():
  for attempt in range(4):
   try:
    url='https://huggingface.co/datasets/'+LOCK['benchmark']['repository']+'/resolve/'+revision+'/'+name
    with requests.get(url,stream=True,timeout=(30,180)) as response:
     response.raise_for_status()
     with dest.with_suffix(dest.suffix+'.partial').open('wb') as output:
      for block in response.iter_content(1024*1024):output.write(block)
    dest.with_suffix(dest.suffix+'.partial').replace(dest)
    if not valid():raise ValueError('Downloaded file checksum/size mismatch: '+name)
    break
   except Exception:
    if attempt==3:raise
    time.sleep(2**attempt)
 print('ASSET_READY',name,dest.stat().st_size,flush=True)
 return dest
def weights():
 from huggingface_hub import hf_hub_download
 import fnmatch
 revision=LOCK['model']['revision'];dest=Path('D:/career-projects/models/SceneEdit')/revision
 selected=['diffusion_pytorch_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','config.json','google/umt5-xxl/*']
 status=DATA/'download-status.json';DATA.mkdir(parents=True,exist_ok=True)
 state={'status':'downloading','pid':os.getpid(),'destination':str(dest),'revision':revision,'files':[]}
 def update():
  state['updated']=datetime.datetime.now().astimezone().isoformat();temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(status)
 update()
 try:
  info_list=[x for x in metadata('model')['siblings'] if any(fnmatch.fnmatch(x['rfilename'],p) for p in selected)]
  for info in info_list:
   name=info['rfilename'];state.update(current_file=name,current_bytes=info['size']);update()
   for attempt in range(12):
    state['attempt']=attempt+1;update()
    try:
     path=Path(hf_hub_download(LOCK['model']['repository'],filename=name,revision=revision,local_dir=str(dest)))
     if path.stat().st_size!=info['size']:raise ValueError('Wrong file size: '+name)
     checksum=digest(path)
     if info.get('lfs') and checksum!=info['lfs']['sha256']:raise ValueError('Wrong SHA256: '+name)
     record={'file':name,'bytes':path.stat().st_size,'sha256':checksum};state['files'].append(record);update();print('MODEL_FILE_VERIFIED',name,record['bytes'],flush=True);break
    except Exception as error:
     state['last_error']=repr(error);update();print('DOWNLOAD_RETRY',name,attempt+1,repr(error),flush=True)
     if attempt==11:raise
     time.sleep(min(2**attempt,30))
  identity={'repository':LOCK['model']['repository'],'revision':revision,'files':state['files']};(dest/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
  state.update(status='ready',finished=datetime.datetime.now().astimezone().isoformat());state.pop('last_error',None);update();print('MODEL_VERIFIED',len(state['files']),flush=True)
 except Exception as error:
  state.update(status='failed',error=repr(error));update();raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--weights',action='store_true');p.add_argument('--benchmark-file',action='append',default=[]);a=p.parse_args()
 for name in a.benchmark_file:get_file(name)
 if a.weights:weights()
