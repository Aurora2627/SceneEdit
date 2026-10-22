"""Lossless pth -> sharded safetensors conversion with memory-mapped input and byte checks."""
import argparse,gc,json,datetime
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scenedit.common import sha,save,read

def convert(source,out,expected_sha,max_bytes=256*1024*1024):
 if sha(source)!=expected_sha:raise ValueError('Original checkpoint identity mismatch')
 out.mkdir(parents=True,exist_ok=True)
 state=torch.load(str(source),weights_only=True,map_location='cpu',mmap=True)
 if not isinstance(state,dict) or not state or not all(isinstance(v,torch.Tensor) for v in state.values()):raise ValueError('Expected a flat tensor state dictionary')
 records=[];group={};size=0;all_keys=[]
 def flush(tensors):
  if not tensors:return
  name='weights-'+str(len(records)).zfill(3)+'.safetensors';path=out/name
  if not path.exists():
   temp=path.with_suffix('.partial');save_file(tensors,str(temp));temp.replace(path)
  with safe_open(str(path),framework='pt',device='cpu') as f:
   if set(f.keys())!=set(tensors):raise ValueError('Shard key mismatch')
   for key,value in tensors.items():
    restored=f.get_tensor(key)
    if restored.dtype!=value.dtype or restored.shape!=value.shape or not torch.equal(restored.view(torch.uint8),value.view(torch.uint8)):raise ValueError('Tensor bytes changed: '+key)
  records.append({'file':name,'sha256':sha(path),'bytes':path.stat().st_size,'keys':list(tensors)});all_keys.extend(tensors);print('SHARD_VERIFIED',source.name,name,len(tensors),flush=True)
 for name,value in state.items():
  n=value.numel()*value.element_size()
  if group and size+n>max_bytes:flush(group);group={};size=0;gc.collect()
  group[name]=value.contiguous();size+=n
 flush(group)
 if set(all_keys)!=set(state):raise ValueError('Conversion missed parameter keys')
 manifest={'source_file':str(source),'source_sha256':expected_sha,'conversion':'memory-mapped pth to safetensors; no dtype conversion, quantization or parameter updates','tensor_count':len(state),'all_tensors_byte_equal':True,'shards':records,'finished':datetime.datetime.now().astimezone().isoformat()};save(out/'identity.json',manifest);del state;gc.collect();return manifest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);a=ap.parse_args();original=read(a.model_dir/'identity.json');root=a.model_dir/'derived-safetensors/v1';identities={}
 for filename in ('models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth'):
  info=next(x for x in original['files'] if x['file']==filename);out=root/('text-encoder' if filename.startswith('models_t5') else 'vae');identities[filename]=convert(a.model_dir/filename,out,info['sha256'])
 save(root/'identity.json',{'original_model_identity_sha256':sha(a.model_dir/'identity.json'),'conversions':identities});print('ALL_CHECKPOINT_CONVERSIONS_VERIFIED',flush=True)
if __name__=='__main__':main()
