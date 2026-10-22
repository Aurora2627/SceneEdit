"""Audited, local-only CUDA VACE inference for background editing."""
import argparse,datetime,subprocess,sys,time,traceback,os
from pathlib import Path
from PIL import Image
from .common import ROOT,read,save,sha,snapshot,video_info
CLIP_STATUS=Path('D:/career-projects/runs/ClipRank-head-training/20261022_first-training/status.json')
def gpu_guard():
 if CLIP_STATUS.exists():
  s=read(CLIP_STATUS)
  if s.get('status') not in ('complete','completed','published'):raise RuntimeError('ClipRank is not complete. SceneEdit GPU inference must wait: '+s.get('stage','unknown'))
def verify_inputs(prepared):
 meta=read(prepared/'prepared.json')
 for name,expected in meta['prepared_sha256'].items():
  if sha(prepared/name)!=expected:raise ValueError('Prepared asset changed: '+name)
 return meta
def verify_model(model_dir):
 lock=read(ROOT/'configs/sources.lock.json');identity=read(model_dir/'identity.json')
 if identity['revision']!=lock['model']['revision'] or identity['repository']!=lock['model']['repository']:raise ValueError('Model source mismatch')
 required={'diffusion_pytorch_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth'}
 if not required.issubset({r['file'] for r in identity['files']}):raise ValueError('Incomplete model identity')
 for record in identity['files']:
  path=model_dir/record['file']
  if not path.is_file() or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError('Model checksum mismatch: '+record['file'])
 commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/DiffSynth-Studio'),'rev-parse','HEAD'],text=True).strip()
 dirty=subprocess.check_output(['git','-C',str(ROOT/'vendor/DiffSynth-Studio'),'status','--porcelain'],text=True).strip()
 if commit!=lock['framework']['commit'] or dirty:raise ValueError('Inference engine differs from source lock')
 return identity

def infer(prepared,model_dir,out,use_reference=True):
 gpu_guard();meta=verify_inputs(prepared);identity=verify_model(model_dir)
 from .checkpoints import model_paths
 checkpoint_paths,converted_identity_sha=model_paths(model_dir)
 import torch
 from .io_backend import install_pread_backend
 io_runtime=install_pread_backend()
 from diffsynth.pipelines.wan_video import WanVideoPipeline,ModelConfig
 import imageio.v2 as imageio
 if not torch.cuda.is_available():raise RuntimeError('CUDA required; use scenedit-cuda')
 p=meta['profile'];out.mkdir(parents=True,exist_ok=False)
 status={'status':'loading','started':datetime.datetime.now().astimezone().isoformat(),'prepared':str(prepared),'profile':p,'use_reference':use_reference,'model_identity_sha256':sha(model_dir/'identity.json'),'model':identity,'code':snapshot(out),'checkpoint_io':io_runtime,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)};save(out/'status.json',status)
 try:
  free,total=torch.cuda.mem_get_info()
  if p['vram_limit_gib']>total/(1024**3)-1:raise RuntimeError('Profile requires more GPU memory. Choose smoke/pilot or use a larger GPU.')
  offload={'offload_dtype':'disk','offload_device':'disk','onload_dtype':'disk','onload_device':'disk','preparing_dtype':torch.bfloat16,'preparing_device':'cuda','computation_dtype':torch.bfloat16,'computation_device':'cuda'}
  os.environ['DIFFSYNTH_DISK_MAP_BUFFER_SIZE']='134217728'
  configs=[ModelConfig(path=path,**offload) for path in checkpoint_paths]
  status.update(converted_checkpoint_identity_sha256=converted_identity_sha,offload_policy='disk-to-CUDA streaming; no retained CPU model weights');save(out/'status.json',status)
  started=time.perf_counter();torch.cuda.reset_peak_memory_stats();print('PIPELINE_LOADING',flush=True)
  pipe=WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda',model_configs=configs,tokenizer_config=ModelConfig(path=str(model_dir/'google/umt5-xxl')),redirect_common_files=False,vram_limit=p['vram_limit_gib'])
  frames=lambda name:[Image.open(prepared/name/(str(i).zfill(4)+'.png')).convert('RGB') for i in range(p['num_frames'])]
  status.update(status='generating');save(out/'status.json',status);print('VIDEO_GENERATING',flush=True)
  with torch.inference_mode():
   video=pipe(prompt=meta['prompt'],negative_prompt='distorted product, changed logo, flicker, blurry, deformed hands, duplicate objects, text overlay',vace_video=frames('control'),vace_video_mask=frames('edit-mask'),vace_reference_image=Image.open(prepared/'reference.png').convert('RGB') if use_reference else None,vace_scale=p['vace_scale'],seed=p['seed'],height=p['height'],width=p['width'],num_frames=p['num_frames'],cfg_scale=p['cfg_scale'],num_inference_steps=p['steps'],tiled=True)
  if len(video)!=p['num_frames']:raise ValueError('Generated frame count mismatch')
  (out/'frames').mkdir()
  for i,frame in enumerate(video):frame.save(out/'frames'/(str(i).zfill(4)+'.png'))
  imageio.mimwrite(str(out/'edited.mp4'),video,fps=p['fps'],codec='libx264',quality=8,macro_block_size=1)
  info=video_info(out/'edited.mp4')
  if (info['frames'],info['width'],info['height'])!=(p['num_frames'],p['width'],p['height']):raise ValueError('Encoded video differs from requested profile')
  torch.cuda.synchronize();status.update(status='evaluating',seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,output_sha256=sha(out/'edited.mp4'),video_info=info,finished=datetime.datetime.now().astimezone().isoformat());save(out/'status.json',status)
  from .metrics import evaluate
  evaluate(prepared,out);status.update(status='complete');save(out/'status.json',status);print('SCENEDIT_COMPLETE',out,flush=True)
 except Exception as error:
  status.update(status='failed',error=str(error),traceback=traceback.format_exc());save(out/'status.json',status);raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--without-reference',action='store_true');a=ap.parse_args();infer(a.prepared,a.model_dir,a.output_dir,not a.without_reference)
if __name__=='__main__':main()
