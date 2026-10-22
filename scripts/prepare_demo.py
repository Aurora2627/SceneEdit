"""Prepare an official non-product example solely for integration checks."""
import json
from pathlib import Path
from scenedit.prepare import prepare
from scenedit.common import ROOT,read,save
lock=read(ROOT/'configs/sources.lock.json');revision=lock['benchmark']['revision']
from download_assets import get_file
for name in ('assets/examples/inpainting/ori_video.mp4','assets/examples/inpainting/src_mask.mp4'):get_file(name)
base=Path('D:/career-projects/data/SceneEdit/benchmark')/revision/'assets/examples/inpainting'
job={'job_id':'vace-dragon-background-smoke','source_video':str(base/'ori_video.mp4'),'mask':str(base/'src_mask.mp4'),'mask_semantics':'foreground_white','product_description':'A turquoise Chinese dragon flying through the air','target_scene':'a snowy mountain valley with soft daylight','source_group':'vace-official-inpainting-demo','data_role':'integration_only','license':'Apache-2.0','source_repository':lock['benchmark']['repository'],'source_revision':revision,'note':'Official fantasy-video example; not product footage, no commercial efficacy claim.'}
job_path=ROOT/'examples/vace-demo-job.json';save(job_path,job)
for name in ('smoke','pilot'):
 out=Path('D:/career-projects/runs/SceneEdit/20261022-vace-demo')/('prepared-'+name)
 if out.exists():
  if not (out/'prepared.json').is_file():raise RuntimeError('Incomplete previous preparation: '+str(out))
  print('PREPARED_EXISTS',out)
 else:prepare(job_path,out,name)
