"""Shared provenance, profile and video utilities."""
import hashlib,json,shutil,subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def profile(name):
 p=read(ROOT/'configs/profiles.json')[name]
 if p['width']%16 or p['height']%16 or p['num_frames']%4!=1 or p['num_frames']<5 or p['fps']<=0 or p['steps']<1:raise ValueError('Invalid video-generation profile')
 return p

def video_info(path):
 cap=cv2.VideoCapture(str(path))
 try:
  if not cap.isOpened():raise ValueError('Cannot open video: '+str(path))
  info={'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':float(cap.get(cv2.CAP_PROP_FPS)),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
  if info['frames']<1 or not np.isfinite(info['fps']) or info['fps']<=0:raise ValueError('Invalid video timing')
  return info
 finally:cap.release()

def load_video_frames(path,indices):
 cap=cv2.VideoCapture(str(path));frames=[]
 try:
  for index in indices:
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
   if not ok:raise ValueError('Video decode failed at '+str(index))
   frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
 finally:cap.release()
 return frames

def fit_canvas(image,width,height,is_mask=False):
 scale=min(width/image.width,height/image.height);w=max(1,round(image.width*scale));h=max(1,round(image.height*scale));left=(width-w)//2;top=(height-h)//2
 resized=image.resize((w,h),Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS)
 canvas=Image.new('L' if is_mask else 'RGB',(width,height),0 if is_mask else (127,127,127));canvas.paste(resized,(left,top))
 return canvas,{'left':left,'top':top,'width':w,'height':h,'scale':scale}

def snapshot(out):
 paths=list((ROOT/'src/scenedit').glob('*.py'))+list((ROOT/'configs').glob('*.json'))
 result={}
 for p in paths:
  rel=p.relative_to(ROOT);target=out/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);result[rel.as_posix()]=sha(p)
 return result
