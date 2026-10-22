"""Prepare research footage for automotive-ad background editing, without a full dataset download."""
import argparse,datetime,io,json,os,zipfile
from pathlib import Path
os.environ.setdefault('HTTP_PROXY','http://127.0.0.1:7890');os.environ.setdefault('HTTPS_PROXY','http://127.0.0.1:7890')
import cv2,numpy as np
from PIL import Image,ImageDraw
from scenedit.common import ROOT,sha,save,read
from scenedit.prepare import prepare
from http_range import RangeReader
URL='https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
ETAG='"31a3032d-54c6d0af8b4dc"';SIZE=832766765
DATA=Path('D:/career-projects/data/SceneEdit/davis-car-cases')
def build_case(zf,name,role):
 root=DATA/name;root.mkdir(parents=True,exist_ok=True);records=[];images=[];masks=[]
 for index in range(51):
  pair=[]
  for folder,suffix in (('JPEGImages','.jpg'),('Annotations','.png')):
   member=f'DAVIS/{folder}/480p/{name}/{index:05d}{suffix}';path=root/'raw'/folder/(f'{index:05d}'+suffix);path.parent.mkdir(parents=True,exist_ok=True)
   info=zf.getinfo(member)
   if not path.exists():
    content=zf.read(member);path.write_bytes(content)
   else:
    import zlib
    if path.stat().st_size!=info.file_size or (zlib.crc32(path.read_bytes())&0xffffffff)!=info.CRC:raise ValueError('Cached frame mismatch: '+member)
   records.append({'member':member,'crc32':info.CRC,'bytes':path.stat().st_size,'sha256':sha(path)})
   pair.append(np.array(Image.open(path).convert('RGB')) if folder=='JPEGImages' else np.array(Image.open(path)))
  images.append(pair[0]);masks.append((pair[1]>0).astype(np.uint8)*255)
  if index%10==0:print('DAVIS_FRAME_PAIR',name,index,flush=True)
 height,width=images[0].shape[:2]
 for filename,frames in (('source.avi',images),('foreground-mask.avi',masks)):
  path=root/filename
  if not path.exists():
   writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),25,(width,height))
   if not writer.isOpened():raise RuntimeError('Lossless video encoder is unavailable')
   for frame in frames:writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) if frame.ndim==3 else cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR))
   writer.release()
 provenance={'dataset':'DAVIS 2017 trainval 480p','official_page':'https://davischallenge.org/davis2017/code.html','archive_url':URL,'archive_etag':ETAG,'archive_bytes':SIZE,'sequence':name,'source_group':name,'role':role,'frames':records,'assembled_fps':25,'timing_note':'Frames are deliberately assembled at 25 fps for this editing benchmark; no claim of recovering original recording timing.','video_sha256':sha(root/'source.avi'),'mask_sha256':sha(root/'foreground-mask.avi')};save(root/'provenance.json',provenance)
 job={'job_id':name+'-coast','source_video':str(root/'source.avi'),'mask':str(root/'foreground-mask.avi'),'mask_semantics':'foreground_white','product_description':'A passenger car driving, retaining the same body color, shape and appearance','target_scene':'a coastal road with the blue sea and distant cliffs, warm late afternoon daylight','source_group':name,'data_role':role,'dataset':'DAVIS 2017','provenance_sha256':sha(root/'provenance.json')};save(ROOT/'examples'/('davis-'+name+'.json'),job)
 for profile in ('smoke','pilot'):
  out=Path('D:/career-projects/runs/SceneEdit/20261022-car-background')/name/('prepared-'+profile)
  if out.exists():
   if not (out/'prepared.json').exists():raise RuntimeError('Incomplete previous preparation: '+str(out))
  else:prepare(ROOT/'examples'/('davis-'+name+'.json'),out,profile)
 sheet=Image.new('RGB',(768,320),'white');draw=ImageDraw.Draw(sheet)
 for j,index in enumerate((0,25,50)):
  sheet.paste(Image.fromarray(images[index]).resize((256,144)),(j*256,16));sheet.paste(Image.fromarray(masks[index]).convert('RGB').resize((256,144)),(j*256,176));draw.text((j*256,0),name+' frame '+str(index),fill='black')
 sheet.save(root/'inspection.jpg');print('DAVIS_CASE_READY',name,role,flush=True)
def main():
 DATA.mkdir(parents=True,exist_ok=True)
 with RangeReader(URL,ETAG,SIZE) as remote:
  with zipfile.ZipFile(remote) as zf:
   names=zf.namelist();print('ARCHIVE_INDEX_READY',len(names),flush=True)
   for filename in names:
    if filename.lower().endswith(('readme.md','readme.txt')) and zf.getinfo(filename).file_size<100000:
     (DATA/'dataset-readme.txt').write_bytes(zf.read(filename));break
   build_case(zf,'car-roundabout','development')
   build_case(zf,'car-turn','heldout_case')
  print('PARTIAL_ARCHIVE_TRANSFER_BYTES',remote.bytes_downloaded,flush=True)
if __name__=='__main__':main()
