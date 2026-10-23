"""Mask conditioning variants and fixed-region diagnostics; no output compositing."""
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .common import read,save,sha
from .infer import verify_inputs

def dilate(mask,radius):
    if type(radius) is not int or not 0<=radius<=16:raise ValueError('radius must be an integer in 0..16')
    tensor=torch.as_tensor(mask,dtype=torch.float32)
    if tensor.ndim!=2:raise ValueError('mask must be 2D')
    if radius==0:return tensor.bool()
    return F.max_pool2d(tensor[None,None],2*radius+1,stride=1,padding=radius)[0,0].bool()

def conditioning(source,edit,valid,radius):
    source=torch.as_tensor(source,dtype=torch.uint8);edit=torch.as_tensor(edit,dtype=torch.bool);valid=torch.as_tensor(valid,dtype=torch.bool)
    if source.shape!=(*edit.shape,3) or valid.shape!=edit.shape:raise ValueError('unaligned inputs')
    foreground=(~edit)&valid;protected=dilate(foreground,radius)
    new_edit=edit&(~protected)&valid
    control=torch.where(new_edit[...,None],torch.full_like(source,127),source)
    return control,new_edit

def valid_pixels(shape,box):
    valid=torch.zeros(shape,dtype=torch.bool);x,y,w,h=(box[k] for k in ('left','top','width','height'))
    if min(x,y)<0 or min(w,h)<1 or x+w>shape[1] or y+h>shape[0]:raise ValueError('invalid letterbox')
    valid[y:y+h,x:x+w]=True
    return valid

def fixed_region_changes(source,output,original_edit,valid):
    src=torch.as_tensor(source,dtype=torch.float32);dst=torch.as_tensor(output,dtype=torch.float32);edit=torch.as_tensor(original_edit,dtype=torch.bool);valid=torch.as_tensor(valid,dtype=torch.bool)
    if src.shape!=dst.shape or src.shape!=(*edit.shape,3) or valid.shape!=edit.shape:raise ValueError('unaligned metrics')
    foreground=(~edit)&valid;expanded=dilate(foreground,3);eroded=~dilate(~foreground,3)
    regions={'foreground_pixel_mae':foreground,'boundary_pixel_mae':(expanded&~eroded)&valid,'far_background_pixel_change':edit&~expanded&valid}
    delta=(dst-src).abs()/255.
    if not all(region.any() for region in regions.values()):raise ValueError('empty fixed region')
    return {name:float(delta[region].mean()) for name,region in regions.items()}

def prepare_variant(base,out,radius,scale):
    meta=verify_inputs(base)
    if not 0.5<=scale<=2.0:raise ValueError('scale out of experimental range')
    shutil.copytree(base,out)
    for i in range(meta['profile']['num_frames']):
        name=str(i).zfill(4)+'.png';source=np.array(Image.open(base/'source'/name).convert('RGB'));edit=np.array(Image.open(base/'edit-mask'/name).convert('L'))>=128;valid=valid_pixels(edit.shape,meta['letterbox'][i])
        control,new_edit=conditioning(source,edit,valid,radius)
        Image.fromarray(control.numpy()).save(out/'control'/name);Image.fromarray(new_edit.numpy().astype(np.uint8)*255).save(out/'edit-mask'/name)
    meta['profile']['vace_scale']=float(scale)
    meta['variant']={'base_prepared_sha256':sha(base/'prepared.json'),'protection_radius_px':radius,'vace_scale':scale,'reference':'unchanged','source_sha256':sha(Path(__file__))}
    meta['prepared_sha256']={name:sha(out/name) for name in meta['prepared_sha256']}
    save(out/'prepared.json',meta);verify_inputs(out)
    return meta

def evaluate_fixed(base,run):
    meta=verify_inputs(base);status=read(run/'status.json');metrics=read(run/'metrics.json')
    if status['status']!='complete' or sha(run/'edited.mp4')!=status['output_sha256']:raise ValueError('invalid completed run')
    rows=[]
    for i in range(meta['profile']['num_frames']):
        name=str(i).zfill(4)+'.png'
        if sha(run/'frames'/name)!=metrics['output_png_sha256'][name]:raise ValueError('changed output frame')
        source=np.array(Image.open(base/'source'/name).convert('RGB'));output=np.array(Image.open(run/'frames'/name).convert('RGB'));edit=np.array(Image.open(base/'edit-mask'/name).convert('L'))>=128;valid=valid_pixels(edit.shape,meta['letterbox'][i])
        rows.append(fixed_region_changes(source,output,edit,valid))
    return {'mean':{key:sum(row[key] for row in rows)/len(rows) for key in rows[0]},'per_frame':rows,'definition':'Original foreground, fixed 3px boundary ring, background outside ring. Letterbox excluded; identical evaluation regions across all variants. Pixel diagnostics only, no semantic or temporal quality conclusion.'}
