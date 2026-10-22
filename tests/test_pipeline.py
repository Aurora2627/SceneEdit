import unittest,tempfile,json,shutil
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from scenedit.prepare import edit_mask,mask_control,prepare
from scenedit.common import fit_canvas
from scenedit.metrics import region_changes
class PipelineTests(unittest.TestCase):
 def test_foreground_white_inverts_for_vace(self):
  m=Image.fromarray(np.array([[0,255]],dtype=np.uint8));self.assertEqual(edit_mask(m,'foreground_white').tolist(),[[True,False]]);self.assertEqual(edit_mask(m,'edit_white').tolist(),[[False,True]])
 def test_explicit_semantics(self):
  with self.assertRaises(ValueError):edit_mask(Image.new('L',(2,2)),'auto')
 def test_gray_only_edit_pixels(self):
  rgb=torch.tensor([[[1,2,3],[4,5,6]]],dtype=torch.uint8);out=mask_control(rgb,torch.tensor([[True,False]]));self.assertEqual(out.tolist(),[[[127,127,127],[4,5,6]]]);self.assertEqual(rgb[0,0].tolist(),[1,2,3])
 def test_letterbox_geometry_and_padding(self):
  canvas,p=fit_canvas(Image.new('L',(80,40),255),160,160,True);self.assertEqual(p['top'],40);self.assertEqual(canvas.getpixel((0,0)),0);self.assertEqual(canvas.getpixel((80,80)),255)
 def test_shape_guard(self):
  with self.assertRaises(ValueError):mask_control(torch.zeros((2,2,3)),torch.zeros((1,2),dtype=torch.bool))
 def test_region_metric_detects_background_only_change(self):
  src=np.zeros((2,2,3),dtype=np.uint8);edit=np.array([[True,False],[True,False]]);out=src.copy();out[edit]=255;metrics=region_changes(src,out,edit);self.assertEqual(metrics['protected_pixel_mae'],0.);self.assertEqual(metrics['background_pixel_mae'],1.)
 def test_unchanged_is_not_edit_success(self):
  src=np.zeros((2,2,3),dtype=np.uint8);r=region_changes(src,src,np.array([[True,False],[True,False]]));self.assertEqual(r['background_pixel_mae'],0.)
 def test_timed_mask_prepare_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for name,is_mask in [('source',False),('mask',True)]:
    w=cv2.VideoWriter(str(root/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),8,(64,32))
    for i in range(6):
     frame=np.zeros((32,64,3),dtype=np.uint8);frame[8:24,16+i:32+i]=255 if is_mask else (30,80,140);w.write(frame)
    w.release()
   job={'job_id':'test-fixture','source_video':str(root/'source.mp4'),'mask':str(root/'mask.mp4'),'mask_semantics':'foreground_white','product_description':'A synthetic rectangle used only as a unit test','target_scene':'test background','source_group':'unit-fixture'};path=root/'job.json';path.write_text(json.dumps(job));out=root/'prepared';meta=prepare(path,out,'smoke');self.assertEqual(meta['source_frame_indices'],[0,1,2,3,4]);self.assertEqual(meta['source_frame_indices'],meta['mask_frame_indices']);self.assertEqual(len(meta['prepared_sha256']),16)
   with self.assertRaises(FileExistsError):prepare(path,out,'smoke')
   job['start_seconds']=100;path.write_text(json.dumps(job))
   with self.assertRaises(ValueError):prepare(path,root/'invalid','smoke')
if __name__=='__main__':unittest.main()
