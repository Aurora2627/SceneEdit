import unittest,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from scenedit.common import save,sha,read
from scenedit.preservation import prepare_variant
class VariantPreparationTests(unittest.TestCase):
 def test_scale_only_pair_has_identical_conditioning_including_padding(self):
  with tempfile.TemporaryDirectory() as folder:
   base=Path(folder)/'base';base.mkdir();source=np.full((20,20,3),80,dtype=np.uint8);edit=np.full((20,20),255,dtype=np.uint8);edit[7:13,7:13]=0
   for name,array in [('source',source),('control',source),('edit-mask',edit)]:
    (base/name).mkdir();Image.fromarray(array).save(base/name/'0000.png')
   Image.fromarray(source).save(base/'reference.png');paths=['source/0000.png','control/0000.png','edit-mask/0000.png','reference.png']
   save(base/'prepared.json',{'profile':{'num_frames':1,'vace_scale':1.0},'letterbox':[{'left':0,'top':2,'width':20,'height':16}],'prepared_sha256':{name:sha(base/name) for name in paths}})
   a,b=Path(folder)/'a',Path(folder)/'b';prepare_variant(base,a,0,1.0);prepare_variant(base,b,0,1.3)
   for name in paths:self.assertEqual(sha(a/name),sha(b/name))
   self.assertTrue((np.array(Image.open(base/'edit-mask/0000.png'))[:2]==255).all());self.assertTrue((np.array(Image.open(a/'edit-mask/0000.png'))[:2]==0).all())
