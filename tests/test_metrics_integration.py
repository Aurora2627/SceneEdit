import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from scenedit.common import save,sha
from scenedit.metrics import evaluate

class MetricIntegrationTests(unittest.TestCase):
    def test_windows_checksum_keys_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);prepared=base/'prepared';run=base/'run'
            for path in (prepared/'source',prepared/'edit-mask',run/'frames'):path.mkdir(parents=True)
            source=np.full((4,4,3),80,dtype=np.uint8);mask=np.zeros((4,4),dtype=np.uint8);mask[:,2:]=255
            output=source.copy();output[:,2:]=160
            Image.fromarray(source).save(prepared/'source/0000.png');Image.fromarray(mask).save(prepared/'edit-mask/0000.png');Image.fromarray(output).save(run/'frames/0000.png')
            hashes={str(Path(name)/'0000.png'):sha(prepared/name/'0000.png') for name in ('source','edit-mask')}
            save(prepared/'prepared.json',{'profile':{'num_frames':1},'prepared_sha256':hashes})
            result=evaluate(prepared,run)
            self.assertEqual(result['mean']['protected_pixel_mae'],0.)
            self.assertAlmostEqual(result['mean']['background_pixel_mae'],80/255,places=6)
            Image.fromarray(source+1).save(prepared/'source/0000.png')
            with self.assertRaisesRegex(ValueError,'changed'):evaluate(prepared,run)
