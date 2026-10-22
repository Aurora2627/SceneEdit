import tempfile
import unittest
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scenedit.io_backend import install_pread_backend

class IOBackendTests(unittest.TestCase):
    def test_pread_preserves_bfloat16_under_inference_mode(self):
        runtime = install_pread_backend()
        self.assertEqual(runtime['backend'], 'pread')
        from diffsynth.core.vram import disk_map
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'small.safetensors'
            source = torch.tensor([0., -0., float('nan'), 1.5], dtype=torch.bfloat16)
            save_file({'weight': source}, str(path))
            with torch.inference_mode():
                with disk_map.safe_open(str(path), framework='pt', device='cpu') as handle:
                    actual = handle.get_tensor('weight')
                self.assertTrue(torch.equal(actual.view(torch.uint8), source.view(torch.uint8)))
            self.assertIsNot(disk_map.safe_open, safe_open)
