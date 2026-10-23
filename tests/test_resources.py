import unittest, tempfile, sys
from pathlib import Path
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from http_range import RangeReader
from convert_checkpoints import convert
from scenedit.common import sha


class Response:
    def __init__(self, data, start, end, status=206):
        self.data = data
        self.status_code = status
        self.headers = {
            "ETag": '"test"',
            "Content-Range": f"bytes {start}-{end}/{len(data)}",
        }
        self.start = start
        self.end = end

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, n):
        yield self.data[self.start : self.end + 1]


class Session:
    def __init__(self, status=206):
        self.data = b"abcdefghijk"
        self.status = status
        self.calls = 0

    def get(self, url, headers, **kwargs):
        self.calls += 1
        a, b = map(int, headers["Range"].split("=")[1].split("-"))
        return Response(self.data, a, b, self.status)

    def close(self):
        pass


class ResourceTests(unittest.TestCase):
    def test_range_seek_and_block_reuse(self):
        with patch("http_range.requests.Session", return_value=Session()) as factory:
            with RangeReader(
                "https://example.invalid", '"test"', 11, block_size=4
            ) as reader:
                reader.seek(3)
                self.assertEqual(reader.read(5), b"defgh")
                reader.seek(5)
                self.assertEqual(reader.read(3), b"fgh")
                self.assertEqual(reader.session.calls, 2)
                reader.seek(-3, 2)
                self.assertEqual(reader.read(99), b"ijk")

    def test_full_download_fallback_rejected(self):
        with (
            patch("http_range.requests.Session", return_value=Session(200)),
            patch("http_range.time.sleep"),
        ):
            with RangeReader(
                "https://example.invalid", '"test"', 11, block_size=4
            ) as reader:
                with self.assertRaises(RuntimeError):
                    reader.read(1)

    def test_lossless_checkpoint_split(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "tiny.pth"
            torch.save(
                {
                    "a": torch.arange(12, dtype=torch.float32),
                    "b": torch.tensor([float("nan"), -0.0, 1.0], dtype=torch.bfloat16),
                },
                source,
            )
            identity = convert(source, root / "shards", sha(source), max_bytes=16)
            self.assertTrue(identity["all_tensors_byte_equal"])
            self.assertEqual(identity["tensor_count"], 2)
            self.assertEqual(len(identity["shards"]), 2)

    def test_wrong_original_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "tiny.pth"
            torch.save({"a": torch.ones(2)}, source)
            with self.assertRaises(ValueError):
                convert(source, root / "shards", "invalid")


if __name__ == "__main__":
    unittest.main()
