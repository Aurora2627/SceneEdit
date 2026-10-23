"""Verify lossless derived checkpoint shards before disk-offloaded inference."""

from pathlib import Path
from .common import read, sha


def model_paths(model_dir):
    original = read(model_dir / "identity.json")
    root = model_dir / "derived-safetensors/v1"
    derived = read(root / "identity.json")
    if derived["original_model_identity_sha256"] != sha(model_dir / "identity.json"):
        raise ValueError("Converted checkpoint belongs to another model identity")
    paths = [str(model_dir / "diffusion_pytorch_model.safetensors")]
    for name, folder in (
        ("models_t5_umt5-xxl-enc-bf16.pth", "text-encoder"),
        ("Wan2.1_VAE.pth", "vae"),
    ):
        info = derived["conversions"][name]
        expected = next(x["sha256"] for x in original["files"] if x["file"] == name)
        if (
            info["source_sha256"] != expected
            or info.get("all_tensors_byte_equal") is not True
        ):
            raise ValueError("Checkpoint conversion is not verified lossless")
        shards = []
        keys = []
        for item in info["shards"]:
            if Path(item["file"]).name != item["file"]:
                raise ValueError("Invalid shard path")
            path = root / folder / item["file"]
            if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                raise ValueError("Converted shard checksum mismatch")
            shards.append(str(path))
            keys.extend(item["keys"])
        if len(keys) != len(set(keys)) or len(keys) != info["tensor_count"]:
            raise ValueError("Converted parameter index mismatch")
        paths.append(shards)
    return paths, sha(root / "identity.json")
