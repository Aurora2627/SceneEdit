"""Pixel diagnostics, not semantic quality scores or production acceptance criteria."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read, save, sha


def region_changes(source, output, edit):
    if source.shape != output.shape or source.shape[:2] != edit.shape:
        raise ValueError("Unaligned metric inputs")
    if not edit.any() or edit.all():
        raise ValueError("Both edit and protected regions are required")
    delta = np.abs(output.astype(np.float32) - source.astype(np.float32)) / 255.0
    return {
        "protected_pixel_mae": float(delta[~edit].mean()),
        "background_pixel_mae": float(delta[edit].mean()),
    }


def evaluate(prepared, run):
    meta = read(prepared / "prepared.json")
    rows = []
    output_hashes = {}
    expected_hashes = {
        key.replace(chr(92), "/"): value
        for key, value in meta["prepared_sha256"].items()
    }
    for i in range(meta["profile"]["num_frames"]):
        name = str(i).zfill(4) + ".png"
        source = np.array(Image.open(prepared / "source" / name).convert("RGB"))
        output = np.array(Image.open(run / "frames" / name).convert("RGB"))
        edit = np.array(Image.open(prepared / "edit-mask" / name).convert("L")) >= 128
        for folder in ("source", "edit-mask"):
            if sha(prepared / folder / name) != expected_hashes[folder + "/" + name]:
                raise ValueError("Prepared metric inputs changed")
        rows.append(dict(frame=i, **region_changes(source, output, edit)))
        output_hashes[name] = sha(run / "frames" / name)
    result = {
        "frames": len(rows),
        "mean": {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("protected_pixel_mae", "background_pixel_mae")
        },
        "per_frame": rows,
        "output_png_sha256": output_hashes,
        "interpretation": "Raw generated frames; lower protected-region change is desirable. Background change only detects editing, not semantic correctness. No pass/fail threshold calibrated. No copy-back or compositing used. Human review must check product/logo, boundary, scene instruction and temporal consistency.",
    }
    save(run / "metrics.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    print(evaluate(a.prepared, a.run))


if __name__ == "__main__":
    main()
