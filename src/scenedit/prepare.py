"""Turn product footage and explicit masks into audited VACE background-edit controls."""

import argparse, datetime, hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import (
    ROOT,
    read,
    save,
    sha,
    profile,
    video_info,
    load_video_frames,
    fit_canvas,
    snapshot,
)


def edit_mask(mask, semantics):
    if semantics not in ("foreground_white", "edit_white"):
        raise ValueError("Explicit mask semantics required")
    white = torch.as_tensor(np.array(mask.convert("L"), copy=True)) >= 128
    return ~white if semantics == "foreground_white" else white


def mask_control(rgb, edit):
    if rgb.ndim != 3 or rgb.shape[:2] != edit.shape or rgb.shape[2] != 3:
        raise ValueError("Mask and RGB shape mismatch")
    return torch.where(edit[:, :, None], torch.full_like(rgb, 127), rgb)


def prepare(job_path, out, profile_name):
    job = read(job_path)
    p = profile(profile_name)
    source = Path(job["source_video"])
    mask_path = Path(job["mask"])
    semantics = job["mask_semantics"]
    for field in ("job_id", "product_description", "target_scene", "source_group"):
        if not isinstance(job.get(field), str) or not job[field].strip():
            raise ValueError("Missing " + field)
    if not source.is_file() or not mask_path.is_file():
        raise FileNotFoundError("Source video and mask are required")
    source_info = video_info(source)
    start = float(job.get("start_seconds", 0))
    times = [start + i / p["fps"] for i in range(p["num_frames"])]
    if start < 0:
        raise ValueError("Negative start time")
    indices = [round(t * source_info["fps"]) for t in times]
    if len(set(indices)) != len(indices) or max(indices) >= source_info["frames"]:
        raise ValueError("Source is too short or target fps duplicates video frames")
    video = load_video_frames(source, indices)
    if mask_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        if not job.get("static_mask_confirmed", False):
            raise ValueError("Static mask requires explicit fixed-subject confirmation")
        masks = [Image.open(mask_path).convert("L")] * len(video)
        mask_info = {"static": True}
        mask_indices = [0] * len(video)
    else:
        mask_info = video_info(mask_path)
        mask_indices = [round(t * mask_info["fps"]) for t in times]
        if max(mask_indices) >= mask_info["frames"]:
            raise ValueError("Mask timeline does not cover requested source interval")
        masks = [im.convert("L") for im in load_video_frames(mask_path, mask_indices)]
    if any(v.size != m.size for v, m in zip(video, masks)):
        raise ValueError("Source and mask must have matching spatial coordinates")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("source", "control", "edit-mask"):
        (out / name).mkdir()
    kept = []
    placements = []
    for index, (frame, mask) in enumerate(zip(video, masks)):
        foreground = (~edit_mask(mask, semantics)).numpy().astype("uint8") * 255
        image, placement = fit_canvas(frame, p["width"], p["height"])
        keep, _ = fit_canvas(Image.fromarray(foreground), p["width"], p["height"], True)
        edit = ~torch.as_tensor(np.array(keep, copy=True), dtype=torch.bool)
        fraction = float((~edit).float().mean())
        if not 0.005 < fraction < 0.95:
            raise ValueError("Degenerate foreground/edit mask at frame " + str(index))
        rgb = torch.as_tensor(np.array(image, copy=True))
        control = mask_control(rgb, edit)
        image.save(out / "source" / (str(index).zfill(4) + ".png"))
        Image.fromarray(control.numpy()).save(
            out / "control" / (str(index).zfill(4) + ".png")
        )
        Image.fromarray(edit.numpy().astype("uint8") * 255).save(
            out / "edit-mask" / (str(index).zfill(4) + ".png")
        )
        kept.append(fraction)
        placements.append(placement)
    reference = job.get("reference_image")
    if reference:
        reference = Path(reference)
        image = Image.open(reference).convert("RGB")
        image, _ = fit_canvas(image, p["width"], p["height"])
        image.save(out / "reference.png")
        reference_sha = sha(reference)
    else:
        first = np.array(Image.open(out / "source/0000.png"))
        mask = np.array(Image.open(out / "edit-mask/0000.png")) >= 128
        first[mask] = 255
        Image.fromarray(first).save(out / "reference.png")
        reference_sha = None
    prompt = (
        job["product_description"].strip().rstrip(".")
        + ". The scene is "
        + job["target_scene"].strip().rstrip(".")
        + ". The product remains visually consistent with the reference."
    )
    source_hashes = snapshot(out)
    prepared = {
        "job": job,
        "profile_name": profile_name,
        "profile": p,
        "prompt": prompt,
        "source_sha256": sha(source),
        "mask_sha256": sha(mask_path),
        "reference_input_sha256": reference_sha,
        "source_video_info": source_info,
        "mask_info": mask_info,
        "timestamps_seconds": times,
        "source_frame_indices": indices,
        "mask_frame_indices": mask_indices,
        "foreground_fraction": kept,
        "letterbox": placements,
        "mask_convention": "white means generate/edit; black means keep; background control pixels are 127",
        "code_sha256": source_hashes,
        "created": datetime.datetime.now().astimezone().isoformat(),
        "prepared_sha256": {
            str(f.relative_to(out)): sha(f)
            for name in ("source", "control", "edit-mask")
            for f in (out / name).glob("*.png")
        },
    }
    prepared["prepared_sha256"]["reference.png"] = sha(out / "reference.png")
    save(out / "prepared.json", prepared)
    print("JOB_PREPARED", job["job_id"], len(video), profile_name, flush=True)
    return prepared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--profile", choices=("smoke", "pilot", "quality"), default="pilot")
    a = ap.parse_args()
    prepare(a.job, a.output_dir, a.profile)


if __name__ == "__main__":
    main()
