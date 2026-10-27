"""Fetch only frozen DAVIS sequences, verify archive CRCs, and prepare paired jobs."""

import argparse
import datetime
import os
import sys
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, save, sha, profile, snapshot
from scenedit.prepare import prepare
from scenedit.benchmark import planned_jobs
from http_range import RangeReader
from prepare_davis_cases import URL, ETAG, SIZE


def prepare_cases(config_path):
    config = read(config_path)
    planned_jobs(config)
    data = Path(config["data_root"])
    data.mkdir(parents=True, exist_ok=True)
    frozen = data / "protocol.json"
    if frozen.exists() and read(frozen) != config:
        raise ValueError("Existing preparation uses a different protocol")
    save(frozen, config)
    save(data / "preparation-source-sha256.json", snapshot(data))
    save(
        data / "preparation-runner.json",
        {
            "sha256": sha(Path(__file__)),
            "started": datetime.datetime.now().astimezone().isoformat(),
        },
    )
    with RangeReader(URL, ETAG, SIZE) as remote, zipfile.ZipFile(remote) as archive:
        for case in config["cases"]:
            root = data / case
            names = sorted(
                n
                for n in archive.namelist()
                if n.startswith(f"DAVIS/JPEGImages/480p/{case}/") and n.endswith(".jpg")
            )[:51]
            if len(names) < 17:
                raise ValueError("Missing or short source sequence: " + case)
            images, masks, hashes = [], [], {}
            for member in names:
                pair = []
                for path in (
                    member,
                    member.replace("JPEGImages", "Annotations").replace(".jpg", ".png"),
                ):
                    relative = "/".join(path.split("/")[1:])
                    local = root / "raw" / relative
                    info = archive.getinfo(path)
                    if local.exists():
                        payload = local.read_bytes()
                    else:
                        cached = (
                            Path(config.get("cache_root", config["data_root"]))
                            / case
                            / "raw"
                            / relative
                        )
                        payload = (
                            cached.read_bytes()
                            if cached.exists()
                            else archive.read(path)
                        )
                        local.parent.mkdir(parents=True, exist_ok=True)
                        local.write_bytes(payload)
                    if (
                        len(payload) != info.file_size
                        or zlib.crc32(payload) & 0xFFFFFFFF != info.CRC
                    ):
                        raise ValueError("Invalid cached/downloaded frame: " + path)
                    hashes[relative] = sha(local)
                    pair.append(
                        np.array(Image.open(local).convert("RGB"))
                        if "JPEGImages" in path
                        else (np.array(Image.open(local)) > 0).astype(np.uint8) * 255
                    )
                images.append(pair[0])
                masks.append(pair[1])
            height, width = images[0].shape[:2]
            for name, frames in (("source.avi", images), ("mask.avi", masks)):
                target = root / name
                if not target.exists():
                    writer = cv2.VideoWriter(
                        str(target),
                        cv2.VideoWriter_fourcc(*"FFV1"),
                        25,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Lossless encoder unavailable")
                    try:
                        for frame in frames:
                            writer.write(
                                cv2.cvtColor(
                                    frame,
                                    (
                                        cv2.COLOR_RGB2BGR
                                        if frame.ndim == 3
                                        else cv2.COLOR_GRAY2BGR
                                    ),
                                )
                            )
                    finally:
                        writer.release()
            save(
                root / "provenance.json",
                dict(
                    dataset="DAVIS 2017 trainval 480p",
                    archive_url=URL,
                    archive_etag=ETAG,
                    archive_bytes=SIZE,
                    sequence=case,
                    frames=len(images),
                    raw_sha256=hashes,
                    assembled_fps=25,
                    source_sha256=sha(root / "source.avi"),
                    mask_sha256=sha(root / "mask.avi"),
                    role="new_source_validation",
                    timing_note="Deliberately assembled at 25 fps; not original capture timing.",
                ),
            )
            for scene, prompt in config["scenes"].items():
                job = dict(
                    job_id=case + "-" + scene,
                    source_video=str(root / "source.avi"),
                    mask=str(root / "mask.avi"),
                    mask_semantics="foreground_white",
                    product_description="A passenger car driving, retaining the same body color, shape and appearance",
                    target_scene=prompt,
                    source_group=case,
                    data_role="new_source_validation",
                    provenance_sha256=sha(root / "provenance.json"),
                )
                job_path = root / (scene + "-job.json")
                if job_path.exists() and read(job_path) != job:
                    raise ValueError("Prepared job changed")
                save(job_path, job)
                target = root / ("prepared-" + scene)
                if not target.exists():
                    prepare(job_path, target, config["profile"])
                from scenedit.infer import verify_inputs

                meta = verify_inputs(target)
                if meta["job"] != job or meta["profile"] != profile(config["profile"]):
                    raise ValueError(
                        "Cached preparation does not match frozen job/profile"
                    )
                print("NEW_CASE_PREPARED", case, scene, flush=True)
        print("INDEPENDENT_INPUTS_READY", remote.bytes_downloaded, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/independent-validation.json"
    )
    prepare_cases(parser.parse_args().config)
