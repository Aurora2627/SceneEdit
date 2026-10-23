"""Evaluate all six frozen outputs without modifying or regenerating them."""

import datetime
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, save, sha
from scenedit.infer import verify_inputs
from scenedit.preservation import valid_pixels
from scenedit.diagnostics import edge_error, temporal_error, flow_consistency_mask


def weighted(rows, value_key, count_key):
    usable = [r for r in rows if r[value_key] is not None and r[count_key] > 0]
    count = sum(r[count_key] for r in usable)
    return sum(r[value_key] * r[count_key] for r in usable) / count if count else None


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    config = read(ROOT / "configs/temporal-diagnostics.json")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "running",
        "started": datetime.datetime.now().astimezone().isoformat(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "Exploratory diagnostics on previously inspected outputs; no generation or training.",
    }
    save(out / "status.json", state)
    try:
        save(out / "protocol.json", config)
        sources = [
            *ROOT.glob("src/scenedit/*.py"),
            *ROOT.glob("configs/*.json"),
            Path(__file__),
        ]
        hashes = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            target = out / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = sha(source)
        save(out / "source-sha256.json", hashes)
        original = Path(config["input_runs"])
        previous_rows = read(original / "results.json")
        protocol = read(ROOT / "configs/robustness.json")
        expected = {
            p["case"] + "-seed" + str(p["seed"]) + "-radius" + str(radius)
            for p in protocol["pairs"]
            for radius in protocol["radii"]
        }
        names = [r["name"] for r in previous_rows]
        if len(names) != 6 or len(set(names)) != 6 or set(names) != expected:
            raise ValueError("Missing, duplicated or unplanned original runs")
        results = []
        cache = {}
        for row in previous_rows:
            base, run = Path(row["prepared"]), Path(row["run"])
            meta = verify_inputs(base)
            status, metrics = read(run / "status.json"), read(run / "metrics.json")
            video_sha = sha(run / "edited.mp4")
            if (
                status["status"] != "complete"
                or video_sha != row["output_sha256"]
                or video_sha != status["output_sha256"]
            ):
                raise ValueError("Changed or incomplete run: " + row["name"])
            source_frames, outputs, masks = [], [], []
            frame_hashes = {}
            for index in range(meta["profile"]["num_frames"]):
                name = str(index).zfill(4) + ".png"
                path = run / "frames" / name
                digest = sha(path)
                if digest != metrics["output_png_sha256"][name]:
                    raise ValueError("Changed output PNG: " + str(path))
                frame_hashes[name] = digest
                source_frames.append(
                    np.array(Image.open(base / "source" / name).convert("RGB"))
                )
                outputs.append(np.array(Image.open(path).convert("RGB")))
                edit = (
                    np.array(Image.open(base / "edit-mask" / name).convert("L")) >= 128
                )
                valid = valid_pixels(edit.shape, meta["letterbox"][index]).numpy()
                masks.append((~edit) & valid)
            edges = [
                dict(frame=i, **edge_error(src, dst, mask))
                for i, (src, dst, mask) in enumerate(zip(source_frames, outputs, masks))
            ]
            temporal = []
            for index in range(1, len(source_frames)):
                key = (row["case"], index)
                if key not in cache:
                    previous = cv2.cvtColor(
                        source_frames[index - 1], cv2.COLOR_RGB2GRAY
                    )
                    current = cv2.cvtColor(source_frames[index], cv2.COLOR_RGB2GRAY)
                    parameters = config["farneback"]
                    back = cv2.calcOpticalFlowFarneback(
                        current, previous, None, **parameters
                    )
                    front = cv2.calcOpticalFlowFarneback(
                        previous, current, None, **parameters
                    )
                    confidence = flow_consistency_mask(
                        back, front, config["consistency_threshold_px"]
                    )
                    flow_path = (
                        out
                        / "flows"
                        / (row["case"] + "-" + str(index).zfill(4) + ".npz")
                    )
                    flow_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(
                        flow_path,
                        backward=back,
                        forward=front,
                        confidence=confidence.numpy(),
                    )
                    cache[key] = (back, confidence, sha(flow_path))
                back, confidence, flow_sha = cache[key]
                values = temporal_error(
                    source_frames[index - 1],
                    source_frames[index],
                    outputs[index - 1],
                    outputs[index],
                    masks[index - 1],
                    masks[index],
                    back,
                    confidence,
                )
                temporal.append(dict(frame=index, flow_sha256=flow_sha, **values))
            accepted = sum(r["temporal_pixels"] for r in temporal)
            available = sum(r["foreground_pixels"] for r in temporal)
            result = {
                "name": row["name"],
                "case": row["case"],
                "seed": row["seed"],
                "radius": row["radius"],
                "role": row["role"],
                "original_spatial_metrics": row["metrics"],
                "edge_gradient_mae": weighted(
                    edges, "edge_gradient_mae", "edge_pixels"
                ),
                "temporal_residual_mae": weighted(
                    temporal, "temporal_residual_mae", "temporal_pixels"
                ),
                "source_warp_mae": weighted(
                    temporal, "source_warp_mae", "temporal_pixels"
                ),
                "output_warp_mae": weighted(
                    temporal, "output_warp_mae", "temporal_pixels"
                ),
                "temporal_coverage": accepted / available if available else None,
                "usable_pairs": sum(r["temporal_pixels"] > 0 for r in temporal),
                "edge_per_frame": edges,
                "temporal_per_pair": temporal,
                "input_prepared_sha256": sha(base / "prepared.json"),
                "output_sha256": video_sha,
                "output_png_sha256": frame_hashes,
            }
            results.append(result)
            save(out / "results.json", results)
            print(
                "DIAGNOSTIC_COMPLETE",
                row["name"],
                result["edge_gradient_mae"],
                result["temporal_residual_mae"],
                flush=True,
            )
        state.update(
            status="complete",
            completed_runs=len(results),
            finished=datetime.datetime.now().astimezone().isoformat(),
        )
        save(out / "status.json", state)
        print("ALL_DIAGNOSTICS_COMPLETE", out, flush=True)
    except Exception as error:
        state.update(status="failed", error=str(error))
        save(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
