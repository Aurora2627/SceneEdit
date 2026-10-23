"""One-shot local run: wait for model verification and ClipRank GPU release."""

import argparse, datetime, os, subprocess, sys, time, traceback
from pathlib import Path
from scenedit.common import ROOT, read, save, sha
from scenedit.infer import CLIP_STATUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wait-minutes", type=float, default=480)
    a = ap.parse_args()
    root = Path("D:/career-projects/runs/SceneEdit/20261022-vace-demo")
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / "queue.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    sources = {
        str(p): sha(p)
        for folder in ("src/scenedit", "configs")
        for p in (ROOT / folder).glob("*")
        if p.suffix in (".py", ".json")
    }
    sources[str(Path(__file__))] = sha(Path(__file__))
    state = {
        "status": "waiting",
        "pid": os.getpid(),
        "started": datetime.datetime.now().astimezone().isoformat(),
        "source_sha256": sources,
        "purpose": "Official non-product integration example; not a product-quality validation",
    }
    save(root / "queue-status.json", state)
    try:
        deadline = time.monotonic() + a.max_wait_minutes * 60
        while True:
            clip = read(CLIP_STATUS)
            path = Path("D:/career-projects/data/SceneEdit/download-status.json")
            download = read(path) if path.exists() else {}
            state.update(
                cliprank_status=clip.get("status"),
                download_status=download.get("status"),
            )
            save(root / "queue-status.json", state)
            if clip.get("status") in ("needs_attention", "failed"):
                raise RuntimeError("ClipRank needs attention; refusing to take GPU")
            if (
                clip.get("status") in ("complete", "completed", "published")
                and download.get("status") == "ready"
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Inputs or GPU did not become ready within the wait budget"
                )
            time.sleep(30)
        for name, expected in sources.items():
            if sha(Path(name)) != expected:
                raise RuntimeError("Source changed after queue launch: " + name)
        used = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        if max(int(x.strip()) for x in used.splitlines()) > 1200:
            raise RuntimeError("GPU memory is still occupied; refusing to overlap jobs")
        model = Path(download["destination"])
        for profile in ("smoke", "pilot"):
            state.update(status="running", stage=profile)
            save(root / "queue-status.json", state)
            command = [
                sys.executable,
                "-u",
                "-X",
                "utf8",
                "-m",
                "scenedit.infer",
                "--prepared",
                str(root / ("prepared-" + profile)),
                "--model-dir",
                str(model),
                "--output-dir",
                str(root / ("generated-" + profile)),
            ]
            with (root / (profile + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        state.update(
            status="complete", finished=datetime.datetime.now().astimezone().isoformat()
        )
        save(root / "queue-status.json", state)
        print("DEMO_COMPLETE", root, flush=True)
    except Exception as error:
        state.update(
            status="needs_attention", error=str(error), traceback=traceback.format_exc()
        )
        save(root / "queue-status.json", state)
        raise


if __name__ == "__main__":
    main()
