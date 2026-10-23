"""Pin upstream source/model/data revisions and inspect official example filenames."""

from pathlib import Path
import json, subprocess, time, urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/career-projects/data/SceneEdit")
DATA.mkdir(exist_ok=True)
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"https": "http://127.0.0.1:7890"})
)


def fetch(url):
    for attempt in range(4):
        try:
            with opener.open(url, timeout=60) as response:
                return json.load(response)
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


model = fetch("https://huggingface.co/api/models/Wan-AI/Wan2.1-VACE-1.3B?blobs=true")
data = fetch("https://huggingface.co/api/datasets/ali-vilab/VACE-Benchmark?blobs=true")
for name, value in [("model-api.json", model), ("benchmark-api.json", data)]:
    (DATA / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
commit = (
    subprocess.check_output(
        ["git", "-C", str(ROOT / "vendor/DiffSynth-Studio"), "rev-parse", "HEAD"]
    )
    .decode()
    .strip()
)
lock = {
    "framework": {
        "repository": "https://github.com/modelscope/DiffSynth-Studio",
        "commit": commit,
    },
    "model": {"repository": "Wan-AI/Wan2.1-VACE-1.3B", "revision": model["sha"]},
    "benchmark": {"repository": "ali-vilab/VACE-Benchmark", "revision": data["sha"]},
    "model_reference": "https://github.com/ali-vilab/VACE",
}
(ROOT / "configs/sources.lock.json").write_text(
    json.dumps(lock, indent=2), encoding="utf-8"
)
print("SOURCE_PINS", json.dumps(lock))
print("MODEL_FILES", [(x["rfilename"], x.get("size")) for x in model["siblings"]])
print(
    "BENCHMARK_EXAMPLES",
    [
        x["rfilename"]
        for x in data["siblings"]
        if any(
            v in x["rfilename"]
            for v in (
                "repainting",
                "inpainting",
                "outpainting",
                "swap",
                "mask",
                "json",
                "csv",
            )
        )
    ][:65],
)
