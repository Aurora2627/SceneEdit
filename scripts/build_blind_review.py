"""Create an offline anonymous A/B review package; keep the answer key outside it."""

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, save, sha
from scenedit.infer import verify_inputs
from scenedit.review import assignments, protocol_id, DIMENSIONS

LABELS = dict(
    subject="主体外观与身份保持",
    boundary="主体边缘自然程度",
    scene="目标背景与指令匹配",
    temporal="运动与跨帧稳定性",
    overall="整体效果",
)


def build_package(results_path, out, seed):
    rows = read(results_path)
    groups = {}
    for row in rows:
        pair = row["name"].rsplit("-radius", 1)[0]
        if row["radius"] in groups.setdefault(pair, {}):
            raise ValueError("Duplicate condition")
        groups[pair][row["radius"]] = row
    if not groups or any(set(pair) != {0, 3} for pair in groups.values()):
        raise ValueError("Incomplete paired results")
    out.mkdir(parents=True, exist_ok=False)
    media = out / "media"
    media.mkdir()
    public, private = [], []
    for assignment in assignments(sorted(groups), seed):
        pair = groups[assignment["pair_id"]]
        original = verify_inputs(Path(pair[0]["prepared"]))
        other = verify_inputs(Path(pair[3]["prepared"]))
        if (
            original["source_sha256"] != other["source_sha256"]
            or original["prompt"] != other["prompt"]
            or sha(Path(pair[0]["prepared"]) / "reference.png")
            != sha(Path(pair[3]["prepared"]) / "reference.png")
        ):
            raise ValueError("Mismatched source, prompt or reference")
        task = assignment["task_id"]
        source_file = media / (task + "-source.mp4")
        frames = [
            Image.open(Path(pair[0]["prepared"]) / "source" / f"{i:04d}.png").convert(
                "RGB"
            )
            for i in range(original["profile"]["num_frames"])
        ]
        imageio.mimwrite(
            str(source_file),
            frames,
            fps=original["profile"]["fps"],
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        assets = dict(source="media/" + source_file.name)
        for side in ("A", "B"):
            radius = 3 if side == assignment["candidate_side"] else 0
            row = pair[radius]
            run = Path(row["run"])
            if (
                read(run / "status.json")["status"] != "complete"
                or sha(run / "edited.mp4") != row["output_sha256"]
            ):
                raise ValueError("Incomplete or changed reviewed video")
            target = media / (task + "-" + side + ".mp4")
            shutil.copyfile(run / "edited.mp4", target)
            assets[side] = "media/" + target.name
        public.append(
            dict(
                task_id=task,
                prompt=original["prompt"],
                assets=assets,
                sha256={label: sha(out / path) for label, path in assets.items()},
            )
        )
        private.append(
            dict(
                assignment,
                case=pair[0]["case"],
                metrics={str(r): pair[r]["metrics"] for r in (0, 3)},
            )
        )
    identifier = protocol_id(public)
    save(out / "tasks.json", dict(protocol_id=identifier, tasks=public))
    key_path = out.parent / (out.name + "-private-key.json")
    if key_path.exists():
        raise FileExistsError("Answer key already exists")
    save(
        key_path,
        dict(protocol_id=identifier, tasks=private, results_sha256=sha(results_path)),
    )
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>视频编辑匿名评审</title>',
        "<style>body{max-width:1100px;margin:30px auto;font:16px sans-serif}video{width:32%}section{border-top:1px solid #bbb;padding:24px 0}label{display:block;margin:12px 0}select{margin-left:12px}button{padding:12px}</style>",
        "<h1>视频背景编辑 A/B 评审</h1><p>按原视频、目标描述独立比较 A/B。每段请完整播放；画质相近选“相当”，两者均不可接受选“均不合格”。不要查看答案文件。请使用个人代号，不填写姓名或联系方式。</p>",
        '<label>评审者代号 <input id="rater" required></label>',
    ]
    for task in public:
        parts.append(
            '<section data-task="'
            + task["task_id"]
            + '"><h2>任务 '
            + task["task_id"]
            + "</h2><p>"
            + html.escape(task["prompt"])
            + "</p>"
        )
        parts.append("<p>左：原视频　中：A　右：B</p>")
        for label in ("source", "A", "B"):
            parts.append(
                '<video controls loop muted playsinline preload="metadata" src="'
                + task["assets"][label]
                + '"></video>'
            )
        for dim in DIMENSIONS:
            parts.append(
                "<label>"
                + LABELS[dim]
                + '<select data-dim="'
                + dim
                + '"><option value="">请选择</option><option value="A">A 更好</option><option value="B">B 更好</option><option value="tie">相当</option><option value="neither">均不合格</option></select></label>'
            )
        parts.append("</section>")
    parts.append(
        '<button id="export">导出完整评分 JSON</button><p id="message"></p><script>'
    )
    parts.append("const protocol=" + json.dumps(identifier) + ";")
    parts.append(
        "document.getElementById('export').onclick=()=>{const rater=document.getElementById('rater').value.trim();const ratings=[...document.querySelectorAll('section')].map(s=>({task_id:s.dataset.task,preferences:Object.fromEntries([...s.querySelectorAll('select')].map(e=>[e.dataset.dim,e.value]))}));if(!rater||ratings.some(r=>Object.values(r.preferences).some(v=>!v))){document.getElementById('message').textContent='请填写代号并完成每一项。';return;}const blob=new Blob([JSON.stringify({protocol_id:protocol,rater_id:rater,ratings},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};</script></html>"
    )
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(
        "REVIEW_PACKAGE_READY",
        out,
        "tasks",
        len(public),
        "KEY_DO_NOT_SHARE",
        key_path,
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    build_package(args.results, args.output, args.seed)
