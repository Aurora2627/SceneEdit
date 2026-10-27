"""Export every frozen new-source result; descriptive analysis after generation only."""

import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, save, sha
from scenedit.benchmark import paired_summary


def main():
    config = read(ROOT / "configs/independent-validation.json")
    run = Path(config["output"])
    state = read(run / "status.json")
    if state["status"] != "complete" or state["completed_runs"] != 8:
        raise ValueError("All eight frozen runs must finish before export")
    rows = read(run / "results.json")
    summary = paired_summary(rows, config)
    for row in rows:
        if sha(Path(row["run"]) / "edited.mp4") != row["output_sha256"]:
            raise ValueError("Changed generated video")
    protocol = read(run / "protocol.json")
    if protocol["config"] != config:
        raise ValueError("Report configuration differs from frozen run")
    environment = read(run / "environment.json")
    public = dict(
        status="automatic_validation_complete_human_review_pending",
        experiment="independent-v3",
        protocol_sha256=sha(run / "protocol.json"),
        source_revision=environment["source_revision"],
        actual_profile=read(Path(rows[0]["prepared"]) / "prepared.json")["profile"],
        config={
            key: config[key]
            for key in ("cases", "scenes", "seeds", "radii", "scale", "scope")
        },
        environment=environment,
        summary=summary,
        rows=[
            {
                key: row[key]
                for key in (
                    "name",
                    "case",
                    "scene",
                    "seed",
                    "radius",
                    "metrics",
                    "seconds",
                    "peak_allocated_gib",
                    "source_sha256",
                    "reference_sha256",
                    "output_sha256",
                )
            }
            for row in rows
        ],
        source_snapshot=protocol["code_sha256"],
        report_source_sha256=sha(Path(__file__)),
    )
    out = Path(__file__).resolve().parent
    save(out / "results.json", public)
    raw = Path(
        "D:/career-projects/data/SceneEdit/validation-tests-v072.log"
    ).read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    (out / "tests.txt").write_text(raw.decode(encoding), encoding="utf-8")
    times = [row["seconds"] for row in rows]
    peaks = [row["peak_allocated_gib"] for row in rows]
    lines = [
        "# v0.7.2 — 新素材成对验证结果",
        "",
        "全部8次CUDA生成完成。人工评审包已生成，真实人工评分尚未收集；本报告不宣称感知质量或语义一致性得到改善。",
        "",
        "## 固定协议",
        "",
        "DAVIS car-shadow与drift-straight；海岸/树林背景；seed=23；保护边界0/3px；控制强度1。448×256、13帧、8fps、20步。共4对生成，但只有2个独立原视频，提示词是重复测量。参数来自旧开发素材，未用本轮结果重新选参。",
        "",
        "## 全部配对结果",
        "",
        "|素材|背景|主体MAE 0px|主体MAE 3px|主体误差下降|边界误差下降|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pair in summary["pairs"]:
        body = pair["metrics"]["foreground_pixel_mae"]
        edge = pair["metrics"]["boundary_pixel_mae"]
        lines.append(
            f"|{pair['case']}|{pair['scene']}|{body['baseline']:.6f}|{body['candidate']:.6f}|{body['reduction_percent']:.2f}%|{edge['reduction_percent']:.2f}%|"
        )
    positive = sum(
        p["metrics"]["foreground_pixel_mae"]["delta"] < 0 for p in summary["pairs"]
    )
    wall = (
        datetime.datetime.fromisoformat(state["finished"])
        - datetime.datetime.fromisoformat(state["started"])
    ).total_seconds()
    lines += [
        "",
        f"4对中有{positive}对主体像素误差降低，其余结果按表保留。负的下降率表示误差上升；不删除负结果。",
        "",
        "所有指标在原始固定区域计算，排除letterbox填充，不进行原始前景复制或输出合成。完整背景变化、耗时、源文件与视频哈希见results.json。背景变化不能解释为符合提示词。",
        "",
        "## 运行与验证",
        "",
        f'- 环境：{environment["gpu"]}；PyTorch {environment["torch"]}；CUDA {environment["cuda"]}。',
        f"- 生成计时范围：{min(times):.1f}–{max(times):.1f}秒；该计时包含管线加载/生成，不包含前置完整权重哈希核验。",
        f"- 队列端到端耗时：{wall/60:.1f}分钟；峰值PyTorch分配显存：{max(peaks):.3f}GiB。",
        "- 45项测试通过；真实评审包另行逐资产校验。",
        "",
        "## 人工评审与限制",
        "",
        "reviewer-1/2分别包含匿名页面及完整视频；答案映射保存在文件夹外。至少两位真实评分者独立填写主体、边缘、场景、时序及整体偏好，再运行analyze_blind_review.py。当前无人工胜率、一致率或kappa结论。",
        "",
        "本轮补充独立素材验证和可复现人工评价流程，未训练生成模型。两段研究素材、13帧短视频不能证明商业广告交付能力、长视频稳定性或广泛泛化。历史v0.6.0时序残差结果仍只适用于历史输出，不能套用于本轮。",
        "",
        "源码与操作：[评审指南](../../docs/independent-review.zh-CN.md)。",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "ALL_NEW_RESULTS_EXPORTED",
        out,
        "positive_body_pairs",
        positive,
        "of",
        len(summary["pairs"]),
    )


if __name__ == "__main__":
    main()
