"""Export the complete diagnostic dataset and paired comparisons for review."""

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, save, sha


def main():
    config = read(ROOT / "configs/temporal-diagnostics.json")
    run = Path(config["output"])
    status, results = read(run / "status.json"), read(run / "results.json")
    if status["status"] != "complete" or len(results) != 6:
        raise ValueError("All six diagnostics must finish before reporting")
    report = ROOT / "reports/v0.6.0"
    if (report / "README.md").exists():
        raise FileExistsError("Do not overwrite a historical report")
    pairs = []
    keys = sorted({(r["case"], r["seed"]) for r in results})
    for case, seed in keys:
        selected = {
            r["radius"]: r for r in results if (r["case"], r["seed"]) == (case, seed)
        }
        if set(selected) != {0, 3}:
            raise ValueError("Incomplete pair")
        baseline, candidate = selected[0], selected[3]
        if baseline["input_prepared_sha256"] != candidate["input_prepared_sha256"]:
            raise ValueError("Different original inputs")
        for a, b in zip(baseline["temporal_per_pair"], candidate["temporal_per_pair"]):
            if any(
                a[k] != b[k]
                for k in (
                    "frame",
                    "flow_sha256",
                    "temporal_pixels",
                    "foreground_pixels",
                    "source_warp_mae",
                )
            ):
                raise ValueError("Temporal correspondences differ between variants")
        if len(baseline["temporal_per_pair"]) != len(candidate["temporal_per_pair"]):
            raise ValueError("Different pair counts")
        pair = {"case": case, "seed": seed, "role": baseline["role"]}
        for metric in ("edge_gradient_mae", "temporal_residual_mae"):
            a, b = baseline[metric], candidate[metric]
            pair[metric] = {
                "baseline": a,
                "candidate": b,
                "reduction_percent": (
                    (a - b) / a * 100
                    if a is not None and a > 0 and b is not None
                    else None
                ),
            }
        pairs.append(pair)
    for name in ("results.json", "status.json", "protocol.json", "source-sha256.json"):
        shutil.copy2(run / name, report / name)
    save(report / "paired-effects.json", pairs)
    table = [
        "| 素材 | seed | 边缘误差：0px → 3px | 下降 | 时序残差：0px → 3px | 下降 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair in pairs:
        edge, temporal = pair["edge_gradient_mae"], pair["temporal_residual_mae"]
        table.append(
            f"| {pair['case']} | {pair['seed']} | {edge['baseline']:.6f} → {edge['candidate']:.6f} | {edge['reduction_percent']:.2f}% | {temporal['baseline']:.6f} → {temporal['candidate']:.6f} | {temporal['reduction_percent']:.2f}% |"
        )
    coverage = [
        "| 运行 | 有效对应点 / 当前车身像素 | 有效帧对 |",
        "| --- | --- | --- |",
    ]
    for row in results:
        coverage.append(
            f"| {row['name']} | {row['temporal_coverage']:.2%} | {row['usable_pairs']}/16 |"
        )
    text = "\n".join(
        [
            "# v0.6.0 车身结构与时序诊断",
            "",
            "这是对既有六组输出的探索性补充评估。未新增模型训练或视频生成，不是盲测，也不能视为确认性统计检验。所有配对均报告。",
            "",
            "## 成对结果",
            "",
            *table,
            "",
            "基线为保护半径 0px，候选为 3px。两者控制强度均为 1.0；开发 seed=7/123 与独立素材 seed=42 分开列出，不混合推断泛化。下降为 (基线−候选)/基线。",
            "",
            "## 指标定义",
            "",
            "- 边缘误差：RGB 转亮度后比较归一化 Sobel x/y 梯度，以原始车身内部腐蚀 1px 的区域计算绝对差，避免边界混入新背景。按有效像素数加权汇总。",
            "- 时序残差：仅从原始相邻帧估计双向 Farneback 光流，以当前帧到上一帧的后向光流，用 PyTorch grid_sample 对齐上一帧。计算 mean(abs((输出当前−warp(输出上一帧))−(原始当前−warp(原始上一帧))))。",
            "- 有效点：当前车身、对齐后上一帧车身、有效画布、坐标范围及双向光流一致性同时满足；一致性阈值 1.5px。source_warp_mae 与 output_warp_mae 另行保存。所有方案复用同一原始光流和对应点。",
            "- 无有效点记为 null，不能记为零或解释成完全稳定。报告覆盖率以暴露遮挡、跟踪失败及区域筛选的影响。",
            "",
            "## 对应点覆盖",
            "",
            *coverage,
            "",
            "## 解释边界",
            "",
            "本次样本中 3px 条件的这两项诊断均低于对应基线。这支持继续研究主体保护，但不证明语义身份、感知画质或商业可用性。均匀且跨帧稳定的颜色偏差也可能有零时序残差，因此必须结合空间误差和人工复核。",
            "光流在低纹理、遮挡和快速运动区域可能不准确；一致性筛选会排除部分困难像素。没有人工标注或学习式感知指标校准。车头、轮毂等细节仍可能发生变形。后续生成改进应在新素材上预先固定协议，分别评估主体保真、背景指令遵循和时序质量。",
            "",
            "## 验证与追溯",
            "",
            "30 项测试通过（新增 8 项），包括已知平移、可控闪烁、稳定颜色偏差、背景隔离、缺失对应点与非法输入。固定配置见 protocol.json；源码快照哈希见 source-sha256.json；全部逐帧/帧对值、输出哈希见 results.json。原始光流 npz 和运行源码快照保存在独立 Windows 运行目录。",
            "",
            "运行：在已配置的 Python/PyTorch 环境中执行 python scripts/evaluate_temporal.py；配置中的输出目录必须不存在。报告导出使用 python scripts/export_temporal_report.py。二者都拒绝覆盖已有结果。",
            "",
        ]
    )
    (report / "README.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
