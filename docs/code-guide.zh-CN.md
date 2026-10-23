# SceneEdit 代码阅读与调试指南

场景：汽车短视频背景替换。在保留车身的同时，用文本和参考图控制新背景。生成基于 Wan2.1-VACE-1.3B，推理由固定版本的 DiffSynth-Studio 执行。

## 建议阅读顺序

| 文件 | 职责 | 阅读重点 |
| --- | --- | --- |
| `src/scenedit/common.py` | 配置、视频读写辅助、哈希与快照 | 原子 JSON 保存、流式哈希、画布对齐 |
| `src/scenedit/prepare.py` | 输入视频、掩码与参考图预处理 | 帧索引、白色编辑区域、letterbox 元数据 |
| `src/scenedit/preservation.py` | 主体边界保护与固定区域评估 | max-pooling 膨胀、有效像素、原始区域比较 |
| `src/scenedit/checkpoints.py` | 原始与转换权重的身份校验 | 文件来源和校验关系 |
| `src/scenedit/io_backend.py` | safetensors 读取适配 | 应用层 pread 后端，不修改上游代码 |
| `src/scenedit/infer.py` | CUDA 推理与结果留档 | 加载校验、磁盘卸载、状态机、输出校验 |
| `src/scenedit/metrics.py` | 输出指标和帧哈希 | PNG 指标与 MP4 完整性检查 |
| `src/scenedit/robustness.py` | 固定配对统计 | 缺失/重复配对校验、相对变化的零分母 |
| `scripts/run_robustness.py` | 六组验证调度 | 固定协议、逐组日志、评估与配对汇总 |

## 三个必须保持的约定

1. 白色掩码代表允许编辑。保护半径扩大车身保留范围；不会在生成后把原车身贴回输出。
2. 各方案必须使用原始输入的同一组评估区域。letterbox 填充不参与指标；边界环包含背景，误差降低不能直接解释成画质提高。
3. 原始 PNG、MP4、模型标识、输入哈希和源代码快照保留在独立运行目录。新实验创建新目录，不能覆盖历史结果。

## 推理流程和断点

在 `verify_inputs` / `verify_model` 检查输入及模型身份；在 `infer` 的 `offload` 配置处检查 BF16 与磁盘到 CUDA 的调度；在 `pipe(...)` 前检查控制帧、掩码和参考图；在 `evaluate_fixed` 检查固定区域指标。

`status.json` 依次记录 loading、generating、evaluating、complete；异常记录 failed 和 traceback。仅出现 MP4 不代表实验已完成。保留 ClipRank 的 GPU 互斥检查，以免两个项目同时占用显存。

VS Code 工程配置保留专用解释器，并添加 src 和 vendor/DiffSynth-Studio 的代码分析路径。运行环境见主 README；格式工具固定在 requirements-dev.txt 中。

## 本轮维护的边界

v0.5.0 只展开压缩语句、统一排版并增加说明。31 个项目 Python 文件在维护前后 AST 一致；22 项原有测试通过，六组保存输出的逐帧指标复算在 1e-7 绝对容差内一致。此次没有新增训练，也没有重新进行 GPU 生成。验证记录见 reports/v0.5.0。
