# SceneEdit：基于 Wan VACE 的汽车视频背景编辑

输入汽车视频、逐帧前景掩码和目标场景，在保留车辆外观与运动的同时编辑背景。面向汽车广告素材制作的研究原型，使用 DAVIS 公共研究素材验证。

## 当前结果（v0.7.2）

- 45项自动测试通过；新增两段未用于选参的素材、两种背景，共8次成对CUDA生成已完成。
- 原始固定车身区域的像素MAE下降8.56%–12.09%，固定边界区域误差下降17.78%–40.76%。
- 两份匿名人工评审包均已通过完整性检查，每份4项任务、12个视频资产；真实人工评分待收集。
- 生成队列端到端耗时27.7分钟，峰值PyTorch分配显存4.458GiB。

|新素材|目标背景|主体像素误差下降|边界像素误差下降|
|---|---|---:|---:|
|car-shadow|海岸公路|9.04%|31.13%|
|car-shadow|树林公路|8.56%|40.76%|
|drift-straight|海岸公路|12.07%|17.78%|
|drift-straight|树林公路|12.09%|20.52%|

这里只包含2个独立原视频，提示词是重复测量。像素误差下降不能直接解释为感知质量、背景语义匹配或商业交付质量改善。全部结果、运行口径与局限见[最终验证报告](reports/v0.7.2/README.md)及[原始数值与哈希](reports/v0.7.2/results.json)。

## 生成底座与项目实现

|部分|作用|
|---|---|
|生成底座|固定版本的Wan2.1-VACE-1.3B与DiffSynth-Studio，提供视频条件生成能力|
|数据与条件|按时间戳同步视频/掩码，统一letterbox变换；PyTorch前景膨胀构造0/3px保护条件，参考图保持一致|
|成对评测|固定种子、提示词和控制强度，在原始区域评价所有条件，排除填充区域；不复制原始前景到生成结果|
|结构与时序诊断|PyTorch Sobel梯度误差、原视频双向光流与grid_sample时序残差；历史结果见v0.6.0报告|
|人工评审|隐藏参数名称的离线A/B页面；记录主体、边缘、场景、时序及整体偏好，汇总一致率与Cohen kappa|
|工程实现|BF16权重无损分片与逐张量核验、磁盘到CUDA流式加载、可恢复队列、完整源码/配置/资产哈希|

本项目的改进集中在条件构造、评测与运行工程；基础生成模型的预训练属于上游工作。当前没有Wan/VACE微调结果。

## 实际运行环境

|组件|版本或硬件|
|---|---|
|系统与开发工具|Windows，VS Code Remote SSH|
|Python|3.11.9|
|PyTorch / torchvision|2.6.0+cu124 / 0.21.0|
|CUDA runtime|12.4|
|transformers / accelerate|4.53.2 / 1.10.1|
|safetensors|0.8.0，pread后端|
|GPU|RTX 4060 Laptop 8GB|

具体依赖以锁定文件为准，解释器与调试入口保存在.vscode配置中。CUDA推理与实验均在Windows运行。

## 运行与调试

在项目的VS Code集成终端运行，或选择“SceneEdit: 独立素材验证与匿名评审”调试入口：

~~~powershell
python scripts/validation_workflow.py
~~~

当前默认实验已完成，再次运行会校验既有结果和评审包，不重新生成或覆盖历史。新实验需在configs/independent-validation.json中设置新的输出目录和数据准备目录，可通过cache_root复用原始帧，并保存配置和源码快照。数据、模型、原始视频及完整运行记录均保存在仓库外。

其他入口：scenedit.prepare负责输入准备，scenedit.infer负责CUDA推理，scripts/analyze_blind_review.py负责真实人工评分分析。完整步骤、恢复规则与评分要求见[操作指南](docs/independent-review.zh-CN.md)。

## 验证档位与边界

|档位|规格|用途|
|---|---|---|
|smoke|256×144，5帧，4步|接口检查|
|pilot|448×256，17帧，20步|历史参考图与主体保护实验|
|short-validation|448×256，13帧，20步|本轮短素材独立验证|
|quality|832×480，81帧，50步|大显存设备配置，未运行|

本轮0/3px条件使用同样的13帧输入；不把不同档位的结果合并计算提升。组装帧率显式设为25fps，并不代表恢复了原始拍摄时间。

历史结构与时序指标是探索性诊断，不能替代感知质量或语义一致性指标。新素材的人工评分尚未收集；不报告人工胜率、评审一致率或显著性结论。长视频、大规模泛化与真实商业交付尚未验证。

## 报告与来源

- [v0.7.2：新素材完整结果](reports/v0.7.2/README.md)
- [v0.6.0：结构与时序诊断](reports/v0.6.0/README.md)
- [v0.4.1：历史跨种子验证](reports/v0.4.1/README.md)
- [代码阅读指南](docs/code-guide.zh-CN.md)、[版本记录](VERSION_HISTORY.md)、[变更记录](CHANGELOG.md)

生成引擎：[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)，以Git子模块固定版本，保留Apache-2.0许可证与NOTICE。基础模型：[Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)。数据：[DAVIS 2017](https://davischallenge.org/davis2017/code.html)，遵守原始数据使用条件。精确版本见configs/sources.lock.json。
