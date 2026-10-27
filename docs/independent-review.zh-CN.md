# 独立素材验证与匿名人工评审

本轮解决两件事：把主体保护参数放到未用于选参的素材上检验；检查像素诊断与真实观看判断是否一致。它不包含 Wan/VACE 训练，也不把匿名评审工具的完成算成人工评测已经完成。

## 固定实验

- 选参仍来自历史 car-roundabout；本轮固定 radius=0 与 radius=3，VACE scale=1，不根据新结果修改参数。
- 新素材为 DAVIS 的 car-shadow 与 drift-straight，两个目标背景为海岸公路、树林公路，seed=23，2×2×1×2=8 次生成。
- 短素材使用独立 short-validation 档位：448×256、13帧、8fps、20步。原 pilot 的17帧配置保持不变。本轮只进行相同档位内的成对比较。
- 首次准备沿用17帧档位，因 car-shadow 长度不足失败。日志保留在 data/SceneEdit/independent-v1；改正后的准备与实验使用独立 v3 目录，已下载原始帧经CRC校验后复用。
- 两个原视频才是两个独立来源；多个提示词、种子和视频帧是重复测量，不能当作更多独立样本计算显著性。

## 运行与恢复

在 SceneEdit 的 VS Code 集成终端使用项目已配置的 Python 解释器运行：

~~~powershell
python scripts/validation_workflow.py
~~~

源码与配置先保存并计算哈希，推理在 CUDA 子进程中依次运行，避免多个模型争抢显存。日志、配置、实际环境、模型标识和代码快照保存在仓库外的实验目录。

~~~text
D:/career-projects/runs/SceneEdit/20261027-independent-v3
~~~

workflow-status.json 是整个流程状态，status.json 是GPU队列状态。每个任务有单独日志。正常失败的再次运行会校验并复用完整成功结果，失败尝试保留；源码、配置或模型身份变化时拒绝在旧实验目录恢复。active.lock 防止重复启动；若Windows强制终止导致锁残留，先核对记录的PID已经退出再人工处理，不能直接启动第二份任务。

## 人工评审

流程会生成 reviewer-1 和 reviewer-2 两个文件夹。分别打开各自的 index.html，可离线查看原视频、目标描述与匿名 A/B 结果；左右位置平衡、任务顺序固定随机，文件名不暴露保护参数。

五个维度：主体外观与身份、轮廓边缘、背景指令匹配、运动及跨帧稳定、整体效果。每项选择 A更好、B更好、相当、均不合格；无默认评分，必须全部填写后才能导出。

至少邀请两位真实、独立评分者，使用不同代号，在讨论前各自完成评分。不要把同一人的两次打分当作两位评审者，不要用模型评分冒充人工评分。程序不能核实评分者是否为独立真人，需要由实验组织者保证。

只把 reviewer-N 文件夹交给相应评分者；同级的 reviewer-N-private-key.json 是答案映射，保留在组织者本机，不随评审包共享。原始数据、视频和评分文件不提交Git。

~~~powershell
python scripts/check_blind_review.py D:/career-projects/runs/SceneEdit/20261027-independent-v3/reviewer-1
python scripts/analyze_blind_review.py --review D:/career-projects/runs/SceneEdit/20261027-independent-v3/reviewer-1-private-key.json ratings-1.json --review D:/career-projects/runs/SceneEdit/20261027-independent-v3/reviewer-2-private-key.json ratings-2.json --output D:/career-projects/runs/SceneEdit/20261027-independent-v3/human-review.json
~~~

分析程序拒绝缺项、重复任务、错误协议及重复评分者代号，报告各维度偏好计数、相当与均不合格计数，以及两两一致率和Cohen kappa。只有一份真实评分时明确标记 single_reviewer_only；无法定义的kappa返回null。

同时逐任务并列人工主体/边缘判断和对应像素诊断方向，保留不一致案例。两段素材不足以证明指标普遍有效，也不能由小幅像素误差下降推断商业交付质量、语义一致性或广泛泛化。

## JD对应能力

这次补充体现生成编辑条件控制、数据与评测协议设计、可恢复的CUDA推理、盲评组织、指标局限分析。它更直接支持即梦的主体/时序评价与火山方舟的可复现人工评测要求。生成模型微调、奖励建模和Agent编排仍然是未完成方向，不能由本轮评测替代。

数据清单更正：初版第二段误写为不存在的car-drift，准备过程在任何GPU推理前停止。现已直接核对固定压缩包目录，car-shadow为40帧、drift-straight为50帧；更正清单使用v3目录，保留v1/v2失败日志，并增加全量素材/掩码预检查。

## 本轮完成状态（v0.7.2）

8次生成已完成，4对结果全部保留。两份评审包均通过4任务/12视频的哈希及页面清单检查；真实人工评分待填写。再次运行VS Code调试入口将只读校验既有结果与评审包，实验status.json哈希在复查前后保持一致。新实验需要新的output与data_root，使用cache_root复用已有原始帧；不可覆盖本轮目录。
