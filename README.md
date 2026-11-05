# MiSS 前期可行性探索

六人小组最终交一个项目，目前分成两组探索两个候选方向。本文件夹服务于其中三人的 **MiSS 前期探索**，目标是判断是否能得到可信的优化结果，不是三人承担整个最终项目。

当前状态：已有方案、配置模板和固定来源记录；本地工作区已下载正式论文与源码。公开仓库通过下面的命令恢复这些资料。**未下载模型权重或完整数据，未运行训练，也没有优化收益结论。**

## 克隆后的准备

```bash
python3 scripts/download_sources.py
python3 scripts/validate_workspace.py
```

第一条只从官方来源下载固定版本的 MiSS / PEFT 源码和论文，并检查 SHA256，不安装依赖、不下载模型权重、不运行训练。若已有源码被修改，脚本会停止而不覆盖。下载需要网络。原论文也可从 [论文入口](papers/README.md) 直接打开。

## 先读这三个文件

1. [实验方案与决策门槛](docs/EXPLORATION_PLAN.md)：baseline、任务、数据、模型、优化方向和三人前期分工。
2. [GPU 机器准备与执行顺序](docs/GPU_RUNBOOK.md)：本机可以做什么、GPU 上先做什么。
3. [论文与代码核对](docs/SOURCE_AUDIT.md)：源码版本、原论文模型，以及不能直接照抄的启动脚本。

## 本轮建议

- 流程检查：Qwen3-0.6B + GSM8K 少量训练样本，标准 MiSS，32 个 optimizer steps。
- 探索主实验：Qwen3-1.7B + GSM8K，非 thinking 模式的监督微调（SFT）。
- 核心比较：同一模型上的原版 MiSS 与候选改进。未微调模型提供任务起点，LoRA 提供实用基线。
- 优化先试：标准 MiSS 的每分片可学习缩放。仅为候选，先查重、实现及验证，不能预先承诺有效。
- 最后给六人组的交付：短报告、可复跑配置、原始日志、收益与成本、继续或放弃的建议。

## 文件夹

```text
papers/             正式 ICLR 2026 论文、参考文献
third_party/        固定 commit 的作者和 PEFT 原始源码
docs/               探索方案、GPU 手册、源码审阅、模型/数据说明
configs/            试跑与探索配置模板（不是上游 CLI 的直接输入）
scripts/            不训练模型的工作区校验和环境检查
src/                团队后续训练入口及改进代码的位置
data/               数据下载、划分清单的位置
results/            实验记录模板；当前没有实测结果
reports/            已有 MiSS 讨论 PPT
archive/            旧选题资料（仅本地，不发布）
sources.lock.json   下载来源、版本号与校验信息
```

## Mac 上的安全检查

在本文件夹打开终端：

```bash
python3 scripts/validate_workspace.py
python3 scripts/check_environment.py
```

以上不下载权重、不执行模型前向或训练。训练依赖、CUDA 及环境锁定只在 GPU 机器上处理。

`configs/` 目前是团队约定的实验规格，不是已经实现的训练器。不要把它们直接传给作者的 `finetune.py`。需要补完的训练/评测接口见 `src/README.md`。

旧 PPT 中的“分片加权”是未验证设想；其中完整课程项目的分工不再适用，以本次前期探索方案为准。
