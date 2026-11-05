# GPU 机器准备与执行顺序

当前交付是前期方案与代码资料，不是已验收的训练系统。团队训练入口、数据处理和评测器尚待 A/B/C 按 `src/README.md` 实现。此文不提供虚构的“已经跑通”命令。

## Mac 阶段（本轮已执行）

下载公开源码、论文、模型/数据说明和结构配置；固定来源版本；离线校验文件。没有安装 torch/CUDA 训练栈，没有下载完整权重与训练数据，没有启动上游训练脚本。

## GPU 环境（交接后执行）

从 GitHub 克隆后，先执行 `python3 scripts/download_sources.py` 恢复第三方源码与论文，再执行 `python3 scripts/validate_workspace.py`。公开仓库不包含本地 archive 或完整第三方快照。

建议 Linux 或 Windows 的 WSL2 + NVIDIA GPU，Python 3.11/3.12，独立虚拟环境。当前固定 PEFT 源码要求 Python >=3.11。

1. 用 `nvidia-smi` 确认卡型号、显存、驱动和可用时间。H100 的具体显存容量需要现场确认。
2. 创建虚拟环境。根据该机器的驱动和 GPU 架构，从 PyTorch 官方安装页选择 CUDA wheel。5060 Ti 属较新架构，不能仅凭“CUDA 可用”推定 wheel/kernels 兼容。
3. 确认一次小张量 CUDA 运算成功后，安装固定的本地 PEFT 源码和所需上层库。

```bash
python3 -m venv .venv
source .venv/bin/activate
# 在此按 https://pytorch.org/get-started/locally/ 安装适合本机的 torch
python -m pip install -r requirements-gpu.in
python scripts/check_environment.py --require-cuda --output results/environment.json
python -m pip check
python -m pip freeze > results/requirements-gpu.lock.txt
```

`requirements-gpu.in` 是待 GPU 验证的依赖入口，不是锁定并验证过的兼容组合。首台 GPU 环境验收后记录 wheel 来源、Python/驱动/CUDA 版本、PEFT commit 和 freeze，其他人复用。不要在三台机器各自随意升级。

第一轮使用 SDPA，暂不需要 DeepSpeed、FlashAttention、bitsandbytes 或在线实验平台。作者脚本的这些额外依赖不作为试跑前提。PEFT 官方 SFT 示例额外依赖 TRL，但团队入口可用 Transformers Trainer，不强制引入 TRL。

## 下载训练资产

模型和数据都使用 `sources.lock.json` 中的 revision。在 GPU 机器的 HF 缓存下载，不提交到团队 Git。需要提前拷贝时也用相同 revision。

模型：先 Qwen3-0.6B，流程通过后才下载 Qwen3-1.7B。数据：`load_dataset('openai/gsm8k', 'main', revision=...)`。不要把仅包含 config.json 的 `docs/model_cards/` 误当作完整模型目录。

## 执行闸门

1. 静态检查 `python scripts/validate_workspace.py`。
2. 环境检查；导入 `MissConfig`、`LoraConfig`、模型与 tokenizer 类。
3. B 生成固定划分和 token mask 审阅样例。
4. A 按 `configs/smoke_qwen3_06b_miss.json` 接通单卡试跑。
5. A/B 共同验收加载、adapter-only 更新、loss/标签、保存和评测。
6. 按探索配置测原版 MiSS 和 LoRA；C 实现候选改进后复用同一入口。
7. 汇总到 `results/experiment_registry.csv` 和 `results/exploration_report.md`。

## 显存不足时

首先确认没有遗留训练进程，micro batch=1，gradient checkpointing 已生效，训练时 use_cache=False；再减序列长度并同时更新全部对照。通过梯度累积保持有效 batch。

先保留原版标准 MiSS、BF16 和一致实验协议。不要只给一个方法做量化或缩短上下文后比较性能。如 1.7B 仍不合适，先用 0.6B 验证实现或转 H100，并明确变更后的研究范围。

## 目前明确未完成

- GPU 依赖兼容性、真实显存与吞吐。
- 团队训练/评测程序以及候选分片加权实现。
- 方法的正向收益、统计显著性与最终课程选题认可。
