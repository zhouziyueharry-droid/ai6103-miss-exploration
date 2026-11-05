# 论文与代码核对

核对日期：2026-11-05。以下是源码静态检查，不是运行测试。

## 下载与版本

- 正式会议论文：ICLR 2026，Jiale Kang、Qingyu Yin，*MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure*。
- 正式 PDF：`papers/MiSS_ICLR2026.pdf`。正式会议年份为 2026，方法早期版本发表于更早年份。
- 作者仓库：`Joluck/MiSS`，固定 commit `26bf1affe14e428d8449e4b3d9d6f35d5ba9991e`。
- PEFT：`huggingface/peft`，固定 commit `f482b819536f475da8d4a304cd7a7e37cae59351`，setup.py 版本 `0.21.3.dev0`。
- 完整来源、模型与数据 revision、SHA256 见 `sources.lock.json` 与 `third_party/SHA256SUMS.json`。

固定当前源码是为了三人使用同一版本，**不是声称这是论文实验时使用的原始环境**。后续若改用稳定版本，应作为一次明确的版本变更，全部对照同步。

## 原文模型与任务

| 位置 | 内容 |
|---|---|
| 表 4 | RoBERTa-base，GLUE 子任务 |
| 表 5 | Llama2-7B/13B、RWKV6-7B、Mistral-7B、Qwen3-4B，数学和代码 |
| 表 8 | Llama-3.2-3B 的 PEFT 比较及显存/时间 |
| 附录 A.2 | RWKV7-3B 补充实验 |
| 附录 B | NLU 单张 RTX 4090；数学 MetaMathQA 子集，代码 CodeFeedback 子集 |

Qwen3-1.7B 与 GSM8K train 的本轮方案不是上述表格的直接复现。文中表 5 写三次独立运行，附录 B 的 NLG 段落写两个 seed，记录该不一致，不能悄悄视为同一设置。

## 标准 MiSS 的代码入口

- `third_party/peft/src/peft/tuners/miss/config.py`：MissConfig。
- `third_party/peft/src/peft/tuners/miss/layer.py`：标准 / bat / mini 的 forward、merge。
- `third_party/peft/src/peft/tuners/miss/model.py`：注入和目标模块。
- `third_party/peft/examples/miss_finetuning/`：公开 SFT 示例和说明，作为参考。
- `third_party/peft/docs/source/package_reference/miss.md`：离线文档。

第一轮固定 `init_weights=true`，即标准模式。`bat` 和 `mini` 属已有变体，不能当作团队提出的改进。

## 作者脚本不能直接作为团队试跑入口的原因

1. `scripts/math/run_miss.sh` 包含作者机器的绝对数据与输出路径、镜像 endpoint、DeepSpeed 配置。
2. 该脚本默认选 Qwen3-4B 和 `mini`，不等于本轮标准 MiSS。
3. shell 参数使用 `--init_miss_weight`，Python dataclass 字段为 `init_miss_weights`。即使解析器可能接受缩写，也不能依赖这种行为。
4. batch size 注释与脚本实际单卡参数不一致，需以实际 world_size × micro batch × accumulation 计算。
5. `finetune.py` 使用自定义指令模板及 `<|EOT|>`，不能直接当作 Qwen3 官方 chat template。团队需要统一 EOS、thinking 开关和 response-only mask。
6. PEFT 示例使用 TRL，示例里的 quantization 提示与核心后端能力可能不同，不将量化作为首轮前提。

上游文件保持原样，团队补丁放在 `src/` 或独立 patch 中；不要静默修改第三方快照。

## 授权与引用记录

PEFT 快照自带 Apache-2.0 LICENSE，原样保留。作者快照未找到顶层 LICENSE，不为其补写开源授权声明。团队使用与报告应注明来源；公开分发作者代码前再核对许可。正式论文按学术规范引用。这里的版本与许可记录不阻止本地学习和方案准备。

## 原始来源

- https://proceedings.iclr.cc/paper_files/paper/2026/hash/e365c5fd3e67afdc398519a1de690068-Abstract-Conference.html
- https://github.com/Joluck/MiSS
- https://github.com/huggingface/peft
- https://huggingface.co/Qwen/Qwen3-0.6B
- https://huggingface.co/Qwen/Qwen3-1.7B
- https://huggingface.co/datasets/openai/gsm8k
