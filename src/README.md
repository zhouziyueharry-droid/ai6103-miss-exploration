# 后续实现接口

本轮只准备方案和工作区，尚未编写团队训练器或改进算法。下载的原始实现保留在 `third_party/`。

三人先约定以下接口，再实现：

- `data.py`：按模型/数据 revision 载入，固定 train/val 索引，使用 Qwen chat template，response-only labels。返回样本 hash 与截断统计。
- `adapters.py`：统一建立原版 MiSS / LoRA / 改进 MiSS，打印实际参数量与所有 trainable 参数名。
- `train.py`：读取 `configs/` 的团队 JSON schema，明确转换成 MissConfig/LoraConfig/TrainingArguments，拒绝未实现的方法、未固定 revision、重复覆盖 run_dir；默认要求 CUDA。
- `evaluate.py`：统一生成与答案解析，保存逐题预测、gold、正确性、截断及提取失败标记。
- `weighted_miss.py`：C 的候选改进位置，保留上游标准实现不动。

训练入口应记录：原始配置、Git/patch hash、模型/数据/PEFT revision、环境、split hash、训练 seed、loss、step/token 吞吐、峰值 allocated/reserved 显存，以及保存/加载核验。

候选方法的正确性检查比大规模跑分先做：g=1 与原版输出/梯度一致；基础参数无梯度；有限loss；adapter 保存/加载一致；eval 模式合并前后输出一致。没有这些检查，先不解释效果高低。
