# 实验配置模板

这些 JSON 是团队待实现训练器的输入约定，**不能直接传给作者脚本或 PEFT 官方示例**。尚无团队训练入口，状态均为 `plan_only_not_gpu_validated`。

- `smoke_qwen3_06b_miss.json`：仅接通流程，32 个 optimizer steps。
- `pilot_qwen3_17b_miss.json`：原版标准 MiSS，优化的主要 baseline。
- `pilot_qwen3_17b_lora.json`：LoRA 实用基线，参数预算近似而不严格匹配。
- `pilot_qwen3_17b_weighted_miss.json`：待实现的分片加权候选，不能当成现成 PEFT 方法运行。

各配置中的 `seeds` 表示探索拟跑的训练 seed 列表；先跑第一个核查，再决定是否补第二个。`split_seed` 固定且独立于训练 seed。

`gpu_hours_cap_for_direction=8` 是整个 MiSS 探索方向的建议预算，包含所有方法、调参、评测，**不是每个配置各 8 小时**。不构成训练时长保证，也不授权租用 GPU。
