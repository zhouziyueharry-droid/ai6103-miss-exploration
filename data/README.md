# 数据位置

当前只有 `docs/model_cards/GSM8K.md` 数据说明，未下载完整数据。

GPU 机器下载固定 revision 的 `openai/gsm8k`，config `main`。原始数据和 tokenized 缓存不提交 Git。

后续将固定划分保存到 `data/splits/`，至少记录 original_index、split、text_sha256。所有方法和训练 seed 共用划分。官方 test 不参与前期选方法或调参。
