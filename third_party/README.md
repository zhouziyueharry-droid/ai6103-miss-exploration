# 固定版本源码

- `miss-author/`：作者公开仓库的完整源码快照，用于理解论文与原始实验脚本。
- `peft/`：Hugging Face PEFT 的完整源码快照，本轮标准 MiSS 与 LoRA 的实现依据。
- `SHA256SUMS.json`：逐文件校验，运行 `python3 scripts/validate_workspace.py` 检查。

版本、原始下载 URL 和压缩包 SHA256 见根目录 `sources.lock.json`。这些目录没有 `.git`，不要在其中执行 git pull；切换版本需要更新来源记录并同步所有对照。

公开仓库不重复托管第三方快照。克隆后运行 `python3 scripts/download_sources.py`，脚本按 lock 中的固定 archive URL 下载、验证和解压，再运行 `python3 scripts/validate_workspace.py`。不能直接使用最新 main。也可在 GPU 机器上复制完整的本地工作文件夹。

完整上游源码保留原有 LICENSE 和版权声明，团队修改另放 `src/` 或独立补丁。
