# A 股数据发布与安装

全量 A 股原始 TXT 与转换后的 OHLCV CSV 不提交到 Git。它们作为单个压缩包发布到 GitHub Release 或对象存储，避免普通 Git 推送卡住或超过平台限制。

## 发布方

在拥有 `data/a_share_ohlcv/` 的机器上执行：

```bash
./.venv/bin/python scripts/package_a_share_data.py
```

该命令生成 `artifacts/a_share_ohlcv.tar.gz` 并打印 SHA-256。将压缩包上传为 GitHub Release asset，记录其直接下载链接与 SHA-256。不要将 `artifacts/` 或数据目录加入 Git。

## 使用方或 Streamlit 部署准备

下载并校验后安装：

```bash
./.venv/bin/python scripts/download_a_share_data.py \
  --url 'https://github.com/<owner>/<repo>/releases/download/<tag>/a_share_ohlcv.tar.gz' \
  --sha256 '<发布时打印的 SHA-256>'
```

安装后数据位于 `data/a_share_ohlcv/`，UI 会自动出现“全部 A 股”数据范围。若要替换已有数据，追加 `--replace`。
