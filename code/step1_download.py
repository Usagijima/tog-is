# -*- coding: utf-8 -*-
"""
第一步：下载数据 + 计算 SHA256（方案 3.md 第 21 节 Day 1-3 任务）

作用：
1. 从 UCI 官方地址下载两份数据集（Online Retail、Bank Marketing）
2. 计算每个文件的 SHA256 指纹并保存到 data/source_manifest.json
   —— 这个指纹就是"数据快照"的身份证，之后所有实验都要记录它，
     保证别人能复现你用的是同一份数据。

用法（在本文件所在目录执行）：
    python step1_download.py
"""

import hashlib
import json
import os
import urllib.request

# ---------- 配置区 ----------
DATA_DIR = "data"  # 下载的文件放在这个文件夹

FILES = {
    # UCI Online Retail 原始数据（Excel 格式），CC BY 4.0 许可
    "online_retail.xlsx": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
        "dataset": "UCI 352 Online Retail",
        "license": "CC BY 4.0",
    },
    # UCI Bank Marketing 原始数据（zip 压缩包，里面含 bank-full.csv 等文件），CC BY 4.0 许可
    "bank.zip": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
        "dataset": "UCI 222 Bank Marketing",
        "license": "CC BY 4.0",
    },
}
# ----------------------------


def sha256_of(path, chunk_size=1 << 20):
    """计算文件的 SHA256 指纹。大文件分块读，不占用太多内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = {"files": {}}

    for filename, info in FILES.items():
        path = os.path.join(DATA_DIR, filename)

        # 已下载过的文件不重复下载（断网重跑也不会浪费时间）
        if os.path.exists(path):
            print(f"[跳过] {filename} 已存在")
        else:
            print(f"[下载中] {filename} ...")
            urllib.request.urlretrieve(info["url"], path)
            print(f"[完成] {filename}")

        digest = sha256_of(path)
        size = os.path.getsize(path)
        manifest["files"][filename] = {
            "dataset": info["dataset"],
            "url": info["url"],
            "license": info["license"],
            "sha256": digest,
            "size_bytes": size,
        }
        print(f"  SHA256: {digest}")
        print(f"  大小: {size / 1024 / 1024:.2f} MB")

    with open(os.path.join(DATA_DIR, "source_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n全部完成，清单已写入 data/source_manifest.json")


if __name__ == "__main__":
    main()
