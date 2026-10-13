# -*- coding: utf-8 -*-
"""
第三步：数据分割 60/20/20 + 测试集切分为 10 个原始块（方案 3.md 第 6.5 节）

防泄漏的核心思想：同一个人的所有记录必须落在同一个分割里，
否则模型等于"提前见过考题"。

Retail 的分组键：
  - 有 CustomerID 的记录 → 按客户分组
  - 缺 CustomerID 的记录 → 按发票号分组（方案允许：匿名记录无法绑回原购买，
    但必须如实说明，不能假装做了客户级隔离）

Bank 的分组键：
  - 该数据集没有可核实的唯一客户标识，所以按"整行内容指纹"分组，
    保证完全重复的记录一定进同一分割（方案明确：不能声称做了客户级隔离）

测试集再切成 10 个互不重叠的"原始块"（按分组键整块划分），
后续 240 个基础任务就是从这些块里构造的。

输出：
  data/split/retail_{dev,valid,test}.csv.gz      三个分割
  data/split/bank_{dev,valid,test}.csv.gz
  data/split_manifest.json                       分割清单（种子、各块行数、分组数）

用法：
    python step3_split.py
"""

import hashlib
import json
import os

import pandas as pd

DATA_DIR = "data"
SPLIT_DIR = os.path.join(DATA_DIR, "split")
os.makedirs(SPLIT_DIR, exist_ok=True)

SEED = 17            # 冻结的随机种子，写进论文
N_TEST_BLOCKS = 10   # 每个数据集测试集切成 10 个原始块
RATIOS = (0.60, 0.20, 0.20)


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def assign_splits(group_keys, seed, ratios=RATIOS):
    """把分组键随机切成 dev/valid/test 三份，返回 {group_key: split_name}。"""
    keys = sorted(set(group_keys))
    shuffled = pd.Series(keys).sample(frac=1.0, random_state=seed).tolist()
    n = len(shuffled)
    n_dev = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    assign = {}
    for i, k in enumerate(shuffled):
        assign[k] = "dev" if i < n_dev else ("valid" if i < n_dev + n_valid else "test")
    return assign


def split_test_into_blocks(test_df, group_col, seed, n_blocks=N_TEST_BLOCKS):
    """测试集按分组键整块切成 n_blocks 个互不重叠的块。"""
    keys = sorted(test_df[group_col].unique())
    shuffled = pd.Series(keys).sample(frac=1.0, random_state=seed + 1).tolist()
    block_of = {}
    for i, k in enumerate(shuffled):
        block_of[k] = i % n_blocks  # 轮转分配，保证各块大小均匀
    return test_df[group_col].map(block_of)


def save_splits(df, split_col, name, manifest):
    """按分割标签存盘，并把统计写进清单。"""
    for split in ["dev", "valid", "test"]:
        part = df[df[split_col] == split]
        path = os.path.join(SPLIT_DIR, f"{name}_{split}.csv.gz")
        part.to_csv(path, index=False, compression="gzip")
        manifest[f"{name}_{split}"] = {
            "rows": len(part),
            "groups": int(part["_group_key"].nunique()),
            "file": os.path.basename(path),
        }
        print(f"[{name}] {split}: {len(part)} 行 / {part['_group_key'].nunique()} 组")


def main():
    manifest = {"seed": SEED, "ratios": list(RATIOS), "test_blocks_per_dataset": N_TEST_BLOCKS}

    # ============ Retail ============
    print("[Retail] 读取清洗数据 ...")
    df = pd.read_csv(os.path.join(DATA_DIR, "clean", "retail_clean.csv.gz"))
    # 分组键：有客户 ID 按客户，没有按发票
    df["_group_key"] = df["CustomerID"].fillna("INV_" + df["InvoiceNo_str"].astype(str)).astype(str)

    assign = assign_splits(df["_group_key"], SEED)
    df["_split"] = df["_group_key"].map(assign)
    print(f"[Retail] 无客户 ID 的记录按发票分组：{int(df['CustomerID'].isna().sum())} 行")

    test_mask = df["_split"] == "test"
    df.loc[test_mask, "_block"] = split_test_into_blocks(df[test_mask], "_group_key", SEED)
    df["_block"] = df["_block"].fillna(-1).astype(int)

    save_splits(df, "_split", "retail", manifest)
    # 保存测试块的划分表（块 -> 分组键），任务编译器第 4-7 天要用
    blocks = df[df["_split"] == "test"][["_group_key", "_block"]].drop_duplicates()
    blocks.to_csv(os.path.join(SPLIT_DIR, "retail_test_blocks.csv.gz"), index=False, compression="gzip")
    manifest["retail_test_blocks"] = blocks["_block"].value_counts().sort_index().to_dict()
    print(f"[Retail] 测试块行数分布: {dict(blocks['_block'].value_counts().sort_index())}")

    # ============ Bank ============
    print("\n[Bank] 读取清洗数据 ...")
    bk = pd.read_csv(os.path.join(DATA_DIR, "clean", "bank_full_clean.csv.gz"))
    # 分组键：整行内容指纹（完全重复的行一定同组）
    bk["_group_key"] = pd.util.hash_pandas_object(bk, index=False).astype(str)

    assign = assign_splits(bk["_group_key"], SEED)
    bk["_split"] = bk["_group_key"].map(assign)
    print(f"[Bank] 注意：按行指纹分组，未声称客户级隔离（方案 6.5 要求如实说明）")

    test_mask = bk["_split"] == "test"
    bk.loc[test_mask, "_block"] = split_test_into_blocks(bk[test_mask], "_group_key", SEED)
    bk["_block"] = bk["_block"].fillna(-1).astype(int)

    save_splits(bk, "_split", "bank", manifest)
    bblocks = bk[bk["_split"] == "test"][["_group_key", "_block"]].drop_duplicates()
    bblocks.to_csv(os.path.join(SPLIT_DIR, "bank_test_blocks.csv.gz"), index=False, compression="gzip")
    manifest["bank_test_blocks"] = bblocks["_block"].value_counts().sort_index().to_dict()
    print(f"[Bank] 测试块行数分布: {dict(bblocks['_block'].value_counts().sort_index())}")

    # ============ 记录数据快照指纹 ============
    manifest["input_files"] = {
        "retail_clean.csv.gz": sha256_of(os.path.join(DATA_DIR, "clean", "retail_clean.csv.gz")),
        "bank_full_clean.csv.gz": sha256_of(os.path.join(DATA_DIR, "clean", "bank_full_clean.csv.gz")),
    }

    with open(os.path.join(DATA_DIR, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("\n分割完成，清单已写入 data/split_manifest.json")


if __name__ == "__main__":
    main()
