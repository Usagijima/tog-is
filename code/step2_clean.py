# -*- coding: utf-8 -*-
"""
第二步：清洗数据 + 生成清洗报告（方案 3.md 第 23 节的约定）

这份脚本严格遵守方案里的清洗规则，重点有四条"不许做"：
  1. Retail 不许删除缺失 CustomerID 的行 —— "客户可识别率"本身就是评测指标，
     删了就等于把指标的数据基础毁掉了。
  2. 不许人工猜测修正任何"看起来矛盾"的记录（比如取消标记和数量符号不一致），
     只做隔离 + 在报告里计数。
  3. 金额统一转成整数便士，避免二进制浮点误差。
  4. 重复记录不许擅自解释为"重复交易"，按固定规则保留并记录数量。

输出：
  data/clean/retail_clean.csv.gz       清洗后的 Retail 数据
  data/clean/bank_full_clean.csv.gz    清洗后的 Bank 数据
  data/cleaning_report.json           清洗报告（每种被处理/隔离的记录数量）

用法：
    python step2_clean.py
"""

import json
import os
import zipfile

import pandas as pd

DATA_DIR = "data"
OUT_DIR = os.path.join(DATA_DIR, "clean")
os.makedirs(OUT_DIR, exist_ok=True)

report = {"retail": {}, "bank": {}}


# ============================================================
# 一、Online Retail 清洗
# ============================================================
def clean_retail():
    print("[Retail] 读取原始 Excel ...")
    df = pd.read_excel(os.path.join(DATA_DIR, "online_retail.xlsx"))
    report["retail"]["raw_rows"] = len(df)

    # --- 1. 隔离"数量为零"的记录（规则：隔离并报告，不删除）---
    qty_zero = df["Quantity"] == 0
    report["retail"]["isolated_quantity_zero"] = int(qty_zero.sum())
    df = df[~qty_zero]

    # --- 2. 隔离"单价不为正"的记录 ---
    price_bad = df["UnitPrice"] <= 0
    report["retail"]["isolated_nonpositive_price"] = int(price_bad.sum())
    df = df[~price_bad]

    # --- 3. 隔离"关键字段无法解析"的记录（时间为空等）---
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    time_bad = df["InvoiceDate"].isna()
    report["retail"]["isolated_unparseable_fields"] = int(time_bad.sum())
    df = df[~time_bad]

    # --- 4. 金额转成整数便士 ---
    # 原始单价单位是英镑（小数），乘以 100 变便士再四舍五入成整数，
    # 后续所有金额计算都用整数，彻底避开浮点误差。
    df["UnitPricePence"] = (df["UnitPrice"] * 100).round().astype("int64")

    # --- 5. 取消标记审计 ---
    # InvoiceNo 以 "C" 开头的通常是取消单。方案要求：以数量符号为准计算金额，
    # "C" 前缀只作为一致性审计字段 —— 两者不一致的记录不修正，只计数报告。
    df["InvoiceNo_str"] = df["InvoiceNo"].astype(str)
    df["is_cancel_invoice"] = df["InvoiceNo_str"].str.startswith("C")
    df["is_negative_qty"] = df["Quantity"] < 0
    mismatch = df["is_cancel_invoice"] != df["is_negative_qty"]
    report["retail"]["cancel_flag_mismatch_rows"] = int(mismatch.sum())
    # 两类不一致各有多少，写进报告方便写数据卡时引用
    report["retail"]["mismatch_cancel_flag_only"] = int((df["is_cancel_invoice"] & ~df["is_negative_qty"]).sum())
    report["retail"]["mismatch_negative_qty_only"] = int((~df["is_cancel_invoice"] & df["is_negative_qty"]).sum())

    # --- 6. 完全重复的记录：保留（不解释为重复交易），只计数 ---
    dup_cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPricePence", "CustomerID", "Country"]
    n_dup = int(df.duplicated(subset=dup_cols).sum())
    report["retail"]["exact_duplicate_rows_kept"] = n_dup

    # --- 7. 缺失 CustomerID 的行保留！只统计数量 ---
    report["retail"]["rows_missing_customer_id"] = int(df["CustomerID"].isna().sum())

    report["retail"]["clean_rows"] = len(df)
    df.to_csv(os.path.join(OUT_DIR, "retail_clean.csv.gz"), index=False, compression="gzip")
    print(f"[Retail] 原始 {report['retail']['raw_rows']} 行 -> 清洗后 {len(df)} 行")


# ============================================================
# 二、Bank Marketing 清洗
# ============================================================
def clean_bank():
    print("[Bank] 解压并读取 bank-full.csv ...")
    zip_path = os.path.join(DATA_DIR, "bank.zip")
    with zipfile.ZipFile(zip_path) as z:
        z.extract("bank-full.csv", path=DATA_DIR)

    df = pd.read_csv(os.path.join(DATA_DIR, "bank-full.csv"), sep=";")
    report["bank"]["raw_rows"] = len(df)
    report["bank"]["raw_columns"] = len(df.columns)

    # --- 1. y 必须是合法历史标签，无法解析的隔离 ---
    valid_labels = {"yes", "no"}
    bad_label = ~df["y"].isin(valid_labels)
    report["bank"]["isolated_invalid_label"] = int(bad_label.sum())
    df = df[~bad_label]

    # --- 2. 关键数值字段无法解析的隔离 ---
    for col in ["age", "balance", "duration", "campaign", "pdays", "previous"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    num_bad = df[["age", "balance", "duration", "campaign", "pdays", "previous"]].isna().any(axis=1)
    report["bank"]["isolated_unparseable_numeric"] = int(num_bad.sum())
    df = df[~num_bad]

    # --- 3. unknown 类别保留（方案明确：job/education/contact 里的 unknown
    #        有业务含义，不许当成脏数据删掉）---
    for col in ["job", "education", "contact"]:
        report["bank"][f"unknown_count_{col}"] = int((df[col] == "unknown").sum())

    # --- 4. 完全重复的行：归到同一组（供后续分割防泄漏用），不删除 ---
    n_dup = int(df.duplicated(keep=False).sum())
    report["bank"]["exact_duplicate_rows_grouped"] = n_dup

    report["bank"]["clean_rows"] = len(df)
    df.to_csv(os.path.join(OUT_DIR, "bank_full_clean.csv.gz"), index=False, compression="gzip")
    print(f"[Bank] 原始 {report['bank']['raw_rows']} 行 -> 清洗后 {len(df)} 行")


def main():
    clean_retail()
    clean_bank()
    with open(os.path.join(DATA_DIR, "cleaning_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n清洗报告已写入 data/cleaning_report.json")


if __name__ == "__main__":
    main()
