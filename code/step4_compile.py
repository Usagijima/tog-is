# -*- coding: utf-8 -*-
"""
第四步：任务编译器（方案 3.md 第 5、6 节）

把每个测试块编译成 12 个基础任务：
    2 种行动规模（选1个群组 / 选2个群组） × 6 组权重 = 12 任务
    每数据集 10 块 × 12 = 120 任务，两数据集共 240 个。

每个任务的原料：
  - 8 个候选群组（每个 ≥30 条记录，Retail 还要求正向金额非零）
  - 每个群组 3 项指标，归一化到 [0,1]，再量化到 0.0001 网格（整数存储）
  - 6 组权重 = (0.5, 0.3, 0.2) 的全部排列
  - 评测专用的真实最优行动（离线评测器用，绝不上线）

关键规则（方案明确要求）：
  - 价格区间 / 余额区间的分位点只能用【开发集】冻结，不许看测试集
  - 归一化尺度 sG / sD 只能用【开发集】确定
  - 群组不够 8 个时用预注册的更粗模板兜底，再不够就标记"不可构造"
  - 不许根据哪个群组"题目好做"来挑选

输出：
  data/episodes.jsonl    240 个任务的完整定义（含离线真值，注意保密管理）
  data/data_card.json    数据卡：尺度参数、区间边界、兜底使用情况

用法：
    python step4_compile.py
"""

import itertools
import json
import os

import pandas as pd

DATA_DIR = "data"
SPLIT_DIR = os.path.join(DATA_DIR, "split")
SEED = 17
MIN_ROWS = 30            # 每个群组最少记录数（工程设置，不是差分隐私）
QUANT = 10000            # 量化到 1e-4 网格，用整数存储

# 6 组权重：(0.5, 0.3, 0.2) 的全部排列，顺序冻结
WEIGHTS = sorted(set(itertools.permutations([0.5, 0.3, 0.2])))

card = {"seed": SEED, "min_rows_per_cohort": MIN_ROWS, "quantization": 1 / QUANT}


# ---------- 通用小工具 ----------
def to_int(z):
    """归一化值量化到 0.0001 网格，返回整数。任务状态就以这个整数为准。"""
    return int(round(z * QUANT))


def clip01(x):
    return min(1.0, max(0.0, x))


# ---------- Retail 群组模板 ----------
def retail_cohorts(block_df, price_bands):
    """按冻结模板生成 8 个群组：UK/非UK × 4 个开发集价格区间。

    返回 (群组列表, 是否用了兜底)。群组不足 8 个合法时，
    用兜底模板：UK/非UK × 2 个粗区间（4个）+ 量最大的 4 个国家（4个）。
    """
    df = block_df
    uk = df["Country"] == "United Kingdom"

    def make(name, mask):
        sub = df[mask]
        if len(sub) >= MIN_ROWS and (sub["Quantity"] > 0).any():
            return {"name": name, "mask": sub}
        return None

    cohorts = []
    for is_uk, label in [(True, "UK"), (False, "nonUK")]:
        base = uk if is_uk else ~uk
        for bi in range(4):
            lo, hi = price_bands[bi], price_bands[bi + 1]
            m = base & (df["UnitPricePence"] >= lo) & (df["UnitPricePence"] < hi)
            c = make(f"{label}_priceband{bi}", m)
            if c:
                cohorts.append(c)
    if len(cohorts) >= 8:
        return cohorts[:8], False

    # 兜底模板
    cohorts = []
    for is_uk, label in [(True, "UK"), (False, "nonUK")]:
        base = uk if is_uk else ~uk
        for bi in (0, 2):  # 用更粗的 2 个区间
            lo, hi = price_bands[bi], price_bands[bi + 2]
            m = base & (df["UnitPricePence"] >= lo) & (df["UnitPricePence"] < hi)
            c = make(f"{label}_coarseband{bi}", m)
            if c:
                cohorts.append(c)
    top_countries = df["Country"].value_counts().head(4).index
    for country in top_countries:
        c = make(f"country_{country}", df["Country"] == country)
        if c:
            cohorts.append(c)
    return cohorts[:8], True


def retail_metrics(sub, sG):
    """Retail 三项指标（方案 6.2 节）。金额为整数便士运算。"""
    pos = sub[(sub["Quantity"] > 0)]
    G = int((pos["Quantity"] * pos["UnitPricePence"]).sum())   # 正向交易总金额
    neg = sub[sub["Quantity"] < 0]
    C = int((-neg["Quantity"] * neg["UnitPricePence"]).sum())  # 取消/负向金额
    n_plus = pos["InvoiceNo_str"].nunique()                    # 不同发票数
    z1 = clip01(G / (n_plus * sG)) if n_plus > 0 else 0.0
    z2 = 1 - clip01(C / G) if G > 0 else 0.0
    z3 = float(sub["CustomerID"].notna().mean())               # 客户可识别率
    return z1, z2, z3, {"G": G, "C": C, "n_plus": n_plus, "rows": len(sub)}


# ---------- Bank 群组模板 ----------
def bank_cohorts(block_df, job_top4, balance_bands):
    """主模板（单字段分组，方案 6.4 节 DSL 示例）：4 个教育类别 + 开发集
    冻结的 4 个最常见职业 = 8 个群组。

    兜底模板：4 个教育类别 + 开发集分位点冻结的 4 个余额区间 = 8 个群组。
    单字段群组每个约 100-500 行，主模板几乎总能满足 ≥30 行的要求。
    """
    df = block_df
    edu_levels = ["primary", "secondary", "tertiary", "unknown"]

    cohorts = []
    for edu in edu_levels:
        sub = df[df["education"] == edu]
        if len(sub) >= MIN_ROWS:
            cohorts.append({"name": f"edu_{edu}", "mask": sub})
    for job in job_top4:
        sub = df[df["job"] == job]
        if len(sub) >= MIN_ROWS:
            cohorts.append({"name": f"job_{job}", "mask": sub})
    if len(cohorts) >= 8:
        return cohorts[:8], False

    # 兜底：教育 4 个 + 余额区间 4 个
    cohorts = []
    for edu in edu_levels:
        sub = df[df["education"] == edu]
        if len(sub) >= MIN_ROWS:
            cohorts.append({"name": f"edu_{edu}", "mask": sub})
    for bi in range(4):
        lo, hi = balance_bands[bi], balance_bands[bi + 1]
        sub = df[(df["balance"] >= lo) & (df["balance"] < hi)]
        if len(sub) >= MIN_ROWS:
            cohorts.append({"name": f"balanceband{bi}", "mask": sub})
    return cohorts[:8], True


def bank_metrics(sub, sD):
    """Bank 三项指标（方案 6.3 节）。"""
    n = len(sub)
    z1 = float((sub["y"] == "yes").mean())                     # 历史转化率
    z2 = 1 - clip01(sub["duration"].sum() / (n * sD))          # 联系时长效率
    unknown = 0
    for col in ["job", "education", "contact"]:
        unknown += int((sub[col] == "unknown").sum())
    z3 = 1 - unknown / (n * 3)                                 # 字段可观察性
    return z1, z2, z3, {"rows": n}


# ---------- 行动与真值 ----------
def best_action(z_int, weights, h):
    """枚举全部合法行动，返回 (最优行动索引列表, 最优评分)。z 用量化整数值。"""
    K = 8
    combos = list(itertools.combinations(range(K), h))
    best_score, best = -1.0, []
    for a in combos:
        # 每个行动的组织评分 = (1/h) * Σ_{k∈a} Σ_i w_i * z_ki
        score = sum(sum(weights[i] * z_int[k * 3 + i] for i in range(3)) for k in a)
        score /= h * QUANT  # 转回 [0,1] 便于阅读
        if score > best_score + 1e-12:
            best_score, best = score, [a]
        elif abs(score - best_score) <= 1e-12:
            best.append(a)
    return best, best_score


def compile_dataset(name, load_fn, cohort_fn, metric_fn, scale_key, scale_val):
    episodes = []
    df = load_fn()
    df = df.drop(columns=["_block"], errors="ignore")  # 避免与块表合并时列名冲突
    blocks = pd.read_csv(os.path.join(SPLIT_DIR, f"{name}_test_blocks.csv.gz"))
    df = df.merge(blocks, on="_group_key", how="inner")
    card[f"{name}_scale"] = scale_val

    n_fallback = 0
    n_unconstructable = 0
    for b in sorted(df["_block"].unique()):
        bdf = df[df["_block"] == b]
        cohorts, used_fallback = cohort_fn(bdf)
        if len(cohorts) < 8:
            n_unconstructable += 1
            print(f"[{name}] 块 {b}: 只有 {len(cohorts)} 个合法群组，标记不可构造")
            continue
        n_fallback += int(used_fallback)

        # 计算 8 群组 × 3 指标
        z_int, cohort_info = [], []
        for c in cohorts:
            z1, z2, z3, detail = metric_fn(c["mask"], scale_val)
            z_int += [to_int(z1), to_int(z2), to_int(z3)]
            cohort_info.append({"name": c["name"], **detail})

        for h in (1, 2):
            for wi, w in enumerate(WEIGHTS):
                best, best_score = best_action(z_int, w, h)
                episodes.append({
                    "episode_id": f"{name}_block{int(b):02d}_h{h}_w{wi}",
                    "dataset": name, "block": int(b), "h": h, "weights": list(w),
                    "cohorts": cohort_info,
                    "z_int": z_int,                       # 24 个整数事实（任务状态）
                    "z": [v / QUANT for v in z_int],      # 还原成 [0,1] 便于人读
                    "n_actions": 8 if h == 1 else 28,
                    "eval_only_optimal_actions": best,    # 离线真值，绝不上线
                    "eval_only_optimal_score": best_score,
                    "used_fallback_cohorts": used_fallback,
                })
    card[f"{name}_blocks_used"] = len(set(e["block"] for e in episodes)) // 12 or len(set(e["block"] for e in episodes))
    card[f"{name}_fallback_blocks"] = n_fallback
    card[f"{name}_unconstructable_blocks"] = n_unconstructable
    return episodes


def main():
    # ---- 用【开发集】冻结尺度参数和区间（绝不看测试集）----
    print("用开发集冻结尺度参数 ...")
    rdev = pd.read_csv(os.path.join(SPLIT_DIR, "retail_dev.csv.gz"))
    rpos = rdev[rdev["Quantity"] > 0]
    avg_invoice = (rpos["Quantity"] * rpos["UnitPricePence"]).sum() / max(1, rpos["InvoiceNo_str"].nunique())
    sG = float(avg_invoice)  # 开发集平均正向发票金额（便士）
    price_bands = [0] + sorted(pd.unique(pd.qcut(rpos["UnitPricePence"], 4, retbins=True)[1][1:-1].round())) + [10**12]
    card["retail_price_bands_pence"] = price_bands
    print(f"  sG = {sG:.0f} 便士; 价格区间 = {price_bands[1:-1]}")

    bdev = pd.read_csv(os.path.join(SPLIT_DIR, "bank_dev.csv.gz"))
    sD = float(bdev["duration"].mean())  # 开发集平均联系时长
    card["bank_duration_scale_sD"] = sD
    print(f"  sD = {sD:.1f} 秒")
    # Bank 模板也全部用开发集冻结：最常见 4 个职业 + 余额四分位区间
    job_top4 = bdev["job"].value_counts().head(4).index.tolist()
    bal_bands = [float(bdev["balance"].min()) - 1] + \
        sorted(pd.qcut(bdev["balance"], 4, retbins=True)[1][1:-1].round().tolist()) + \
        [float(bdev["balance"].max()) + 1]
    card["bank_job_top4"] = job_top4
    card["bank_balance_bands"] = bal_bands
    print(f"  职业模板 = {job_top4}; 余额区间 = {bal_bands[1:-1]}")

    def load_bank():
        bk = pd.read_csv(os.path.join(SPLIT_DIR, "bank_test.csv.gz"))
        return bk

    def load_retail():
        return pd.read_csv(os.path.join(SPLIT_DIR, "retail_test.csv.gz"))

    print("\n编译 Retail 任务 ...")
    ep_r = compile_dataset("retail", load_retail,
                           lambda b: retail_cohorts(b, price_bands),
                           retail_metrics, "sG", sG)
    print("\n编译 Bank 任务 ...")
    ep_b = compile_dataset("bank", load_bank,
                           lambda b: bank_cohorts(b, job_top4, bal_bands),
                           bank_metrics, "sD", sD)

    episodes = ep_r + ep_b
    with open(os.path.join(DATA_DIR, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(DATA_DIR, "data_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    print(f"\n完成：共 {len(episodes)} 个任务（预期 240）")
    print(f"  Retail: {len(ep_r)} 任务；Bank: {len(ep_b)} 任务")
    print(f"  兜底块数：Retail {card['retail_fallback_blocks']} / Bank {card['bank_fallback_blocks']}")
    print(f"  不可构造块：Retail {card['retail_unconstructable_blocks']} / Bank {card['bank_unconstructable_blocks']}")


if __name__ == "__main__":
    main()
