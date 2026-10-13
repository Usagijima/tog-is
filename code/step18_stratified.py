# -*- coding: utf-8 -*-
"""
第十八步：分层主表（数据集 × 行动规模 h，论文 16.1 节要求的分层报告）

输出 analysis/main_table_stratified.md：
  每个 数据集(retail/bank) × h(1/2) × 条件 × 方法 的均后悔/均成本（seed 17）
  并给出"排序跨层是否稳定"的自动检查（LLM 方法内部排序 + 确定性方法内部排序）。
"""

import json
import os
from collections import defaultdict

import numpy as np

DATA = "data"
OUT = "analysis"
SEED = 17
B_MAIN = 12

LLM = ["SAS-Tool", "SAS-SC", "MAS-Deliberate", "Exchange-then-Decide"]
DET = ["ODC-WC", "ODC-CS", "Uniform-Cert", "Width-Cert", "NoLLM-E2E", "FullInfo"]
CONDS = ["A", "C", "S"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = {e["episode_id"]: e for e in load(os.path.join(DATA, "episodes.jsonl"))}
    llm = load(os.path.join(DATA, "formal_results.jsonl"))
    det = load(os.path.join(DATA, "formal_deterministic.jsonl"))

    llm_seeds = sorted({r["seed"] for r in llm})
    # 聚合单元：(dataset, h, cond, method) -> regrets/costs（纳入全部种子）
    agg = defaultdict(lambda: {"reg": [], "cost": [], "cert": []})
    for r in llm:
        if not r.get("audit_ok") or r["episode_id"] not in eps:
            continue
        ep = eps[r["episode_id"]]
        key = (ep["dataset"], ep["h"], r["condition"], r["method"])
        agg[key]["reg"].append(r["true_regret"])
        agg[key]["cost"].append(r["cost"])
    for r in det:
        if r["episode_id"] not in eps:
            continue
        b_eff = 24 if r["method"] == "FullInfo" else B_MAIN
        if r["budget"] != b_eff:
            continue
        ep = eps[r["episode_id"]]
        key = (ep["dataset"], ep["h"], r["condition"], r["method"])
        agg[key]["reg"].append(r["true_regret"])
        agg[key]["cost"].append(r["cost"])
        agg[key]["cert"].append(r["certified"])

    lines = [f"# 分层主表（种子 {llm_seeds}，n=240×{len(llm_seeds)}/格；后悔为宏平均）\n"]
    order = LLM + DET
    check_notes = []
    for ds in ("retail", "bank"):
        for h in (1, 2):
            lines.append(f"\n## 数据集 {ds}，h={h}（选 {h} 个群组）\n")
            lines.append("| 条件 | 方法 | 均后悔 | SE | 均成本 | 认证率 | n |")
            lines.append("|---|---|---|---|---|---|---|")
            for cond in CONDS:
                for m in order:
                    key = (ds, h, cond, m)
                    if key not in agg or not agg[key]["reg"]:
                        continue
                    a = agg[key]
                    cert = (f"{np.mean(a['cert']):.3f}" if a["cert"] and m in DET
                            else "—")
                    lines.append(
                        f"| {cond} | {m} | {np.mean(a['reg']):.4f} "
                        f"| {np.std(a['reg'])/np.sqrt(len(a['reg'])):.4f} "
                        f"| {np.mean(a['cost']):.1f} | {cert} | {len(a['reg'])} |")
            # 排序稳定性检查（LLM 组内按后悔）
            ranks = []
            for cond in CONDS:
                vals = [(m, np.mean(agg[(ds, h, cond, m)]["reg"]))
                        for m in LLM if (ds, h, cond, m) in agg]
                if len(vals) == 4:
                    ranks.append([m for m, _ in sorted(vals, key=lambda x: x[1])])
            if ranks and all(r == ranks[0] for r in ranks):
                check_notes.append(
                    f"- {ds} h={h}：LLM 方法后悔排序跨三条件稳定（"
                    + " < ".join(ranks[0]) + "）")
            elif ranks:
                check_notes.append(f"- {ds} h={h}：LLM 排序跨条件有变动 {ranks}")

    # 跨层方向检查：h=2 后悔普遍高于 h=1？
    h1, h2 = [], []
    for (ds, h, cond, m), a in agg.items():
        if m in LLM and cond == "A":
            (h1 if h == 1 else h2).append(np.mean(a["reg"]))
    lines.append("\n# 跨层稳定性自动检查\n")
    lines += check_notes
    lines.append(f"\n- LLM 方法条件 A 均后悔：h=1 层平均 {np.mean(h1):.4f}"
                 f"（n={len(h1)} 格），h=2 层平均 {np.mean(h2):.4f}"
                 f"（n={len(h2)} 格）——"
                 + ("h=2 全面更难，与 EXP-016 一致 ✓"
                    if np.mean(h2) > np.mean(h1) else "注意：h 难度方向不一致"))

    with open(os.path.join(OUT, "main_table_stratified.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(check_notes))
    print(f"h=1: {np.mean(h1):.4f}  h=2: {np.mean(h2):.4f}")
    print("->", os.path.join(OUT, "main_table_stratified.md"))


if __name__ == "__main__":
    main()
