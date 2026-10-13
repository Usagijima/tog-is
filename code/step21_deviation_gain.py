# -*- coding: utf-8 -*-
"""
第二十二步：有限偏离收益（bounded deviation gain，论文 6.5 节 M5）

定义（对每条 LLM 轨迹）：
  部门 i 的 KPI 为 V_i(a;z) = (1/h) * sum_{k in a} z_{ki}（与组织权重无关的部门列均值）；
  a* 为组织最优行动（对 W 取 max，W = sum_i w_i V_i）。
  Δ_i = V_i(â) - V_i(a*)：所选行动相对最优行动给部门 i 带来的 KPI 变化。
  偏离收益 DevGain = max_i Δ_i —— 受益最大部门的私利所得。
  "有限"：V_i ∈ [0,1]，故 DevGain ∈ [-1,1]；且 W(â) ≤ W(a*) 意味着
  sum_i w_i Δ_i ≤ 0 —— 任何私利收益都由组织得分（与其他部门）支付。

检验：按 (任务, 方法, 种子) 配对，比较 S/C/A 三条件的 DevGain；
Holm 校正（方法族 = 4 LLM 方法 × 3 对比）。

输出：analysis/deviation_gain.md
"""

import itertools
import json
import os

import numpy as np
from scipy import stats

DATA = "data"
OUT = "analysis"

LLM_METHODS = ["SAS-Tool", "SAS-SC", "MAS-Deliberate", "Exchange-then-Decide"]
CONDS = ["A", "C", "S"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def kpi(z, a, dept, h):
    return sum(z[k * 3 + dept] for k in a) / h


def org_score(z, a, w, h):
    return sum(w[i] * kpi(z, a, i, h) for i in range(3))


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = {e["episode_id"]: e for e in load(os.path.join(DATA, "episodes.jsonl"))
           if e["block"] != "dev"}
    # 预计算每个任务的全部行动与组织最优行动
    all_actions = {}
    a_star = {}
    for ep_id, e in eps.items():
        acts = [list(c) for c in itertools.combinations(range(8), e["h"])]
        all_actions[ep_id] = acts
        a_star[ep_id] = max(acts, key=lambda a: org_score(e["z_int"], a,
                                                          e["weights"], e["h"]))

    # DevGain[(episode, method, cond, seed)]
    gain = {}
    org_loss = {}
    n_skip = 0
    for r in load(os.path.join(DATA, "formal_results.jsonl")):
        if not r.get("audit_ok") or r["episode_id"] not in eps:
            continue
        e = eps[r["episode_id"]]
        z, w, h = e["z_int"], e["weights"], e["h"]
        a, as_ = r["action"], a_star[r["episode_id"]]
        if len(a) != h:
            n_skip += 1
            continue
        deltas = [(kpi(z, a, i, h) - kpi(z, as_, i, h)) / 100000.0
                  for i in range(3)]
        key = (r["episode_id"], r["method"], r["condition"], r["seed"])
        gain[key] = max(deltas)
        org_loss[key] = (org_score(z, as_, w, h) - org_score(z, a, w, h)) / 100000.0
    if n_skip:
        print(f"警告：{n_skip} 条行动长度与 h 不符，已跳过")

    # 条件均值
    lines = ["# 有限偏离收益（bounded deviation gain，M5）\n",
             "DevGain = max_i [V_i(â) − V_i(a*)]，受益最大部门的私利所得；"
             "按 (任务, 方法, 种子) 配对检验。\n",
             "| 方法 | 条件 | 均 DevGain | SE | n |",
             "|---|---|---|---|---|"]
    for m in LLM_METHODS:
        for c in CONDS:
            vals = [v for (e, mm, cc, s), v in gain.items() if mm == m and cc == c]
            lines.append(f"| {m} | {c} | {np.mean(vals):+.4f} "
                         f"| {stats.sem(vals):.4f} | {len(vals)} |")

    # 配对检验 + Holm
    tests, pvals = [], []
    for m in LLM_METHODS:
        for c1, c0 in [("S", "A"), ("S", "C"), ("C", "A")]:
            common = sorted({(e, s) for (e, mm, cc, s) in gain
                             if mm == m and cc == c1}
                            & {(e, s) for (e, mm, cc, s) in gain
                               if mm == m and cc == c0})
            d = np.array([gain[(e, m, c1, s)] - gain[(e, m, c0, s)]
                          for e, s in common])
            t, p = stats.ttest_1samp(d, 0)
            tests.append((m, f"{c1}−{c0}", len(common), d.mean(), t, p))
            pvals.append(p)
    order = np.argsort(pvals)
    holm = {}
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = max(prev, min(1.0, (len(pvals) - rank) * pvals[idx]))
        holm[idx] = adj
        prev = adj
    lines += ["\n| 方法 | 对比 | n | 均差 | t | p | Holm校正后 |",
              "|---|---|---|---|---|---|---|"]
    for i, (m, cmp_, n, md, t, p) in enumerate(tests):
        sig = "*" if holm[i] < 0.05 else ""
        lines.append(f"| {m} | {cmp_} | {n} | {md:+.4f} | {t:.2f} "
                     f"| {p:.4f} | {holm[i]:.4f}{sig} |")

    # 汇总统计（全部方法合并）
    pooled = {c: [v for (e, mm, cc, s), v in gain.items() if cc == c]
              for c in CONDS}
    lines += ["\n## 合并（4 方法 × 240 任务 × 3 种子）\n",
              "| 条件 | 均 DevGain | 5% 分位 | 中位 | 95% 分位 |",
              "|---|---|---|---|---|"]
    for c in CONDS:
        v = np.array(pooled[c])
        lines.append(f"| {c} | {v.mean():+.4f} | {np.percentile(v, 5):+.4f} "
                     f"| {np.median(v):+.4f} | {np.percentile(v, 95):+.4f} |")
    frac_pos = {c: float(np.mean(np.array(pooled[c]) > 1e-9)) for c in CONDS}
    lines.append("")
    for c in CONDS:
        lines.append(f"- 条件 {c} 下 DevGain > 0 的轨迹比例：{frac_pos[c]:.1%}")
    # 私利收益与组织损失的关系（同轨迹配对）
    both = [(gain[k], org_loss[k]) for k in gain]
    dg = np.array([b[0] for b in both])
    ol = np.array([b[1] for b in both])
    lines.append(f"- DevGain 与组织损失（=真实后悔）的 Pearson 相关："
                 f"r = {np.corrcoef(dg, ol)[0, 1]:.3f}（n={len(both)}）")

    with open(os.path.join(OUT, "deviation_gain.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
