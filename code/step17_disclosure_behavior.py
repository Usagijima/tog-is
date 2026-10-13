# -*- coding: utf-8 -*-
"""
第十七步：部门披露行为分析（论文"策略性隐瞒"的行为学证据）

数据源：data/disclosures.jsonl（seed 17，240 任务 × A/C/S；每格取首条记录）
结合 data/episodes.jsonl 的真实状态 z 与组织最优行动，测量：

  M1 本部门 KPI 最优群组披露率      —— 部门有没有秀出自己的第一名
  M2 组织最优群组的部门事实披露率   —— 决策最关键的事实有没有被披露
  M3 E0 与 A 脚本的重合度（Jaccard）—— LLM 披露偏离固定规则多少
  M4 兜底率与规则分布               —— LLM 服从性体检

输出：analysis/disclosure_behavior.md + fig_disclosure.png
"""

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step6_certificate import CertificateCore

DATA = "data"
OUT = "analysis"
SEED = 17
CONDS = ["A", "C", "S"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    os.makedirs(OUT, exist_ok=True)
    episodes = {e["episode_id"]: e for e in load(os.path.join(DATA, "episodes.jsonl"))}

    # 每格取首条（与正式实验 load_E0 行为一致）；纳入全部已完成种子
    SEEDS = sorted({r["seed"] for r in load(os.path.join(DATA, "disclosures.jsonl"))})
    disc = {}
    for r in load(os.path.join(DATA, "disclosures.jsonl")):
        key = (r["episode_id"], r["condition"], r["seed"])
        if key not in disc:
            disc[key] = r

    # 每任务预计算：部门本部门最优群组、组织最优群组
    stats = {c: {"m1": [], "m2": [], "m3": [], "fallback": 0, "rules": Counter()}
             for c in CONDS}
    n_total = 0
    for (ep_id, cond, _), rec in sorted(disc.items()):
        if ep_id not in episodes:      # 跳过开发任务的遗留记录
            continue
        ep = episodes[ep_id]
        z = ep["z_int"]
        core = CertificateCore(ep)
        # 组织最优行动：直接对所有行动算真实得分取最大
        def score(a_idx):
            a = core.actions[a_idx]
            return sum(z[k * 3 + i] * ep["weights"][i]
                       for k in a for i in range(3)) / ep["h"]
        a_star_idx = max(range(len(core.actions)), key=score)
        org_groups = set(core.actions[a_star_idx])

        e0 = rec["E0"]
        disclosed_groups = {j // 3 for j in e0}
        s = stats[cond]
        n_total += 1

        # M1：三个部门各自的本部门最优群组是否被披露（取部门列第一的群组）
        for dept in range(3):
            col = [z[k * 3 + dept] for k in range(8)]
            best_k = max(range(8), key=lambda k: (col[k], -k))
            s["m1"].append(1.0 if best_k in disclosed_groups else 0.0)
        # M2：组织最优群组的 3 个部门事实是否出现在 E0
        for dept in range(3):
            j_star = next(iter(org_groups)) * 3 + dept
            s["m2"].append(1.0 if j_star in e0 else 0.0)
        # 规则与兜底
        for d in rec["disclosures"].values():
            s["rules"][d["rule"]] += 1
            if "fallback" in d["rule"]:
                s["fallback"] += 1

    # M3：与 A 脚本 E0 的 Jaccard（每条件 vs A，按种子配对）
    base = {}
    for (ep_id, cond, sd), rec in disc.items():
        if cond == "A":
            base[(ep_id, sd)] = set(rec["E0"])
    for (ep_id, cond, sd), rec in disc.items():
        if cond != "A" and (ep_id, sd) in base:
            a, b = set(rec["E0"]), base[(ep_id, sd)]
            stats[cond]["m3"].append(len(a & b) / len(a | b))

    # ---------- 输出 ----------
    lines = [f"# 部门披露行为分析（种子 {SEEDS}，{240 * len(SEEDS)} 任务格，每格 3 部门）\n",
             "| 指标 | A（脚本） | C（合作） | S（自利） |",
             "|---|---|---|---|"]
    def row(name, key):
        vals = [f"{np.mean(stats[c][key]):.3f}" if stats[c][key] else "—"
                for c in CONDS]
        return f"| {name} | " + " | ".join(vals) + " |"
    lines.append(row("M1 本部门最优群组披露率", "m1"))
    lines.append(row("M2 组织最优群组的部门事实披露率", "m2"))
    lines.append(row("M3 与 A 脚本 E0 的 Jaccard 重合", "m3"))
    nd = {c: len(stats[c]["m1"]) for c in CONDS}   # 部门披露次数 = 任务格数 × 3
    lines.append("| M4 部门兜底率 | — | " +
                 f"{stats['C']['fallback']/nd['C']:.3f} | {stats['S']['fallback']/nd['S']:.3f} |")
    lines.append("\n## 规则分布（C 条件）\n")
    for rule, n in stats["C"]["rules"].most_common():
        lines.append(f"- {rule}: {n}")
    lines.append("\n## 规则分布（S 条件）\n")
    for rule, n in stats["S"]["rules"].most_common():
        lines.append(f"- {rule}: {n}")
    with open(os.path.join(OUT, "disclosure_behavior.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    xs = np.arange(3)
    titles = ["M1 秀出本部门第一名", "M2 披露组织最优群组事实",
              "M3 与 A 脚本的 E0 重合度"]
    for ax, key, title in zip(axes, ["m1", "m2", "m3"], titles):
        vals = [np.mean(stats[c][key]) if stats[c][key] else np.nan
                for c in CONDS]
        bars = ax.bar(xs, vals, color=["#DD8452", "#4C72B0", "#C44E52"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(CONDS)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"图6  三种披露条件下的部门行为差异（种子 {SEEDS}，n={240 * len(SEEDS)}/条件）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_disclosure.png"), dpi=150)
    plt.close(fig)
    print("图 ->", os.path.join(OUT, "fig_disclosure.png"))


if __name__ == "__main__":
    main()
