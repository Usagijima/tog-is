# -*- coding: utf-8 -*-
"""
第二十步：收据替换因果实验——确定性接收端（方案 3.md 第 14.3 节）

方案 14.3：固定真实状态、接收方、推荐行动、来源身份、事实条数与格式，
只替换选出的事实集合 → 得到"披露集合本身"的纯因果效应。

确定性接收端天然满足冻结条件：ODC-WC / Width-Cert 的决策不读取部门推荐
（formal 模式 dept_best=None），提示词恒定，唯一变化通道就是 E0 事实集合。
因此 formal_deterministic.jsonl 的三条件对比本身就是该实验的实现：
本脚本做逐任务配对分析（同任务、同方法、跨条件），把"趋势"升级为因果读数。

LLM 接收端的收据替换需要服务器，待种子批完成后另行执行（EXP 留口）。

输出：analysis/receipt_replacement.md + fig_receipt_replacement.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA = "data"
OUT = "analysis"
SEED = 17
B = 12
METHODS = ["ODC-WC", "Width-Cert", "NoLLM-E2E"]
PAIRS = [("C", "A"), ("S", "A"), ("S", "C")]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(DATA, "formal_deterministic.jsonl"),
                                       encoding="utf-8")]
    seeds = sorted({r["seed"] for r in rows})
    reg = {}
    for r in rows:
        if r["budget"] == B and r["method"] in METHODS:
            reg[(r["episode_id"], r["seed"], r["method"], r["condition"])] = r["true_regret"]

    lines = [f"# 收据替换因果实验（确定性接收端，种子 {seeds}，B=12，按 (任务,种子) 配对）\n",
             "冻结：真实状态 / 接收算法 / 提示 / 推荐（不进入决策）/ 条数（6）/ 格式；"
             "唯一变化：E0 事实集合。\n",
             "| 方法 | 替换 | n | 配对均差（后悔） | t | p | Holm |",
             "|---|---|---|---|---|---|---|"]
    tests = []
    for m in METHODS:
        for c1, c0 in PAIRS:
            keys = sorted({(e, s) for (e, s, mm, c) in reg if mm == m})
            common = [(e, s) for e, s in keys
                      if (e, s, m, c1) in reg and (e, s, m, c0) in reg]
            if len(common) < 30:
                continue
            d = np.array([reg[(e, s, m, c1)] - reg[(e, s, m, c0)]
                          for e, s in common])
            t, p = stats.ttest_1samp(d, 0)
            tests.append((m, f"{c1}−{c0}", len(common), d.mean(), t, p, d))
    order = np.argsort([t[5] for t in tests])
    m_h = len(tests)
    holm = {}
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = max(prev, min(1.0, (m_h - rank) * tests[idx][5]))
        holm[idx] = adj
        prev = adj
    fig_data = {}
    for i, (m, cmp_, n, md, t, p, d) in enumerate(tests):
        sig = "*" if holm[i] < 0.05 else ""
        lines.append(f"| {m} | {cmp_} | {n} | {md:+.4f} | {t:.2f} "
                     f"| {p:.4f} | {holm[i]:.4f}{sig} |")
        fig_data.setdefault(m, {})[cmp_] = d
    with open(os.path.join(OUT, "receipt_replacement.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------- 图：配对差分布 ----------
    fig, axes = plt.subplots(1, len(METHODS), figsize=(13, 4.2), sharey=True)
    comps = ["C−A", "S−A", "S−C"]
    for ax, m in zip(axes, METHODS):
        data = [fig_data[m][c] for c in comps if c in fig_data.get(m, {})]
        bp = ax.boxplot(data, tick_labels=[c for c in comps if c in fig_data.get(m, {})],
                        showfliers=False, patch_artist=True,
                        medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.6)
        ax.axhline(0, color="red", lw=1, ls="--")
        ax.set_title(m)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("配对后悔差（替换条件 − 基准条件）")
    fig.suptitle(f"图8  收据替换实验：只换披露事实集合的因果效应（确定性接收端，B=12，种子 {seeds}）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_receipt_replacement.png"), dpi=150)
    plt.close(fig)
    print("图 ->", os.path.join(OUT, "fig_receipt_replacement.png"))


if __name__ == "__main__":
    main()
