# -*- coding: utf-8 -*-
"""
第十五步：论文主表 + 图表汇总（种子 17 初步版）

数据源：
  - data/formal_results.jsonl         4 个 LLM 方法（自然停止成本，单轨迹）
  - data/formal_deterministic.jsonl   6 个确定性方法（B=6/12/18/24 全检查点）

输出（analysis/ 目录）：
  - main_table_seed17.md / .tex   主表：条件 × 方法 × 后悔/成本/认证率
  - paired_tests_seed17.md        条件差异配对检验（C−A、S−A，逐方法）
  - fig_regret_by_condition.png   分组柱状图：后悔 × 方法 × 条件
  - fig_cost.png                  成本对比
  - fig_anytime.png               anytime 曲线（确定性）+ LLM 自然成本点
  - fig_regret_dist.png           后悔分布箱线图

注意：种子 29/43 批跑完后重跑本脚本即得最终版（脚本自动纳入全部已完成种子）。
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
SEED_MAIN = 17
B_MAIN = 12

LLM_METHODS = ["SAS-Tool", "SAS-SC", "MAS-Deliberate", "Exchange-then-Decide"]
DET_METHODS = ["ODC-WC", "ODC-CS", "Uniform-Cert", "Width-Cert", "NoLLM-E2E",
               "FullInfo"]
CONDS = ["A", "C", "S"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    os.makedirs(OUT, exist_ok=True)
    llm = load(os.path.join(DATA, "formal_results.jsonl"))
    det = load(os.path.join(DATA, "formal_deterministic.jsonl"))

    # ---------- 主表（每个 条件×方法 一行；LLM 取自然停止，确定性取 B=12，FullInfo 取 B=24） ----------
    rows = []
    for cond in CONDS:
        for m in LLM_METHODS:
            sub = [r for r in llm if r["condition"] == cond and r["method"] == m
                   and r["seed"] == SEED_MAIN and r.get("audit_ok")]
            if not sub:
                continue
            regs = [r["true_regret"] for r in sub]
            rows.append({"条件": cond, "方法": m, "类型": "LLM",
                         "后悔": np.mean(regs), "后悔SE": stats.sem(regs),
                         "成本": np.mean([r["cost"] for r in sub]),
                         "认证率": None, "兜底率": np.mean([r["fallback"] for r in sub]),
                         "n": len(sub)})
        for m in DET_METHODS:
            b_eff = 24 if m == "FullInfo" else B_MAIN
            sub = [r for r in det if r["condition"] == cond and r["method"] == m
                   and r["budget"] == b_eff and r["seed"] == SEED_MAIN]
            if not sub:
                continue
            regs = [r["true_regret"] for r in sub]
            rows.append({"条件": cond, "方法": m, "类型": "确定性",
                         "后悔": np.mean(regs), "后悔SE": stats.sem(regs),
                         "成本": np.mean([r["cost"] for r in sub]),
                         "认证率": np.mean([r["certified"] for r in sub]),
                         "兜底率": None, "n": len(sub)})

    with open(os.path.join(OUT, "main_table_seed17.md"), "w",
              encoding="utf-8") as f:
        f.write("# 主表（种子 17，n=240/格）\n\n")
        f.write("| 条件 | 方法 | 类型 | 均后悔 | SE | 均成本 | 认证率 | 兜底率 | n |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            cert = f"{r['认证率']:.3f}" if r["认证率"] is not None else "—"
            fb = f"{r['兜底率']:.3f}" if r["兜底率"] is not None else "—"
            f.write(f"| {r['条件']} | {r['方法']} | {r['类型']} | {r['后悔']:.4f} "
                    f"| {r['后悔SE']:.4f} | {r['成本']:.1f} | {cert} | {fb} | {r['n']} |\n")
        f.write("\n注：LLM 方法为自然停止成本；确定性方法取预算 B=12（FullInfo 按其定义取 B=24）。\n")
        f.write("种子 29/43 批完成后重跑 step15_summary.py 生成最终版。\n")
    print("主表 ->", os.path.join(OUT, "main_table_seed17.md"))

    # ---------- 配对检验：条件差异（同任务同方法同种子，LLM + 确定性 B=12） ----------
    lines = ["# 条件差异配对检验（种子 17，逐方法，Holm 校正按方法族）\n"]
    pvals_all = []
    tests = []
    for m in LLM_METHODS + [d for d in DET_METHODS if d != "FullInfo"]:
        for pair in [("C", "A"), ("S", "A"), ("S", "C")]:
            def series(cond):
                if m in LLM_METHODS:
                    return {r["episode_id"]: r["true_regret"] for r in llm
                            if r["condition"] == cond and r["method"] == m
                            and r["seed"] == SEED_MAIN and r.get("audit_ok")}
                return {r["episode_id"]: r["true_regret"] for r in det
                        if r["condition"] == cond and r["method"] == m
                        and r["budget"] == B_MAIN and r["seed"] == SEED_MAIN}
            s1, s0 = series(pair[0]), series(pair[1])
            common = sorted(set(s1) & set(s0))
            if len(common) < 30:
                continue
            d = np.array([s1[e] - s0[e] for e in common])
            t, p = stats.ttest_1samp(d, 0)
            tests.append((m, f"{pair[0]}−{pair[1]}", len(common), d.mean(), t, p))
            pvals_all.append(p)
    # Holm 校正
    order = np.argsort(pvals_all)
    holm = {}
    m_h = len(pvals_all)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = max(prev, min(1.0, (m_h - rank) * pvals_all[idx]))
        holm[idx] = adj
        prev = adj
    lines.append("| 方法 | 对比 | n | 均差 | t | p | Holm校正后 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, (m, cmp_, n, md, t, p) in enumerate(tests):
        sig = "*" if holm[i] < 0.05 else ""
        lines.append(f"| {m} | {cmp_} | {n} | {md:+.4f} | {t:.2f} "
                     f"| {p:.4f} | {holm[i]:.4f}{sig} |")
    with open(os.path.join(OUT, "paired_tests_seed17.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("配对检验 ->", os.path.join(OUT, "paired_tests_seed17.md"))

    # ---------- 图 1：后悔分组柱状图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    allm = LLM_METHODS + DET_METHODS
    colors = (["#4C72B0"] * 4) + (["#DD8452"] * 6)
    for ax, cond in zip(axes, CONDS):
        means = [next((r["后悔"] for r in rows if r["条件"] == cond
                       and r["方法"] == m), np.nan) for m in allm]
        ses = [next((r["后悔SE"] for r in rows if r["条件"] == cond
                     and r["方法"] == m), 0) for m in allm]
        x = np.arange(len(allm))
        ax.bar(x, means, yerr=[1.96 * s for s in ses], color=colors,
               capsize=3, error_kw={"linewidth": 1})
        ax.set_title(f"条件 {cond}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(["SAS-T", "SAS-SC", "MAS", "ETD", "ODC-WC",
                            "ODC-CS", "Unif", "Width", "E2E", "Full"],
                           rotation=45, ha="right", fontsize=9)
        if cond == "A":
            ax.set_ylabel("真实后悔（越低越好）")
    fig.suptitle(f"图1  各方法真实后悔 × 披露条件（种子 17，n=240，误差线=95%CI）",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_regret_by_condition.png"), dpi=150)
    plt.close(fig)
    print("图1 -> fig_regret_by_condition.png")

    # ---------- 图 2：成本对比 ----------
    fig, ax = plt.subplots(figsize=(10, 4.6))
    means, labels = [], []
    for m in allm:
        cs = [r["成本"] for r in rows if r["方法"] == m]
        means.append(np.mean(cs) if cs else np.nan)
        labels.append(m)
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors)
    for xi, v in zip(x, means):
        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    ax.axhline(24, ls="--", c="gray", lw=1)
    ax.text(0.1, 24.3, "预算上限 24", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("平均查询成本")
    ax.set_title("图2  查询成本对比（条件 A/C/S 均值；LLM=自然停止，确定性=B=12）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cost.png"), dpi=150)
    plt.close(fig)
    print("图2 -> fig_cost.png")

    # ---------- 图 3：anytime 曲线 ----------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    style = {"ODC-WC": ("-o", "#DD8452"), "ODC-CS": ("-s", "#C44E52"),
             "Uniform-Cert": ("-^", "#8C8C8C"), "Width-Cert": ("-D", "#55A868"),
             "NoLLM-E2E": ("--o", "#CCB974")}
    for m, (mk, c) in style.items():
        xs, ys = [], []
        for B in [6, 12, 18, 24]:
            sub = [r for r in det if r["method"] == m and r["budget"] == B
                   and r["condition"] == "A" and r["seed"] == SEED_MAIN]
            if sub:
                xs.append(np.mean([r["cost"] for r in sub]))
                ys.append(np.mean([r["true_regret"] for r in sub]))
        ax.plot(xs, ys, mk, color=c, label=m, ms=5)
    # LLM 自然停止点（三条件均值）
    for m in LLM_METHODS:
        sub = [r for r in llm if r["method"] == m and r["seed"] == SEED_MAIN
               and r.get("audit_ok")]
        if sub:
            ax.scatter(np.mean([r["cost"] for r in sub]),
                       np.mean([r["true_regret"] for r in sub]),
                       marker="*", s=220, zorder=5,
                       label=f"{m}（自然停止）")
    ax.set_xlabel("平均查询成本")
    ax.set_ylabel("真实后悔")
    ax.set_title("图3  anytime 曲线（确定性方法，条件 A）与 LLM 方法自然停止点（三条件均值）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_anytime.png"), dpi=150)
    plt.close(fig)
    print("图3 -> fig_anytime.png")

    # ---------- 图 4：后悔分布箱线 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, cond in zip(axes, CONDS):
        data, labels_ = [], []
        for m in allm:
            if m in LLM_METHODS:
                v = [r["true_regret"] for r in llm if r["condition"] == cond
                     and r["method"] == m and r["seed"] == SEED_MAIN
                     and r.get("audit_ok")]
            else:
                b_eff = 24 if m == "FullInfo" else B_MAIN
                v = [r["true_regret"] for r in det if r["condition"] == cond
                     and r["method"] == m and r["budget"] == b_eff
                     and r["seed"] == SEED_MAIN]
            if v:
                data.append(v)
                labels_.append({"MAS-Deliberate": "MAS",
                                "Exchange-then-Decide": "ETD",
                                "Uniform-Cert": "Unif",
                                "Width-Cert": "Width",
                                "NoLLM-E2E": "E2E"}.get(m, m))
        bp = ax.boxplot(data, tick_labels=labels_, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(f"条件 {cond}", fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        if cond == "A":
            ax.set_ylabel("单任务真实后悔")
    fig.suptitle("图4  后悔分布（种子 17，箱线；隐藏离群点）", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_regret_dist.png"), dpi=150)
    plt.close(fig)
    print("图4 -> fig_regret_dist.png")


if __name__ == "__main__":
    main()
