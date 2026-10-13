# -*- coding: utf-8 -*-
"""
第十五步·最终版：论文主表 + 图表汇总（全部种子）

与 step15_summary.py（种子17初步版）的区别：
  - LLM 方法：纳入全部已完成种子（17/29/43），每格 n=240×3=720
    配对检验按 (episode_id, seed) 配对 —— 同一任务同一种子内比条件，
    种子主效应在差分中自动抵消
  - 确定性方法：数据仅有种子 17（批任务只补跑了 LLM），维持 n=240 并显式标注
  - 新增跨种子稳定性表：LLM 方法逐种子格子均值（复现性证据）
  - 旧的种子17输出文件保留不动

输出（analysis/ 目录）：
  - main_table_final.md        主表（LLM n=720，确定性 n=240 标注）
  - paired_tests_final.md      条件差异配对检验（Holm 校正按方法族）
  - cross_seed_check.md        跨种子稳定性表 + 逐种子配对检验复现性
  - fig_*_final.png            图1~4（全部可用种子）
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
B_MAIN = 12

LLM_METHODS = ["SAS-Tool", "SAS-SC", "MAS-Deliberate", "Exchange-then-Decide"]
DET_METHODS = ["ODC-WC", "ODC-CS", "Uniform-Cert", "Width-Cert", "NoLLM-E2E",
               "FullInfo"]
CONDS = ["A", "C", "S"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SHORT = {"MAS-Deliberate": "MAS", "Exchange-then-Decide": "ETD",
         "Uniform-Cert": "Unif", "Width-Cert": "Width", "NoLLM-E2E": "E2E",
         "SAS-Tool": "SAS-T", "SAS-SC": "SAS-SC", "ODC-WC": "ODC-WC",
         "ODC-CS": "ODC-CS", "FullInfo": "Full"}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def holm_adjust(pvals):
    """返回与 pvals 顺序对应的 Holm 校正后 p 值列表。"""
    order = np.argsort(pvals)
    m_h = len(pvals)
    holm = {}
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = max(prev, min(1.0, (m_h - rank) * pvals[idx]))
        holm[idx] = adj
        prev = adj
    return [holm[i] for i in range(len(pvals))]


def paired_test(records_a, records_b):
    """输入两个 {(episode_id, seed): regret} 字典，返回 (n, 均差, t, p)。"""
    common = sorted(set(records_a) & set(records_b))
    if len(common) < 30:
        return None
    d = np.array([records_a[k] - records_b[k] for k in common])
    t, p = stats.ttest_1samp(d, 0)
    return len(common), d.mean(), t, p


def main():
    os.makedirs(OUT, exist_ok=True)
    llm = load(os.path.join(DATA, "formal_results.jsonl"))
    det = load(os.path.join(DATA, "formal_deterministic.jsonl"))

    llm_seeds = sorted({r["seed"] for r in llm})
    det_seeds = sorted({r["seed"] for r in det})
    print(f"LLM 种子: {llm_seeds}  确定性种子: {det_seeds}")

    # ---------- 主表 ----------
    rows = []
    for cond in CONDS:
        for m in LLM_METHODS:
            sub = [r for r in llm if r["condition"] == cond and r["method"] == m
                   and r.get("audit_ok")]
            if not sub:
                continue
            regs = [r["true_regret"] for r in sub]
            rows.append({"条件": cond, "方法": m, "类型": "LLM",
                         "后悔": np.mean(regs), "后悔SE": stats.sem(regs),
                         "成本": np.mean([r["cost"] for r in sub]),
                         "认证率": None, "兜底率": np.mean([r["fallback"] for r in sub]),
                         "n": len(sub), "seeds": len({r['seed'] for r in sub})})
        for m in DET_METHODS:
            b_eff = 24 if m == "FullInfo" else B_MAIN
            sub = [r for r in det if r["condition"] == cond and r["method"] == m
                   and r["budget"] == b_eff]
            if not sub:
                continue
            regs = [r["true_regret"] for r in sub]
            rows.append({"条件": cond, "方法": m, "类型": "确定性",
                         "后悔": np.mean(regs), "后悔SE": stats.sem(regs),
                         "成本": np.mean([r["cost"] for r in sub]),
                         "认证率": np.mean([r["certified"] for r in sub]),
                         "兜底率": None, "n": len(sub),
                         "seeds": len({r['seed'] for r in sub})})

    with open(os.path.join(OUT, "main_table_final.md"), "w",
              encoding="utf-8") as f:
        det_txt = (f"确定性方法种子 {det_seeds}，n=240/格"
                   if len(det_seeds) == 1 else
                   f"确定性方法 n={240 * len(det_seeds)}/格 = 240任务×{len(det_seeds)}种子{det_seeds}")
        f.write(f"# 主表（最终版：LLM 方法 n={240 * len(llm_seeds)}/格 = 240任务×{len(llm_seeds)}种子"
                f"{llm_seeds}；{det_txt}）\n\n")
        f.write("| 条件 | 方法 | 类型 | 均后悔 | SE | 均成本 | 认证率 | 兜底率 | n | 种子数 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            cert = f"{r['认证率']:.3f}" if r["认证率"] is not None else "—"
            fb = f"{r['兜底率']:.3f}" if r["兜底率"] is not None else "—"
            f.write(f"| {r['条件']} | {r['方法']} | {r['类型']} | {r['后悔']:.4f} "
                    f"| {r['后悔SE']:.4f} | {r['成本']:.1f} | {cert} | {fb} "
                    f"| {r['n']} | {r['seeds']} |\n")
        det_note = (f"确定性方法为种子 {det_seeds}。"
                    if len(det_seeds) < len(llm_seeds)
                    else f"确定性方法与 LLM 方法同为种子 {llm_seeds}。")
        f.write("\n注：LLM 方法为自然停止成本；确定性方法取预算 B=12（FullInfo 按其定义取 B=24）。\n")
        f.write(det_note + "\n")
    print("主表 ->", os.path.join(OUT, "main_table_final.md"))

    # ---------- 配对检验（最终版） ----------
    lines = ["# 条件差异配对检验（最终版，逐方法，Holm 校正按方法族）\n",
             "配对方式：LLM 方法按 (任务, 种子) 配对（n=720，种子主效应在差分中抵消）；"
             "确定性方法仅种子 17（n=240）。\n"]
    tests, pvals_all = [], []
    for m in LLM_METHODS + [d for d in DET_METHODS if d != "FullInfo"]:
        for pair in [("C", "A"), ("S", "A"), ("S", "C")]:
            def series(cond):
                if m in LLM_METHODS:
                    return {(r["episode_id"], r["seed"]): r["true_regret"]
                            for r in llm if r["condition"] == cond
                            and r["method"] == m and r.get("audit_ok")}
                return {(r["episode_id"], r["seed"]): r["true_regret"]
                        for r in det if r["condition"] == cond
                        and r["method"] == m and r["budget"] == B_MAIN}
            res = paired_test(series(pair[0]), series(pair[1]))
            if res is None:
                continue
            n, md, t, p = res
            tests.append((m, f"{pair[0]}−{pair[1]}", n, md, t, p))
            pvals_all.append(p)
    holm = holm_adjust(pvals_all)
    lines.append("| 方法 | 对比 | n | 均差 | t | p | Holm校正后 |")
    lines.append("|---|---|---|---|---|---|---|")
    for (m, cmp_, n, md, t, p), hp in zip(tests, holm):
        sig = "*" if hp < 0.05 else ""
        lines.append(f"| {m} | {cmp_} | {n} | {md:+.4f} | {t:.2f} "
                     f"| {p:.4f} | {hp:.4f}{sig} |")
    with open(os.path.join(OUT, "paired_tests_final.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("配对检验 ->", os.path.join(OUT, "paired_tests_final.md"))

    # ---------- 跨种子稳定性表（LLM 方法） ----------
    lines = ["# 跨种子稳定性检查（LLM 方法，逐种子格子均值）\n",
             "每个格子 = 条件×方法×种子（n=240）。若三种子方向一致，"
             "说明种子17的初步结论可复现。\n",
             "| 条件 | 方法 | 种子17 | 种子29 | 种子43 | 种子间极差 | 方向一致 |",
             "|---|---|---|---|---|---|---|"]
    per_seed_p = {m: {f"{a}−{b}": [] for a, b in
                      [("C", "A"), ("S", "A"), ("S", "C")]}
                  for m in LLM_METHODS}
    for cond in CONDS:
        for m in LLM_METHODS:
            per_seed = {}
            for s in llm_seeds:
                sub = [r["true_regret"] for r in llm
                       if r["condition"] == cond and r["method"] == m
                       and r["seed"] == s and r.get("audit_ok")]
                per_seed[s] = np.mean(sub) if sub else np.nan
            rng = np.nanmax(list(per_seed.values())) - np.nanmin(list(per_seed.values()))
            consistent = rng < 0.02  # 2个百分点内视为一致
            lines.append(f"| {cond} | {m} | " + " | ".join(
                f"{per_seed[s]:.4f}" for s in llm_seeds) +
                f" | {rng:.4f} | {'✓' if consistent else '✗'} |")
    lines.append("\n## 逐种子配对检验复现性（p 值，未校正；看三列是否同向且量级相当）\n")
    lines.append("| 方法 | 对比 | " + " | ".join(f"种子{s}" for s in llm_seeds) + " | 方向一致 |")
    lines.append("|---|---|" + "---|" * (len(llm_seeds) + 1))
    for m in LLM_METHODS:
        for pair in [("C", "A"), ("S", "A"), ("S", "C")]:
            ps, diffs = [], []
            for s in llm_seeds:
                sa = {r["episode_id"]: r["true_regret"] for r in llm
                      if r["condition"] == pair[0] and r["method"] == m
                      and r["seed"] == s and r.get("audit_ok")}
                sb = {r["episode_id"]: r["true_regret"] for r in llm
                      if r["condition"] == pair[1] and r["method"] == m
                      and r["seed"] == s and r.get("audit_ok")}
                common = sorted(set(sa) & set(sb))
                d = np.array([sa[e] - sb[e] for e in common])
                t, p = stats.ttest_1samp(d, 0)
                ps.append(p)
                diffs.append(d.mean())
            same_dir = (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
                        or all(abs(d) < 1e-12 for d in diffs))
            lines.append(f"| {m} | {pair[0]}−{pair[1]} | " + " | ".join(
                f"{p:.4f}" for p in ps) + f" | {'✓' if same_dir else '✗'} |")
    with open(os.path.join(OUT, "cross_seed_check.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("跨种子检查 ->", os.path.join(OUT, "cross_seed_check.md"))

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
        ax.set_xticklabels([SHORT[m] for m in allm],
                           rotation=45, ha="right", fontsize=9)
        if cond == "A":
            ax.set_ylabel("真实后悔（越低越好）")
    fig.suptitle(f"图1  各方法真实后悔 × 披露条件（LLM: n=720=240×3种子{llm_seeds}；"
                 f"确定性: 种子17 n=240；误差线=95%CI）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_regret_by_condition_final.png"), dpi=150)
    plt.close(fig)
    print("图1 -> fig_regret_by_condition_final.png")

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
    ax.set_title("图2  查询成本对比（条件 A/C/S 均值；LLM=自然停止(3种子)，确定性=B=12(种子17))")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cost_final.png"), dpi=150)
    plt.close(fig)
    print("图2 -> fig_cost_final.png")

    # ---------- 图 3：anytime 曲线（确定性仅种子17；LLM 点为3种子×3条件均值） ----------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    style = {"ODC-WC": ("-o", "#DD8452"), "ODC-CS": ("-s", "#C44E52"),
             "Uniform-Cert": ("-^", "#8C8C8C"), "Width-Cert": ("-D", "#55A868"),
             "NoLLM-E2E": ("--o", "#CCB974")}
    for m, (mk, c) in style.items():
        xs, ys = [], []
        for B in [6, 12, 18, 24]:
            sub = [r for r in det if r["method"] == m and r["budget"] == B
                   and r["condition"] == "A"]
            if sub:
                xs.append(np.mean([r["cost"] for r in sub]))
                ys.append(np.mean([r["true_regret"] for r in sub]))
        ax.plot(xs, ys, mk, color=c, label=m, ms=5)
    for m in LLM_METHODS:
        sub = [r for r in llm if r["method"] == m and r.get("audit_ok")]
        if sub:
            ax.scatter(np.mean([r["cost"] for r in sub]),
                       np.mean([r["true_regret"] for r in sub]),
                       marker="*", s=220, zorder=5,
                       label=f"{m}（自然停止, 3种子）")
    ax.set_xlabel("平均查询成本")
    ax.set_ylabel("真实后悔")
    ax.set_title("图3  anytime 曲线（确定性方法，条件 A，种子17）与 LLM 自然停止点（3种子×3条件均值）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_anytime_final.png"), dpi=150)
    plt.close(fig)
    print("图3 -> fig_anytime_final.png")

    # ---------- 图 4：后悔分布箱线 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, cond in zip(axes, CONDS):
        data, labels_ = [], []
        for m in allm:
            if m in LLM_METHODS:
                v = [r["true_regret"] for r in llm if r["condition"] == cond
                     and r["method"] == m and r.get("audit_ok")]
            else:
                b_eff = 24 if m == "FullInfo" else B_MAIN
                v = [r["true_regret"] for r in det if r["condition"] == cond
                     and r["method"] == m and r["budget"] == b_eff]
            if v:
                data.append(v)
                labels_.append(SHORT.get(m, m))
        bp = ax.boxplot(data, tick_labels=labels_, showfliers=False,
                        patch_artist=True)
        for patch, c in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(f"条件 {cond}", fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        if cond == "A":
            ax.set_ylabel("单任务真实后悔")
    fig.suptitle(f"图4  后悔分布（LLM: 3种子{llm_seeds}合并；确定性: 种子17；箱线，隐藏离群点）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_regret_dist_final.png"), dpi=150)
    plt.close(fig)
    print("图4 -> fig_regret_dist_final.png")


if __name__ == "__main__":
    main()
