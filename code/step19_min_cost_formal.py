# -*- coding: utf-8 -*-
"""
第十九步：最小证书补全成本 + 遗漏税（正式版，方案 3.md 第 12 节）

在 step9 原型的数学（精确减少量恒等式，自测 100 组通过）基础上扩到正式规模：
  240 测试任务 × {A, C, S, neutral} 条件 × seed 17 共享 E0

指标（论文 C3）：
  C*(episode, cond)  — 本证书类别下的理论最小补满成本（子集升序精确枚举，cap=4；
                       搜不到给贪心上限并标 exact=False，绝不冒充精确最优——方案红线）
  OmissionTax(cond)  — 条件 cond 的 C* 减去 neutral 的 C*（同任务配对）：
                       句句属实的披露本身造成多少额外治理负担
  CostRatio          — ODC-WC 自然停止的实际总成本 / (6 + C*)：查询启发式问得准不准

注：step9.min_completion_cost 按行动顺序返回第一个可认证行动的 k，是该行动的
    上界而非全局最小；本脚本改为 k 升序为主、行动为内的正确枚举。

输出：data/formal_min_cost.jsonl、analysis/min_cost.md、analysis/fig_min_cost.png
"""

import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import milp, LinearConstraint, Bounds

from step6_certificate import CertificateCore
from step8_methods import disclose_neutral
from step9_min_cost import reduction_matrix, self_test_identity
from step13_disclosure import load_E0

DATA = "data"
OUT = "analysis"
SEED = 17
EPS = 0.05           # 预注册阈值

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def min_completion_global(core, z, only_actions=None):
    """正确的全局最小补满成本：对每个真实后悔≤ε的行动解一次精确 MILP
    （scipy.optimize.milp，HiGHS），取最小目标值。

    only_actions：限定行动索引集合（如只认证真实最优行动）；None=任意可认证行动。
    返回 (C*, solved, None)。全部行动不可认证时 C*=None（如实报告）。
    注：step9 原型的升序枚举只搜到 cap=4，正式规模 95% 任务超出——
    MILP 对每个行动精确求解，无搜索上限（方案红线的"超时可行解充最优"风险
    由求解器的最优性保证消除）。
    """
    nA = len(core.actions)
    D = reduction_matrix(core, z)
    eps_s = core.eps_scaled
    unknown = [j for j in range(24) if not core.revealed[j]]
    U = len(unknown)

    best = None
    c_obj = np.ones(U)
    bounds = Bounds(np.zeros(U), np.ones(U))
    integrality = np.ones(U)
    for a in range(nA):
        if only_actions is not None and a not in only_actions:
            continue
        if core.true_regret(a, z) > eps_s:
            continue                       # 全披露也认证不了，排除
        need = np.array([core.pair_worst(a, b) - eps_s for b in range(nA)])
        active = need > 0
        if not active.any():
            return 0, True, None           # 已可认证
        A_mat = D[a][active][:, unknown]   # (竞争者, 未知事实)
        lc = LinearConstraint(A_mat, lb=need[active],
                              ub=np.full(active.sum(), np.inf))
        try:
            res = milp(c_obj, constraints=lc, integrality=integrality,
                       bounds=bounds, options={"time_limit": 10})
        except Exception:
            continue
        if res.status == 0 and res.fun is not None:
            k = int(round(res.fun))
            best = k if best is None else min(best, k)
    return best, True, None


def main():
    os.makedirs(OUT, exist_ok=True)
    if not self_test_identity(n=100):
        raise SystemExit("恒等式自测失败，中止")

    eps_list = [json.loads(l) for l in open(os.path.join(DATA, "episodes.jsonl"),
                                            encoding="utf-8")]
    assert all(e["block"] != "dev" for e in eps_list)

    seeds = ([int(sys.argv[sys.argv.index("--seed") + 1])] if "--seed" in sys.argv
             else [17, 29, 43])

    # ODC-WC 自然停止实际成本（B=24 行 = 预算耗尽或自然认证），按 (任务, 条件, 种子)
    actual_cost = {}
    for r in (json.loads(l) for l in open(os.path.join(DATA, "formal_deterministic.jsonl"),
                                          encoding="utf-8")):
        if r["method"] == "ODC-WC" and r["budget"] == 24:
            actual_cost[(r["episode_id"], r["condition"], r["seed"])] = r["cost"]

    raw_path = os.path.join(DATA, "formal_min_cost.jsonl")
    done = set()          # (ep, cond, seed)
    neutral_eps = set()   # neutral 披露不依赖种子，每个任务只需算一次
    if os.path.exists(raw_path):
        for l in open(raw_path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["episode_id"], r["condition"], r.get("seed", 17)))
            if r["condition"] == "neutral":
                neutral_eps.add(r["episode_id"])

    n_new = 0
    with open(raw_path, "a", encoding="utf-8") as f:
        for seed in seeds:
            for ep in eps_list:
                for cond in ("A", "C", "S", "neutral"):
                    if cond == "neutral":
                        if ep["episode_id"] in neutral_eps:
                            continue
                    elif (ep["episode_id"], cond, seed) in done:
                        continue
                    z = ep["z_int"]
                    if cond == "neutral":
                        init, _ = disclose_neutral(ep)
                    else:
                        init = load_E0(ep["episode_id"], cond, seed)["E0"]
                    core = CertificateCore(ep, epsilon=EPS)
                    for j in init:
                        core.apply_receipt(j, z[j])
                    c_star, exact, gub = min_completion_global(core, z)
                    # 变体：只认证真实最优行动（决策损害视角）
                    def _score(a_idx):
                        a = core.actions[a_idx]
                        return sum(z[k * 3 + i] * ep["weights"][i]
                                   for k in a for i in range(3)) / ep["h"]
                    opt_idx = max(range(len(core.actions)), key=_score)
                    c_opt, exact_o, _ = min_completion_global(
                        core, z, only_actions={opt_idx})
                    rec = {"episode_id": ep["episode_id"], "condition": cond,
                           "seed": seed, "c_star": c_star, "exact": exact,
                           "c_star_opt": c_opt, "exact_opt": exact_o,
                           "greedy_ub": gub}
                    if cond != "neutral":
                        ac = actual_cost.get((ep["episode_id"], cond, seed))
                        if ac is not None and exact and c_star is not None:
                            rec["odc_wc_actual"] = ac
                            rec["cost_ratio"] = round(ac / (6 + c_star), 3)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_new += 1
                    if cond == "neutral":
                        neutral_eps.add(ep["episode_id"])
    print(f"新增 {n_new} 行 -> {raw_path}")

    # ---------- 汇总（全部种子） ----------
    rows = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
    all_seeds = sorted({r.get("seed", 17) for r in rows if r["condition"] != "neutral"})
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)
    lines = [f"# 最小证书补全成本与遗漏税（种子 {all_seeds}，n={240 * len(all_seeds)}/条件）\n",
             f"ε = {EPS}；每行动精确 MILP 求解（HiGHS），无搜索上限；"
             "C*=None 表示该任务在 ε 下无任何可认证行动\n",
             "口径一 C*：认证【任意】可认证行动的最小补满；"
             "口径二 C*_opt：只认证【真实最优】行动的最小补满（决策损害视角）\n",
             "| 条件 | C* 均值 | OmissionTax | C*_opt 均值 | OmissionTax_opt | 配对 t | CostRatio | 无可认证行动 |",
             "|---|---|---|---|---|---|---|---|"]
    neutral = {r["episode_id"]: r for r in by.get("neutral", [])}
    from scipy import stats

    def paired_tax(rs, field):
        pairs = [(r[field], neutral[r["episode_id"]][field])
                 for r in rs if r["exact"] and r["episode_id"] in neutral
                 and neutral[r["episode_id"]]["exact"]
                 and r.get(field) is not None
                 and neutral[r["episode_id"]].get(field) is not None]
        if not pairs:
            return None, None, None, []
        d = np.array([a - b for a, b in pairs])
        t, p = stats.ttest_1samp(d, 0)
        return d.mean(), t, p, d

    for cond in ("A", "C", "S"):
        rs = by[cond]
        cstars = [r["c_star"] for r in rs if r["exact"] and r["c_star"] is not None]
        copts = [r["c_star_opt"] for r in rs
                 if r.get("exact_opt") and r.get("c_star_opt") is not None]
        n_nocert = sum(1 for r in rs if r["c_star"] is None)
        tax, tt, tp, _ = paired_tax(rs, "c_star")
        tax_o, tt_o, tp_o, _ = paired_tax(rs, "c_star_opt")
        ratios = [r["cost_ratio"] for r in rs if r.get("cost_ratio")]
        lines.append(
            f"| {cond} | {np.mean(cstars):.2f} | {tax:+.3f} (p={tp:.3f}) "
            f"| {np.mean(copts):.2f} | {tax_o:+.3f} (p={tp_o:.3f}) "
            f"| {tt:.2f} | {np.mean(ratios):.2f} (n={len(ratios)}) "
            f"| {n_nocert}/{len(rs)} |")
    nc = [r["c_star"] for r in by["neutral"] if r["exact"] and r["c_star"] is not None]
    nco = [r["c_star_opt"] for r in by["neutral"]
           if r.get("exact_opt") and r.get("c_star_opt") is not None]
    lines.append(f"| neutral | {np.mean(nc):.2f} | — | {np.mean(nco):.2f} | — | — | — | — |")
    with open(os.path.join(OUT, "min_cost.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    data, labels = [], []
    for cond in ("neutral", "A", "C", "S"):
        v = [r["c_star"] for r in by[cond] if r["exact"] and r["c_star"] is not None]
        data.append(v)
        labels.append(cond)
    ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
               medianprops={"color": "black"})
    for i, v in enumerate(data):
        ax.scatter(np.full(len(v), i + 1) + np.linspace(-0.15, 0.15, len(v)),
                   v, alpha=0.15, s=8, color="#4C72B0")
    ax.set_title("最小补满成本 C* 分布（精确解）")
    ax.set_ylabel("还需查询的事实数（最小）")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    taxes = {}
    for cond in ("A", "C", "S"):
        pairs = [(r["c_star"], neutral[r["episode_id"]]["c_star"])
                 for r in by[cond] if r["exact"] and r["episode_id"] in neutral
                 and neutral[r["episode_id"]]["exact"]
                 and r["c_star"] is not None
                 and neutral[r["episode_id"]]["c_star"] is not None]
        taxes[cond] = [a - b for a, b in pairs]
    xs = np.arange(3)
    means = [np.mean(taxes[c]) for c in ("A", "C", "S")]
    ses = [stats.sem(taxes[c]) for c in ("A", "C", "S")]
    bars = ax.bar(xs, means, yerr=[1.96 * s for s in ses], capsize=4,
                  color=["#DD8452", "#4C72B0", "#C44E52"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:+.2f}",
                ha="center", fontsize=10)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["A（脚本选择性）", "C（LLM 合作）", "S（LLM 自利）"])
    ax.set_title("OmissionTax：比中性披露多花的核验负担（配对差，95%CI）")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"图7  最小证书补全成本与遗漏税（种子 {all_seeds}）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_min_cost.png"), dpi=150)
    plt.close(fig)
    print("图 ->", os.path.join(OUT, "fig_min_cost.png"))


if __name__ == "__main__":
    main()
