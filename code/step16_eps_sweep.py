# -*- coding: utf-8 -*-
"""
第十六步：epsilon 标定扫描（论文 6.4 节的回应：认证率为何这么低）

对 ODC-WC / ODC-CS / Width-Cert 在 7 个 epsilon 档位重跑全部确定性轨迹
（240 任务 × 3 条件 × seed 17 共享 E0，预算 24，检查点回放），回答：
  1. 认证率随 epsilon 怎么变？多大的证书才有实际放行意义？
  2. 被认证的行动真实后悔是否都不超过 epsilon？（命题一的实证体检）

输出：
  data/formal_eps_sweep.jsonl      原始行
  analysis/eps_sweep.md            汇总表
  analysis/fig_eps_sweep.png       认证率-epsilon 曲线
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step6_certificate import CertificateCore
from step8_methods import (query_odc_wc, query_odc_cs, query_width,
                           replay_at_budget, BUDGETS)
from step13_disclosure import load_E0

DATA = "data"
OUT = "analysis"
SEED = 17
EPS_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
METHODS = {"ODC-WC": query_odc_wc, "ODC-CS": query_odc_cs, "Width-Cert": query_width}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def run_traj(ep, init, query_fn, epsilon, budget=24):
    """与 step8.run_certified 相同，但 epsilon 可参数化。"""
    z = ep["z_int"]
    core = CertificateCore(ep, epsilon=epsilon)
    cost = 0
    traj = []
    for j in init:
        core.apply_receipt(j, z[j])
        cost += 1
    while True:
        a, r, ok = core.certify()
        traj.append({"cost": cost, "action": a, "regret_bound": r,
                     "certified": ok})
        if ok or cost >= budget:
            break
        j = query_fn(core)
        if j is None:
            break
        core.apply_receipt(j, z[j])
        cost += 1
    return traj


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = [json.loads(l) for l in open(os.path.join(DATA, "episodes.jsonl"),
                                       encoding="utf-8")]
    assert all(e["block"] != "dev" for e in eps)

    seeds = ([int(sys.argv[sys.argv.index("--seed") + 1])] if "--seed" in sys.argv
             else [17, 29, 43])

    raw_path = os.path.join(DATA, "formal_eps_sweep.jsonl")
    done = set()
    if os.path.exists(raw_path):
        for l in open(raw_path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["episode_id"], r["condition"], r["method"],
                      r["epsilon"], r["budget"], r.get("seed", 17)))

    n_new = 0
    with open(raw_path, "a", encoding="utf-8") as f:
        for seed in seeds:
            for ep in eps:
                for cond in ("A", "C", "S"):
                    e0 = load_E0(ep["episode_id"], cond, seed)["E0"]
                    for m, qfn in METHODS.items():
                        for eps_ in EPS_GRID:
                            traj = run_traj(ep, e0, qfn, eps_)
                            eval_core = CertificateCore(ep)
                            z = ep["z_int"]
                            for B in BUDGETS:
                                key = (ep["episode_id"], cond, m, eps_, B, seed)
                                if key in done:
                                    continue
                                step = replay_at_budget(traj, B)
                                if step is None:
                                    continue
                                tr = eval_core.true_regret(step["action"], z) / 100000
                                f.write(json.dumps({
                                    "episode_id": ep["episode_id"], "condition": cond,
                                    "method": m, "seed": seed, "epsilon": eps_,
                                    "budget": B, "action": step["action"],
                                    "true_regret": round(tr, 4),
                                    "regret_bound": round(step["regret_bound"] / 100000, 4),
                                    "certified": step["certified"],
                                    "cost": step["cost"]}, ensure_ascii=False) + "\n")
                                n_new += 1
    print(f"新增 {n_new} 行 -> {raw_path}")

    # ---------- 汇总（全部种子） ----------
    rows = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
    all_seeds = sorted({r.get("seed", 17) for r in rows})
    lines = [f"# ε 标定扫描汇总（种子 {all_seeds}，n={240 * len(all_seeds)}/格）\n",
             "| ε | 方法 | B | 认证率 | 认证者均真实后悔 | 认证者最大后悔 | 全体均后悔 | 有效性违规 |",
             "|---|---|---|---|---|---|---|---|"]
    fig_data = {}
    for eps_ in EPS_GRID:
        for m in METHODS:
            for B in [12, 24]:
                sub = [r for r in rows if r["epsilon"] == eps_
                       and r["method"] == m and r["budget"] == B]
                cert = [r for r in sub if r["certified"]]
                viol = sum(1 for r in cert if r["true_regret"] > eps_ + 1e-9)
                rate = len(cert) / len(sub) if sub else 0
                fig_data[(m, B)] = fig_data.get((m, B), []) + [(eps_, rate)]
                lines.append(
                    f"| {eps_} | {m} | {B} | {rate:.3f} | "
                    f"{np.mean([r['true_regret'] for r in cert]) if cert else float('nan'):.4f} | "
                    f"{max((r['true_regret'] for r in cert), default=float('nan')):.4f} | "
                    f"{np.mean([r['true_regret'] for r in sub]):.4f} | {viol} |")
    with open(os.path.join(OUT, "eps_sweep.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("汇总 ->", os.path.join(OUT, "eps_sweep.md"))

    # 有效性体检
    bad = [r for r in rows if r["certified"] and r["true_regret"] > r["epsilon"] + 1e-9]
    print(f"有效性体检：被认证但真实后悔>ε 的行数 = {len(bad)}（应为 0）")

    # ---------- 图：认证率-ε 曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    style = {"ODC-WC": "-o", "ODC-CS": "-s", "Width-Cert": "-^"}
    for ax, B in zip(axes, [12, 24]):
        for m, mk in style.items():
            pts = sorted(fig_data[(m, B)])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, mk, label=m, ms=5)
        ax.set_xscale("log")
        ax.set_xticks(EPS_GRID)
        ax.set_xticklabels([str(e) for e in EPS_GRID])
        ax.set_xlabel("ε（放行阈值）")
        ax.set_title(f"预算 B={B}")
        ax.grid(alpha=0.3)
        ax.axvline(0.05, ls="--", c="gray", lw=1)
    axes[0].set_ylabel("认证率（预算内取得 ε-证书的比例）")
    axes[0].legend()
    axes[1].text(0.052, 0.05, "预注册 ε=0.05", fontsize=9, color="gray",
                 rotation=90)
    fig.suptitle(f"图5  认证率随 ε 的变化（种子 {all_seeds}，条件 A/C/S 合并，n={720 * len(all_seeds)}/点）",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_eps_sweep.png"), dpi=150)
    plt.close(fig)
    print("图 ->", os.path.join(OUT, "fig_eps_sweep.png"))


if __name__ == "__main__":
    main()
