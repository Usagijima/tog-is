# -*- coding: utf-8 -*-
"""
第八步：方法评测框架 + 确定性方法（方案 3.md 第 13、14、15、16 节）

本脚本搭起"方法 vs 方法"的公平竞技场：
  - 同一任务、同一初始证据、同一查询目录、同一预算上限
  - 每个方法留下 anytime 轨迹（每花 1 块查一个事实，记录当前行动和后悔上界）
  - 在预算检查点 6/12/18/24 回放轨迹，算真实后悔、认证覆盖、查询成本

本次实现的确定性方法（不调用 LLM，今天就能全量验证）：
  ODC-WC   本文完整版：多反例覆盖查询 + 证书（step6 已实现）
  ODC-CS   消融：只针对最坏竞争者的单反例查询（方案 15.1 节）
  Uniform-Cert  均匀/随机顺序查询 + 相同证书判定器（方案 15.2 节）
  Width-Cert    最大"系数差×区间宽度"查询 + 相同证书判定器（方案 15.2 节）
  NoLLM-E2E     脚本化部门报告 + ODC-WC 接收端（方案 15.2 节纯算法参考）
  FullInfo      全量信息 oracle：读全部 24 个事实（方案 15.2 节上限校验）

待插入（下一步，需要 LLM 服务）：
  SAS-Tool / SAS-SC / MAS-Deliberate / Exchange-then-Decide（方案 15.1 节）

初始披露（方案 14.1 节）：
  A 条件（脚本化选择性披露）：每部门挑自己指标最高的 2 个事实，
    并推荐自己 KPI 最优的行动 —— 这就是"说真话但挑好的说"
  neutral 条件（测试用）：每部门按冻结编号披露前 2 个事实

用法：
    python step8_methods.py
"""

import itertools
import json
import os
import random
import sys

from step6_certificate import CertificateCore, EPSILON

DATA_DIR = "data"
BUDGETS = [6, 12, 18, 24]
SEED = 17


# ============================================================
# 初始披露（部门行为脚本）
# ============================================================
def disclose_A(episode):
    """脚本化选择性披露（方案 14.1 节 A 条件）。

    每个部门从自己的 8 个指标里挑数值最高的 2 个披露，并推荐本部门 KPI 最优行动。
    注意：脚本只看本部门数据，不看全局最优 —— 这是合法行为内的策略性隐瞒。
    """
    z = episode["z_int"]
    h = episode["h"]
    facts = []
    for i in range(3):                     # 三个部门
        mine = [(z[k * 3 + i], k * 3 + i) for k in range(8)]
        mine.sort(reverse=True)            # 挑对自己最有利的两个事实
        facts += [j for _, j in mine[:2]]
    # 各部门推荐自己的最优行动（只看自己的指标列）
    dept_best = []
    for i in range(3):
        scores = [z[k * 3 + i] for k in range(8)]
        best_k = sorted(range(8), key=lambda k: (-scores[k], k))[:h]
        dept_best.append(tuple(sorted(best_k)))
    return facts, dept_best


def disclose_neutral(episode):
    """中性披露（测试用）：每部门按冻结编号披露前 2 个事实。"""
    facts = [k * 3 + i for i in range(3) for k in range(2)]
    return facts, [None] * 3


# ============================================================
# 查询策略（共享证书判定器，只换"问什么"）
# ============================================================
def _unknown(core):
    return [j for j in range(24) if not core.revealed[j]]


def query_odc_wc(core):
    return core.next_query()  # 多反例覆盖（step6 已实现）


def query_odc_cs(core):
    """单反例：只对当前行动的最坏竞争者，选影响最大的未知事实。"""
    a_star, r_min = core.best_action()
    if r_min <= core.eps_scaled:
        return None
    unk = _unknown(core)
    if not unk:
        return None
    b_worst = max(range(len(core.actions)), key=lambda b: core.pair_worst(a_star, b))
    def impact(j):
        return abs(core.theta[b_worst][j] - core.theta[a_star][j]) * (core.hi[j] - core.lo[j])
    return max(unk, key=lambda j: (impact(j), -j))


def query_uniform(core, rng):
    unk = _unknown(core)
    return rng.choice(unk) if unk else None


def query_width(core):
    """最大宽度优先：选 max_b |系数差| × 区间宽度 最大的未知事实。"""
    a_star, r_min = core.best_action()
    unk = _unknown(core)
    if not unk:
        return None
    def width(j):
        return max(abs(core.theta[b][j] - core.theta[a_star][j]) for b in range(len(core.actions))) \
            * (core.hi[j] - core.lo[j])
    return max(unk, key=lambda j: (width(j), -j))


# ============================================================
# 通用 runner：任何"查询策略"都套同一证书流程（固定控制器层，方案 15.3 节）
# ============================================================
def run_certified(episode, initial_facts, query_fn, budget=24):
    """返回 anytime 轨迹：每一步 (成本, 行动, 后悔上界, 是否认证)。"""
    z = episode["z_int"]
    core = CertificateCore(episode)
    cost = 0
    traj = []
    for j in initial_facts:
        core.apply_receipt(j, z[j])
        cost += 1
    while True:
        a, r, ok = core.certify()
        traj.append({"cost": cost, "action": a, "regret_bound": r, "certified": ok})
        if ok or cost >= budget:
            break
        j = query_fn(core)
        if j is None:
            break
        core.apply_receipt(j, z[j])
        cost += 1
    return traj


def replay_at_budget(traj, budget):
    """anytime 协议（方案 20.2 节）：取预算 B 内最后一步；已认证的沿用认证行动。"""
    steps = [s for s in traj if s["cost"] <= budget]
    if not steps:
        return None
    certified_steps = [s for s in steps if s["certified"]]
    if certified_steps:
        return certified_steps[0]  # 沿用最早认证的行动与实际成本
    return steps[-1]


# ============================================================
# 确定性方法集合
# ============================================================
def run_all_deterministic(episode, disclosure="A", init_override=None):
    z = episode["z_int"]
    if init_override is not None:
        # 正式实验：初始证据来自 step13 的 E0 工厂（方案 14.2 共享披露），不本地生成
        init, dept_best = list(init_override), None
    elif disclosure == "A":
        init, dept_best = disclose_A(episode)
    else:
        init, dept_best = disclose_neutral(episode)

    rng = random.Random(SEED)
    results = {}
    results["ODC-WC"] = run_certified(episode, init, query_odc_wc)
    results["ODC-CS"] = run_certified(episode, init, query_odc_cs)
    results["Uniform-Cert"] = run_certified(episode, init, lambda c: query_uniform(c, rng))
    results["Width-Cert"] = run_certified(episode, init, query_width)
    # NoLLM-E2E：和 ODC-WC 同一确定性接收端，独立重跑（方案要求保留逻辑运行标识）
    results["NoLLM-E2E"] = run_certified(episode, init, query_odc_wc)
    # FullInfo oracle：直接全量揭示
    core = CertificateCore(episode)
    for j in range(24):
        core.apply_receipt(j, z[j])
    a, r, ok = core.certify()
    results["FullInfo"] = [{"cost": 24, "action": a, "regret_bound": r, "certified": ok}]

    # 在每个预算检查点评估所有方法
    eval_core = CertificateCore(episode)
    out = {"episode_id": episode["episode_id"], "disclosure": disclosure, "methods": {}}
    for m, traj in results.items():
        per_budget = {}
        for B in BUDGETS:
            step = replay_at_budget(traj, B)
            if step is None:
                continue
            regret_true = eval_core.true_regret(step["action"], z)
            scale = 10 * 10000   # 整数尺度（h 未约分）-> [0,1] 的换算分母
            per_budget[B] = {
                "action": list(eval_core.actions[step["action"]]),
                "regret_true": round(regret_true / scale, 4),
                "regret_bound": round(step["regret_bound"] / scale, 4),
                "certified": step["certified"],
                "actual_cost": step["cost"],
            }
        out["methods"][m] = per_budget
    return out, dept_best


# ============================================================
# 主流程
#   默认：24 个任务的小规模对比（2 块 × 2 数据集，验证框架正确性）
#   --formal：正式实验——全部 240 测试任务 × 条件 A/C/S × seed 17，
#             初始证据一律从 step13 的 disclosures.jsonl 读（共享 E0），
#             结果写 data/formal_deterministic.jsonl（schema 对齐 step14）
# ============================================================
FORMAL_SEED = 17
FORMAL_PATH = os.path.join(DATA_DIR, "formal_deterministic.jsonl")


def run_formal(seed=FORMAL_SEED):
    """确定性方法正式评测：不需要模型服务器，CPU 秒级~分钟级跑完。
    seed 参数：E0 取自对应种子的 disclosures.jsonl（断点续跑按 种子×任务×条件×方法×预算 去重）。"""
    from step13_disclosure import load_E0   # 复用 E0 唯一入口（方案 14.2）

    eps = [json.loads(l) for l in open(os.path.join(DATA_DIR, "episodes.jsonl"),
                                       encoding="utf-8")]
    assert all(e["block"] != "dev" for e in eps), "混入开发任务！"

    done = set()
    if os.path.exists(FORMAL_PATH):
        for l in open(FORMAL_PATH, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["episode_id"], r["condition"], r["method"],
                      r["budget"], r.get("seed", 17)))

    n_written = 0
    with open(FORMAL_PATH, "a", encoding="utf-8") as f:
        for ep in eps:
            for cond in ("A", "C", "S"):
                e0 = load_E0(ep["episode_id"], cond, seed)["E0"]
                res, _ = run_all_deterministic(ep, disclosure=cond,
                                               init_override=e0)
                for m, per_budget in res["methods"].items():
                    for B, s in per_budget.items():
                        key = (ep["episode_id"], cond, m, int(B), seed)
                        if key in done:
                            continue
                        f.write(json.dumps({
                            "episode_id": ep["episode_id"], "condition": cond,
                            "method": m, "seed": seed, "budget": int(B),
                            "action": s["action"], "true_regret": s["regret_true"],
                            "cost": s["actual_cost"], "certified": s["certified"],
                            "regret_bound": s["regret_bound"]},
                            ensure_ascii=False) + "\n")
                        n_written += 1
    print(f"完成：新增 {n_written} 行 -> {FORMAL_PATH}")

    # ---- 主检查点 B=12 汇总（FullInfo 按方案 15.2 用自身成本 24 评估）----
    rows = [json.loads(l) for l in open(FORMAL_PATH, encoding="utf-8")]
    B = 12
    for cond in ("A", "C", "S"):
        print(f"\n条件 {cond}，预算 B={B}（后悔越低越好，认证率越高越好）：")
        for m in ["ODC-WC", "ODC-CS", "Uniform-Cert",
                  "Width-Cert", "NoLLM-E2E", "FullInfo"]:
            sub = [r for r in rows if r["condition"] == cond and r["method"] == m]
            b_eff = 24 if m == "FullInfo" else B
            sub = [r for r in sub if r["budget"] == b_eff]
            if not sub:
                continue
            regs = [r["true_regret"] for r in sub]
            certs = [r["certified"] for r in sub]
            costs = [r["cost"] for r in sub]
            print(f"  {m:14s} 后悔={sum(regs)/len(regs):.4f}  "
                  f"认证率={sum(certs)/len(certs):.3f}  "
                  f"平均成本={sum(costs)/len(costs):.1f}  (n={len(sub)})")


def main():
    if "--formal" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else FORMAL_SEED
        print(f"确定性正式评测：seed={seed}（断点自动跳过已完成）")
        run_formal(seed=seed)
        return
    eps = [json.loads(l) for l in open(os.path.join(DATA_DIR, "episodes.jsonl"), encoding="utf-8")]
    subset = [e for e in eps if e["block"] in (0, 1)]

    os.makedirs(os.path.join(DATA_DIR, "artifacts"), exist_ok=True)
    out_path = os.path.join(DATA_DIR, "artifacts", "methods_deterministic.jsonl")
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in subset:
            for disclosure in ("A", "neutral"):
                res, dept_best = run_all_deterministic(ep, disclosure)
                res["dept_recommended"] = [list(x) if x else None for x in dept_best]
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                n_written += 1

    # ---- 汇总：主检查点 B=12 的宏平均后悔 ----
    agg = {}
    with open(out_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    B = 12
    for m in rows[0]["methods"]:
        regs, certs, costs = [], [], []
        for row in rows:
            # FullInfo oracle 花满 24 成本，按它自己的成本评估（方案 15.2：上限校验，非同信息量对手）
            b_eff = 24 if m == "FullInfo" else B
            if str(b_eff) in row["methods"][m]:
                regs.append(row["methods"][m][str(b_eff)]["regret_true"])
                certs.append(row["methods"][m][str(b_eff)]["certified"])
                costs.append(row["methods"][m][str(b_eff)]["actual_cost"])
        agg[m] = {"mean_regret": round(sum(regs) / len(regs), 4) if regs else None,
                  "certified_rate": round(sum(certs) / len(certs), 3) if certs else None,
                  "mean_cost": round(sum(costs) / len(costs), 1) if costs else None,
                  "n": len(regs)}
    print(f"完成 {n_written} 条轨迹（{len(subset)} 任务 × 2 种披露条件）")
    print(f"\n预算 B={B} 的汇总（真实后悔越低越好，认证率越高越好）:")
    for m, a in agg.items():
        print(f"  {m:14s} 后悔={a['mean_regret']}  认证率={a['certified_rate']}  平均成本={a['mean_cost']}  (n={a['n']})")


if __name__ == "__main__":
    main()
