# -*- coding: utf-8 -*-
"""
第九步：最小证书补全成本 oracle（方案 3.md 第 12 节）

回答一个问题："从现在开始，最少还要花多少钱查事实，才能认证某个行动？"

意义（方案 12.3）：
  只看实际查询数，分不清"这题确实需要很多事实"和"协议问错了问题"。
  这个 oracle 算的是【本证书类别下的理论最小成本】，用来：
  - CostRatio = 实际花费 / 理论最小 → 衡量查询启发式问得准不准
  - OmissionTax = 选择性披露的最小成本 - 中性披露的最小成本
    → 即使句句属实，隐瞒本身造成了多少额外治理负担

严格的计算规则（方案 12.2）：
  - 得知真实值 z* 后，揭示事实 j 使 V_ab 恰好减少 D_abj（精确恒等式，自测验证）
  - 排除"即使全披露真实后悔也 > ε"的行动
  - 每个固定行动 a 的补满是覆盖问题：选最少的未知事实使所有竞争者约束满足
  - 生产用 MILP 求解器；本原型规模小（≤18 个未知事实），按子集大小升序枚举精确求解
  - 超时/搜不到就如实报告"未解决"，绝不把超时可行解当精确最优（方案红线）

用法：
    python step9_min_cost.py
"""

import itertools
import json
import os

import numpy as np

from step6_certificate import CertificateCore, EPSILON
from step8_methods import disclose_A, disclose_neutral, run_certified, query_odc_wc

DATA_DIR = "data"
SEARCH_CAP = 8  # 补满集合大小搜索上限；超过则报告"未解决（>8）"


# ============================================================
# 精确减少量与恒等式自测（方案 12.1）
# ============================================================
def reduction_matrix(core, z):
    """D[a][b][j]：得知 z_j 真实值后 V_ab 的精确减少量（整数尺度）。"""
    nA, d = len(core.actions), core.d
    D = np.zeros((nA, nA, d), dtype=np.int64)
    for a in range(nA):
        for b in range(nA):
            for j in range(d):
                dlt = core.theta[b][j] - core.theta[a][j]
                worst = max(dlt * core.lo[j], dlt * core.hi[j])
                D[a, b, j] = worst - dlt * z[j]
    return D


def self_test_identity(n=100, seed=17):
    """恒等式 V(E∪S) = V(E) - Σ_{j∈S} D_abj（方案 12.1）。"""
    rng = np.random.default_rng(seed)
    eps = [json.loads(l) for l in open(os.path.join(DATA_DIR, "episodes.jsonl"), encoding="utf-8")]
    for _ in range(n):
        ep = eps[int(rng.integers(len(eps)))]
        z = ep["z_int"]
        core = CertificateCore(ep)
        D = reduction_matrix(core, z)
        a, b = int(rng.integers(len(core.actions))), int(rng.integers(len(core.actions)))
        hidden = [j for j in range(24) if not core.revealed[j]]
        S = rng.choice(hidden, size=int(rng.integers(0, 6)), replace=False)
        v_before = core.pair_worst(a, b)
        for j in S:
            core.apply_receipt(j, z[j])
        v_after = core.pair_worst(a, b)
        expect = v_before - int(D[a, b, list(S)].sum())
        if v_after != expect:
            print(f"  [FAIL] 恒等式不成立: {v_after} != {expect}")
            return False
    print(f"恒等式自测通过（{n} 组随机检查）✓")
    return True


# ============================================================
# 最小补全成本 C_ε^*(E)（方案 12.2）
# ============================================================
def min_completion_cost(core, z, cap=SEARCH_CAP):
    """返回 (C*, 是否精确解决)。按子集大小升序枚举。"""
    nA = len(core.actions)
    D = reduction_matrix(core, z)
    eps_s = core.eps_scaled

    # 先排除：即使全量披露，真实后悔仍 > ε 的行动（方案 12.2）
    feasible_actions = []
    for a in range(nA):
        if core.true_regret(a, z) <= eps_s:
            feasible_actions.append(a)

    unknown = [j for j in range(24) if not core.revealed[j]]
    for a in feasible_actions:
        # 每个竞争者需要的覆盖量
        need = np.array([core.pair_worst(a, b) - eps_s for b in range(nA)])
        active = need > 0
        Da = D[a][active][:, unknown]          # 只保留有约束的竞争者和未知事实
        need = need[active]
        if len(need) == 0:
            return 0, True                     # 该行动已可认证
        Da_T = Da.T                             # (未知事实数, 竞争者数)
        # 按子集大小升序：找到最小 k 使 ∃S:|S|=k 且覆盖全部约束
        for k in range(1, min(cap, len(unknown)) + 1):
            for S in itertools.combinations(range(len(unknown)), k):
                if (Da_T[list(S)].sum(axis=0) >= need).all():
                    return k, True
        # 搜到上限还没解：报告未解决（可能真实 C* 更大）
    return cap + 1, False


def diagnose(episode, disclosure):
    """对单个任务：oracle 成本 + ODC-WC 实际成本 → CostRatio。"""
    z = episode["z_int"]
    init, _ = (disclose_A if disclosure == "A" else disclose_neutral)(episode)

    core0 = CertificateCore(episode)
    for j in init:
        core0.apply_receipt(j, z[j])
    c_star, exact = min_completion_cost(core0, z)

    # ODC-WC 实际轨迹（若拿到证书才有 CostRatio）
    traj = run_certified(episode, init, query_odc_wc)
    certified = [s for s in traj if s["certified"]]
    actual_add = certified[0]["cost"] - len(init) if certified else None

    return {
        "episode_id": episode["episode_id"],
        "disclosure": disclosure,
        "c_star": c_star, "exact": exact,
        "c_init": len(init),
        "odc_certified": bool(certified),
        "odc_actual_add": actual_add,
        "cost_ratio": (round((len(init) + actual_add) / (len(init) + c_star), 3)
                       if certified and exact else None),
    }


def main():
    if not self_test_identity():
        return

    eps = [json.loads(l) for l in open(os.path.join(DATA_DIR, "episodes.jsonl"), encoding="utf-8")]
    subset = [e for e in eps if e["block"] == 0]   # 每数据集块 0，共 24 任务

    rows = []
    for ep in subset:
        for disc in ("A", "neutral"):
            r = diagnose(ep, disc)
            rows.append(r)
            tax = None
            print(f"  {r['episode_id']:24s} {disc:7s} "
                  f"C*={r['c_star']}{'' if r['exact'] else '(未解决)'} "
                  f"ODC实际+{r['odc_actual_add']} 比例={r['cost_ratio']}")

    # OmissionTax = C*(A选择性) - C*(中性)（方案 12.4）
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["episode_id"], {})[r["disclosure"]] = r
    taxes, ratios = [], []
    for ep_id, pair in by_ep.items():
        if "A" in pair and "neutral" in pair and pair["A"]["exact"] and pair["neutral"]["exact"]:
            taxes.append(pair["A"]["c_star"] - pair["neutral"]["c_star"])
            if pair["A"]["cost_ratio"]:
                ratios.append(pair["A"]["cost_ratio"])
    print(f"\n===== 汇总（块0，{len(subset)} 任务 × 2 条件）=====")
    print(f"OmissionTax: 均值={np.mean(taxes):+.2f}  范围=[{min(taxes)}, {max(taxes)}]  "
          f"(为正=选择性披露更费钱；方案 12.4 提醒：可以为负)")
    print(f"CostRatio(ODC-WC): 均值={np.mean(ratios):.2f}  (越接近1问得越准；n={len(ratios)})")

    os.makedirs(os.path.join(DATA_DIR, "artifacts"), exist_ok=True)
    with open(os.path.join(DATA_DIR, "artifacts", "min_cost_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows,
                   "omission_tax_mean": float(np.mean(taxes)),
                   "cost_ratio_mean": float(np.mean(ratios))}, f, ensure_ascii=False, indent=2)
    print("明细已存 data/artifacts/min_cost_diagnosis.json")


if __name__ == "__main__":
    main()
