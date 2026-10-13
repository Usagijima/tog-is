# -*- coding: utf-8 -*-
"""
第六步：决策证书核心（方案 3.md 第 9、10、11 节）——整个研究的心脏

它回答一个问题："现在的信息，够不够放心拍板？"

机制：
  1. 每个未披露的事实维护一个区间 [下界, 上界]，已核验的事实缩成一点
  2. 对当前候选行动 a，算"最坏情况下竞争对手最多能赢多少" = 后悔上界 R(a|E)
  3. 选后悔上界最小的行动；若它 <= ε（0.05）就盖章"已认证"放行
  4. 不达标就按"多反例覆盖"启发式花钱查最关键的事实，再重算

数学保证（命题一）：只要真实状态还在区间盒里，被认证行动的真实后悔 <= ε。
全程用整数运算（权重×10、指标×10000），杜绝浮点误差错误放行。

自测（对应方案第 26 节）：
    python step6_certificate.py
"""

import itertools
import json
import os
import random

DATA_DIR = "data"
EPSILON = 0.05           # 预注册阈值（方案 9.3 节）
W_SCALE = 10             # 权重 (0.5,0.3,0.2) -> 整数 (5,3,2)
Z_SCALE = 10000          # 指标量化网格（step4 已用）
SEED = 17


def weights_to_int(w):
    return [int(round(x * W_SCALE)) for x in w]


class CertificateCore:
    """盒状不确定性下的精确整数后悔证书 + 多反例覆盖查询。"""

    def __init__(self, episode, epsilon=EPSILON):
        self.ep = episode
        self.h = episode["h"]
        self.K = 8
        self.d = 24
        self.actions = list(itertools.combinations(range(self.K), self.h))
        self.theta = self._coefficients()          # 每个行动在各事实上的整数系数
        # 【整数尺度约定】所有后悔量保持 h 未约分的形式（scale = W_SCALE × Z_SCALE），
        # 不在中途做整数除法 —— 保证方案 12.1 节的恒等式 V(E∪S) = V(E) - ΣD
        # 在整数世界严格成立（2026-10-04 第二次修正，见 EXP-007）。
        self.eps_scaled = int(round(epsilon * W_SCALE * Z_SCALE))
        # 区间从定义域 [0, Z_SCALE] 开始；只有收据能收紧它
        self.lo = [0] * self.d
        self.hi = [Z_SCALE] * self.d
        self.revealed = [False] * self.d

    def _coefficients(self):
        """行动 a 对事实 (k,i) 的系数 = w_i / h（整数尺度下乘 W_SCALE*Z_SCALE 的公共
        分母在比较时可约掉，这里只存 w_i 的整数形式，打分再统一除以 h）。"""
        w = weights_to_int(self.ep["weights"])
        theta = []
        for a in self.actions:
            t = [0] * self.d
            for k in a:
                for i in range(3):
                    t[k * 3 + i] = w[i]
            theta.append(t)
        return theta

    # ---------- 后悔计算（第 9.2 节闭式）----------
    def pair_worst(self, a_idx, b_idx):
        """竞争行动 b 相对 a 的最坏优势（整数尺度，h 未约分）。"""
        ta, tb = self.theta[a_idx], self.theta[b_idx]
        total = 0
        for j in range(self.d):
            d = tb[j] - ta[j]
            if d > 0:
                total += d * self.hi[j]
            elif d < 0:
                total += d * self.lo[j]
        return total

    def regret(self, a_idx):
        return max(self.pair_worst(a_idx, b) for b in range(len(self.actions)))

    def best_action(self):
        rs = [self.regret(a) for a in range(len(self.actions))]
        a = min(range(len(rs)), key=lambda i: (rs[i], i))  # 并列按冻结 ID 次序
        return a, rs[a]

    # ---------- 证据收紧（第 9.1 节）----------
    def apply_receipt(self, fact_index, value_int):
        if not (0 <= value_int <= Z_SCALE):
            raise ValueError("收据值越界，侧车应已拒绝")
        self.lo[fact_index] = self.hi[fact_index] = value_int
        self.revealed[fact_index] = True

    # ---------- 多反例覆盖查询（第 10 节）----------
    def coverage_scores(self):
        a_star, r_min = self.best_action()
        if r_min <= self.eps_scaled:
            return None  # 已认证，无需查询
        scores = {}
        for j in range(self.d):
            if self.revealed[j]:
                continue
            width = self.hi[j] - self.lo[j]
            s = 0
            for b in range(len(self.actions)):
                gamma = self.pair_worst(a_star, b) - self.eps_scaled
                if gamma <= 0:
                    continue
                d = abs(self.theta[b][j] - self.theta[a_star][j])
                s += min(gamma, d * width)
            scores[j] = s
        return scores

    def next_query(self):
        scores = self.coverage_scores()
        if scores is None:
            return None
        unknown = [j for j in scores if self.hi[j] > self.lo[j]]
        if not unknown:
            return None
        # 并列按冻结次序（事实编号）；全零分时选目录首个未知项保底（第 10.3 节）
        return min(unknown, key=lambda j: (-scores[j], j))

    # ---------- 放行判定（第 9.3 节）----------
    def certify(self):
        a, r = self.best_action()
        return (a, r, r <= self.eps_scaled)  # (行动, 后悔上界, 是否认证)

    def true_regret(self, a_idx, z_int):
        """离线评测专用：行动 a 在真实状态下的后悔（整数尺度，h 未约分）。"""
        sa = sum(t * z for t, z in zip(self.theta[a_idx], z_int))
        sbest = max(sum(t * z for t, z in zip(self.theta[b], z_int)) for b in range(len(self.actions)))
        return sbest - sa


def run_certificate(core, get_value, initial_facts, budget):
    """通用证书流程（方案 10.4 节伪代码）。

    get_value: 事实编号 -> 真实量化值（在线由侧车收据提供，离线测试直接给）
    initial_facts: 初始披露的事实编号列表（6 个，成本已计入）
    budget: 总查询预算（含初始 6 个）
    返回 anytime 轨迹：各预算检查点上的 (行动, 后悔上界, 是否认证, 已花成本)
    """
    cost = 0
    trajectory = []
    for j in initial_facts:
        core.apply_receipt(j, get_value(j))
        cost += 1
    while True:
        a, r, ok = core.certify()
        trajectory.append({"cost": cost, "action": a, "regret_bound": r, "certified": ok})
        if ok or cost >= budget:
            break
        j = core.next_query()
        if j is None:
            trajectory.append({"cost": cost, "action": a, "regret_bound": r, "certified": ok, "abstain": True})
            break
        core.apply_receipt(j, get_value(j))
        cost += 1
    return trajectory


# ============================================================
# 自测：对应方案第 26 节验证清单
# ============================================================
def self_test(n_problems=300):
    rng = random.Random(SEED)
    eps = [json.loads(l) for l in open(os.path.join(DATA_DIR, "episodes.jsonl"), encoding="utf-8")]
    ok_count = {"cert_safe": 0, "monotone": 0, "exact_when_full": 0, "heuristic_beats_uniform": 0}

    for t in range(n_problems):
        ep = rng.choice(eps)
        z = ep["z_int"]
        core = CertificateCore(ep)
        # 随机证据：6 到 14 个事实
        n_init = rng.randint(6, 14)
        facts = rng.sample(range(24), n_init)
        for j in facts:
            core.apply_receipt(j, z[j])

        # 检查 1：被认证行动的真实后悔必须 <= ε（命题一的实现正确性）
        a, r, ok = core.certify()
        if ok:
            tr = core.true_regret(a, z)
            if tr <= core.eps_scaled:
                ok_count["cert_safe"] += 1
            else:
                print(f"  [FAIL] 认证不安全: {ep['episode_id']} true={tr} bound={r}")
        else:
            ok_count["cert_safe"] += 1  # 未认证不涉及安全承诺

        # 检查 2：单调性（命题二）——再加一个有效证据，最小后悔上界不增
        r_before = core.best_action()[1]
        hidden = [j for j in range(24) if not core.revealed[j]]
        if hidden:
            core2 = CertificateCore(ep)
            for j in facts:
                core2.apply_receipt(j, z[j])
            core2.apply_receipt(hidden[0], z[hidden[0]])
            if core2.best_action()[1] <= r_before:
                ok_count["monotone"] += 1
        else:
            ok_count["monotone"] += 1

        # 检查 3：全部揭示后必须能得到 R=0 的行动（命题四）
        core3 = CertificateCore(ep)
        for j in range(24):
            core3.apply_receipt(j, z[j])
        _, r3, ok3 = core3.certify()
        if ok3 and r3 == 0:
            ok_count["exact_when_full"] += 1
        else:
            print(f"  [FAIL] 全量揭示未达 R=0: {ep['episode_id']} r={r3}")

        # 检查 4：多反例启发式 vs 均匀查询——谁的认证成本更低
        if not ok and hidden:
            def run_with(policy):
                c = CertificateCore(ep)
                cost = 0
                for j in facts:
                    c.apply_receipt(j, z[j])
                    cost += 1
                while cost < 24:
                    _, r, done = c.certify()
                    if done:
                        return cost
                    unk = [j for j in range(24) if not c.revealed[j]]
                    j = c.next_query() if policy == "wc" else rng.choice(unk)
                    c.apply_receipt(j, z[j])
                    cost += 1
                _, r, done = c.certify()
                return cost if done else 99
            if run_with("wc") <= run_with("uniform"):
                ok_count["heuristic_beats_uniform"] += 1

    print("===== 证书核心自测 =====")
    print(f"  cert_safe: {ok_count['cert_safe']}/{n_problems}")
    print(f"  monotone: {ok_count['monotone']}/{n_problems}")
    print(f"  exact_when_full: {ok_count['exact_when_full']}/{n_problems}")
    # 启发式不保证每次都赢（方案 10.3 节明确不声称近似比），看胜率
    win_rate = ok_count["heuristic_beats_uniform"] / n_problems
    print(f"  heuristic_win_rate: {win_rate:.1%}（不追求 100%，显著过半即健康）")
    all_ok = (ok_count["cert_safe"] == n_problems and ok_count["monotone"] == n_problems
              and ok_count["exact_when_full"] == n_problems and win_rate >= 0.5)
    print("全部通过 ✓" if all_ok else "存在失败项 ✗")

    # 演示：一个真实任务上的 anytime 轨迹（论文素材）
    demo = CertificateCore(eps[0])
    traj = run_certificate(demo, lambda j: eps[0]["z_int"][j], initial_facts=[0, 3, 6, 9, 12, 15], budget=24)
    os.makedirs(os.path.join(DATA_DIR, "artifacts"), exist_ok=True)
    with open(os.path.join(DATA_DIR, "artifacts", "certificate_demo.json"), "w", encoding="utf-8") as f:
        json.dump({"episode": eps[0]["episode_id"], "trajectory": traj,
                   "epsilon": EPSILON}, f, ensure_ascii=False, indent=2)
    print(f"\n演示轨迹已存 data/artifacts/certificate_demo.json（{eps[0]['episode_id']}）")
    return all_ok


if __name__ == "__main__":
    self_test()
