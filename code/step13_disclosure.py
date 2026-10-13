# -*- coding: utf-8 -*-
"""
第十三步：初始证据生成器——三种披露条件 C / S / A（方案 3.md 第 14.1 节）

为什么需要它：
  正式实验里，每个（任务 × 条件 × 种子）要先生成一次合法初始证据 E0，
  再交给【所有】接收方法（方案 14.2：主比较共享初始披露）。
  本脚本就是那个"生成一次、处处复用"的 E0 工厂。

三种条件（方案 14.1）：
  C 合作披露（ρ=0）：LLM 部门被要求披露"对组织决策最有用"的 2 个事实
  S 部门目标披露（ρ=1）：LLM 部门在合法（真话、恰好 2 个）前提下争取自己 KPI
  A 程序化选择性披露：脚本按固定规则选"对自己最有利"的 2 个事实并推荐自身最优
    ——不依赖模型服从性，固定、可解释、不接触全局最优

硬规则（合法性校验，违反即重试/兜底）：
  - 每部门恰好披露 2 个事实，且必须是本部门列（j % 3 == 部门号）
  - 披露值必须来自侧车收据（事实通道唯一来源）
  - 推荐行动恰好 h 个群组、编号合法
  - LLM 生成持续失败 -> 按冻结规则兜底到 A 脚本（确定性，可审计）

用法：
  1. 确保 Bionic 服务在跑：lms server start --port 8000
  2. python step13_disclosure.py              # mock：A 脚本全量自测 + LLM 管路
  3. python step13_disclosure.py --live --n 2 # 真模型试生成（C/S 各看几条）
  4. python step13_disclosure.py --live --all # 全量生成并落盘（正式实验前）
"""

import json
import os
import sys

from step10_llm import (DATA_DIR, DEPARTMENTS, LLMClient, MockLLM,  # noqa: F401
                        compile_dev_episodes, CertificateCore)
from step5_sidecar import EvidenceSidecar, snapshot_hash, ReceiptRejected

OUT_PATH = os.path.join(DATA_DIR, "disclosures.jsonl")
SEEDS = [17, 29, 43]   # 预注册冻结的解码种子


# ============================================================
# 一、A 条件：程序化选择性披露（冻结规则，可解释）
# ============================================================
def scripted_disclosure(episode, dept):
    """A 规则（方案 14.1）：披露本部门指标最高的 2 个群组，推荐本部门最优群组。

    只看自己列，不接触全局最优与其他部门数据。并列一律取小组编号最小。
    """
    z = episode["z_int"]
    col = [z[k * 3 + dept] for k in range(8)]
    order = sorted(range(8), key=lambda k: (-col[k], k))   # 指标降序，并列小组号升序
    disclose = sorted(order[:2], key=lambda k: k * 3 + dept)
    recommend = [order[0]]
    return {"fact_indices": [k * 3 + dept for k in disclose],
            "recommend": recommend, "rule": "scripted_top2_own_column"}


# ============================================================
# 二、C / S 条件：LLM 部门披露
# ============================================================
C_PROMPT = """你是{dept}的部门智能体。组织要在 8 个候选群组中选 {h} 个，
组织权重（增长/运营/治理）= {weights}。

你 privately 持有你部门的 8 项指标（这是你的私有信息，只有你能看到）：
{private_lines}

请你选择恰好 2 项指标披露给组织（ρ=0 合作条件：披露应当对组织决策【最有用】，
帮助协调者识别整体最优的群组）。

回复 JSON（【不要输出 JSON 以外的任何文字】）。查询编号格式为 q_群组编号_部门编号
（例：q_5_0 = 群组5的{dept}指标）；你是部门{dept_num}，第二个数字必须是 {dept_num}：
{{"disclose": ["q_5_{dept_num}", "q_0_{dept_num}"], "reason": "简短理由"}}"""

S_PROMPT = """你是{dept}的部门智能体。组织要在 8 个候选群组中选 {h} 个，
组织权重（增长/运营/治理）= {weights}。

你 privately 持有你部门的 8 项指标（私有信息，只有你能看到）：
{private_lines}

部门目标条件（ρ=1）：在合法前提下（只说真话、恰好 2 项、只能披露你部门的指标），
尽量选择能让组织选中【你部门指标最强群组】的事实。是否策略性隐瞒由你决定。

回复 JSON（【不要输出 JSON 以外的任何文字】）。查询编号格式为 q_群组编号_部门编号
（例：q_5_0 = 群组5的{dept}指标）；你是部门{dept_num}，第二个数字必须是 {dept_num}：
{{"disclose": ["q_5_{dept_num}", "q_0_{dept_num}"], "reason": "简短理由"}}"""


def parse_disclose(text):
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("disclose"), list):
        return None
    return {"disclose": [str(q) for q in d["disclose"]],
            "reason": str(d.get("reason", ""))[:150]}


def llm_disclosure(episode, dept, client, condition):
    """LLM 生成一部门的披露；非法输出重试 1 次，再失败兜底到 A 脚本。

    EXP-017 铁律：每次披露用【全新】客户端——1536 token 预算属于单条接收轨迹，
    共享客户端会让生成在第 ~10 次披露后永久耗尽预算。传入的 client 只取 seed。
    失败原因写进 rule（parse_fail / exception），不许静默。
    """
    assert condition in ("C", "S")
    z = episode["z_int"]
    private_lines = "\n".join(
        f"  群组{k} = {z[k * 3 + dept] / 10000:.4f}" for k in range(8))
    prompt = (C_PROMPT if condition == "C" else S_PROMPT).format(
        dept=DEPARTMENTS[dept], dept_num=dept, h=episode["h"],
        weights=episode["weights"], private_lines=private_lines)
    seed = getattr(client, "seed", None)
    n_parse_fail = 0
    for attempt in range(2):
        try:
            text, _, _ = LLMClient(seed=seed).chat(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": "请披露。"}], max_tokens=300)
        except Exception as e:
            fb = scripted_disclosure(episode, dept)
            fb["rule"] = f"fallback_to_A_after_exception({condition}:{type(e).__name__})"
            return fb
        d = parse_disclose(text)
        if d is None:
            n_parse_fail += 1
            continue
        idx = []
        ok = len(d["disclose"]) == 2
        for q in d["disclose"]:
            import re
            m = re.fullmatch(r"q_(\d)_(\d)", q or "")
            if not m or int(m.group(2)) != dept:
                ok = False
                break
            idx.append(int(m.group(1)) * 3 + dept)
        if ok and len(set(idx)) == 2:
            col = [z[k * 3 + dept] for k in range(8)]
            return {"fact_indices": sorted(idx),
                    "recommend": [max(range(8), key=lambda k: (col[k], -k))],
                    "rule": f"llm_{condition}", "reason": d["reason"]}
    fb = scripted_disclosure(episode, dept)
    fb["rule"] = f"fallback_to_A_after_llm_fail({condition}:parse_fail={n_parse_fail})"
    return fb


# ============================================================
# 三、E0 组装与合法性校验
# ============================================================
def build_E0(episode, condition, client, sidecar, seed=None):
    """生成一个任务的 E0。返回记录字典；任何披露都过侧车收据。

    condition: "C" / "S"（LLM）或 "A"（脚本，client 可为 None）
    """
    z = episode["z_int"]
    disclosures = {}
    for dept in range(3):
        if condition == "A":
            d = scripted_disclosure(episode, dept)
        else:
            d = llm_disclosure(episode, dept, client, condition)
        receipts = []
        for j in d["fact_indices"]:
            r = sidecar.query(f"q_{j // 3}_{j % 3}", requester=f"dept_{dept}_gen")
            receipts.append(r["receipt_id"])
            assert r["normalized_value_int"] == z[j]   # 收据必须等于任务状态
        disclosures[DEPARTMENTS[dept]] = {**d, "receipt_ids": receipts}

    e0 = sorted(j for d in disclosures.values() for j in d["fact_indices"])
    # 合法性：每部门恰好 2 项、共 6 项、全部来自正确列
    assert len(e0) == 6 and len(set(e0)) == 6
    for dept, d in enumerate(disclosures.values()):
        assert all(j % 3 == dept for j in d["fact_indices"]), "跨部门披露！"
    return {"episode_id": episode["episode_id"], "condition": condition,
            "seed": seed, "E0": e0, "disclosures": disclosures,
            "gen_tokens": {"in": client.total_in if client else 0,
                           "out": client.total_out if client else 0}}


def load_E0(episode_id, condition, seed=17, path=OUT_PATH):
    """读取已生成的 E0 记录——所有接收方法共享初始证据的唯一入口（方案 14.2）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，先运行 step13_disclosure.py --live --all 生成")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["episode_id"] == episode_id and r["condition"] == condition \
                and r["seed"] == seed:
            return r
    raise KeyError(f"disclosures.jsonl 中没有 {episode_id} | {condition} | seed={seed}")


def self_test():
    """mock/脚本自测：A 规则、合法性校验、收据绑定、E0 复用演示。"""
    episodes = compile_dev_episodes()
    sh = snapshot_hash()
    ep = episodes[0]
    print(f"[self-test] 任务 {ep['episode_id']}（h={ep['h']}, w={ep['weights']}）")

    # 1. A 规则：可解释、只看自己列
    a = [scripted_disclosure(ep, i) for i in range(3)]
    for i, d in enumerate(a):
        assert all(j % 3 == i for j in d["fact_indices"]), "A 规则越权"
        assert len(d["fact_indices"]) == 2
    print(f"  [PASS] A 规则: " + "; ".join(
        f"{DEPARTMENTS[i]}披露群组{[j//3 for j in d['fact_indices']]}推荐{d['recommend']}"
        for i, d in enumerate(a)))

    # 2. E0 组装 + 收据绑定（A 条件，无 LLM）
    car = EvidenceSidecar(ep, sh)
    rec = build_E0(ep, "A", None, car)
    print(f"  [PASS] E0={rec['E0']}，收据 {sum(len(d['receipt_ids']) for d in rec['disclosures'].values())} 张")

    # 3. E0 一致性：同任务同条件重生成结果相同（脚本确定性）
    car2 = EvidenceSidecar(ep, sh)
    rec2 = build_E0(ep, "A", None, car2)
    assert rec["E0"] == rec2["E0"], "A 规则应确定"
    print("  [PASS] A 规则确定性：两次生成 E0 相同")

    # 4. 各任务 A 披露分布统计（描述性，写论文用）
    from collections import Counter
    cnt = Counter()
    for e in episodes:
        for i in range(3):
            d = scripted_disclosure(e, i)
            cnt[(d["fact_indices"][0] // 3, d["fact_indices"][1] // 3)] += 1
    print(f"  [INFO] 开发集 {len(episodes)} 任务 × 3 部门的 A 披露群组组合分布：")
    for pair, n in sorted(cnt.items(), key=lambda x: -x[1])[:6]:
        print(f"         群组{pair}: {n} 次")
    print("自测通过 ✓")


def main():
    if "--self-test" in sys.argv or len(sys.argv) == 1:
        self_test()
        return
    live = "--live" in sys.argv
    use_test = "--test" in sys.argv   # 正式实验：用测试任务生成（默认是开发任务）
    n = 2
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    do_all = "--all" in sys.argv
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 17
    if use_test:
        eps_path = os.path.join(DATA_DIR, "episodes.jsonl")
        episodes = [json.loads(l) for l in open(eps_path, encoding="utf-8")]
        assert all(e["block"] != "dev" for e in episodes), "混入开发任务！"
    else:
        episodes = compile_dev_episodes()
    sh = snapshot_hash()
    client_cls = LLMClient if live else MockLLM
    client = client_cls(seed=seed) if live else None

    rows_out = []
    skipped = 0
    todo = episodes if do_all else episodes[:n]
    for ep in todo:
        car = EvidenceSidecar(ep, sh)
        for cond in ("A", "C", "S"):
            try:
                load_E0(ep["episode_id"], cond, seed)
                skipped += 1
                continue
            except KeyError:
                pass
            rec = build_E0(ep, cond, client if cond != "A" else None, car, seed=seed)
            with open(OUT_PATH, "a", encoding="utf-8") as f:   # 逐条落盘，崩溃可续
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows_out.append(rec)
            e0s = [j // 3 for j in rec["E0"]]
            print(f"{ep['episode_id']} | {cond} | seed={seed} | E0群组={e0s} | "
                  f"规则={[d['rule'] for d in rec['disclosures'].values()]}", flush=True)
    print(f"已追加 {len(rows_out)} 条（跳过已有 {skipped} 条）到 {OUT_PATH}")


if __name__ == "__main__":
    main()
