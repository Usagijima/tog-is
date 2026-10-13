# -*- coding: utf-8 -*-
"""
第十二步：Exchange-then-Decide 方法（方案 3.md 第 15.1 节）

参照 HiddenBench 的"交换、提出反对理由、再决定"结构做受限工具适配
（方案明确：保留主动交换与反对当前领先选项的思想；是适配，不冒充原 benchmark 复现）。

流程（相比 MAS-Deliberate 的开放式协商，这里是【固定结构】三轮）：
  1. 交换轮：三个部门各自发言（论证 + 推荐行动 + 提议核验）
  2. 反对轮：公布当前领先选项（推荐众数，并列取小组编号最小），
     每个部门必须表态：反对（给理由+替代）或附议
  3. 决定：协调者看到全部发言，可付费查询一批事实 / 用算分器，然后拍板；
     若查询了一批，再给一个必须拍板的轮次

公平性与红线与 step11 相同：
  - 同一 E0、同一查询目录与成本、同一算分器、token 上限跨全部调用累计
  - 事实纪律：聊天只许引用已核验事实表的数字
  - 提示词只在开发集调；解析失败有限重试 + 冻结兜底

用法：
  1. 确保 Bionic 服务在跑：lms server start --port 8000
  2. python step12_exchange.py            # mock 自测
  3. python step12_exchange.py --live --n 4
"""

import json
import os
import re
import sys

from step10_llm import (DATA_DIR, DEPARTMENTS,  # noqa: F401
                        LLMClient, MockLLM, BudgetExceeded,
                        compile_dev_episodes, parse_decision, CertificateCore,
                        SYS_PROMPT, _point_estimate, resolve_init, run_seed_from)
from step11_mas import DEPT_PROMPT, parse_dept_speech


# ============================================================
# 一、反对轮提示词
# ============================================================
OBJECTION_PROMPT = """你是{dept}的代表。组织决策协商进入【反对轮】：当前领先选项是 {leader}。
已核验事实表：
{facts}

规则：
1. 审查领先选项：它是否被已核验事实支持？有没有被忽视的短板？
2. 表态：反对（给出理由和更好的替代行动）或附议。
3. 【事实纪律】只能引用已核验事实表的数字，不得编造数值。

回复 JSON（【不要输出 JSON 以外的任何文字】）：
{{"think": "简短理由", "stance": "object" 或 "support",
  "argument": "一句话理由（给协调者看）", "alternative": [群组编号...]}}
stance=support 时 alternative 填当前领先选项；stance=object 时填你推荐的替代（恰好 {h} 个）。"""


def parse_objection(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    stance = str(d.get("stance", "")).lower()
    alt = d.get("alternative")
    try:
        alt = [int(x) for x in alt] if isinstance(alt, list) else []
    except (TypeError, ValueError):
        alt = []   # 非数字群组号视为未给替代（EXP-015）
    return {"stance": "object" if stance == "object" else "support",
            "argument": str(d.get("argument", ""))[:200], "alternative": alt}


# ============================================================
# 二、Exchange-then-Decide 主流程
# ============================================================
def run_etd(episode, initial_facts, client, query_budget=24):
    """返回与 run_sas_tool 同构的结果字典（多 leader / objection 记录）。"""
    z = episode["z_int"]
    core = CertificateCore(episode)
    facts_lines, log, queries_made = [], [], []
    parse_failures = 0
    cost = 0
    for j in initial_facts:
        core.apply_receipt(j, z[j])
        cost += 1
        k, i = j // 3, j % 3
        facts_lines.append(f"  {DEPARTMENTS[i]}·群组{k} = {z[j] / 10000:.4f}")

    def current_prompt_parts():
        unknown = [f"q_{k}_{i}" for k in range(8) for i in range(3)
                   if not core.revealed[k * 3 + i]]
        facts = "\n".join(facts_lines) if facts_lines else "  （无）"
        catalog = ", ".join(unknown) if unknown else "（无，必须拍板）"
        return facts, catalog

    def valid_action(a):
        return (len(a) == episode["h"] and len(set(a)) == episode["h"]
                and all(0 <= k < 8 for k in a))

    # ---------- 轮 1：交换 ----------
    exchange = {}
    for i in range(3):
        facts, catalog = current_prompt_parts()
        prompt = DEPT_PROMPT.format(dept=DEPARTMENTS[i], h=episode["h"],
                                    weights=episode["weights"], facts=facts,
                                    cost=cost, catalog=catalog)
        try:
            text, _, _ = client.chat([{"role": "system", "content": prompt},
                                      {"role": "user", "content": "请发言。"}],
                                     max_tokens=400)
        except BudgetExceeded:
            log.append("[budget] 交换轮预算耗尽，转入决定")
            exchange = {}
            break
        speech = parse_dept_speech(text)
        if speech is None:
            log.append(f"[dept{i}] {DEPARTMENTS[i]}交换发言解析失败")
            continue
        rec = sorted(speech["recommend"]) if valid_action(speech["recommend"]) else None
        exchange[i] = {"argument": speech["argument"], "recommend": rec}
        rec_s = f"（推荐 {rec}）" if rec else "（未给出合法推荐）"
        log.append(f"[dept{i}] {DEPARTMENTS[i]}: {speech['argument']}{rec_s}")

    # ---------- 领先选项 ----------
    from collections import Counter
    recs = [tuple(v["recommend"]) for v in exchange.values() if v["recommend"]]
    if recs:
        tally = Counter(recs)
        leader = list(min(tally, key=lambda a: (-tally[a], a)))
    else:
        leader = None   # 没有部门给出合法推荐
    log.append(f"[leader] 当前领先选项: {leader}（来自 {len(recs)} 个部门推荐）")

    # ---------- 轮 2：反对 ----------
    objections = {}
    for i in range(3):
        facts, _ = current_prompt_parts()
        prompt = OBJECTION_PROMPT.format(dept=DEPARTMENTS[i], leader=leader,
                                         facts=facts, h=episode["h"])
        try:
            text, _, _ = client.chat([{"role": "system", "content": prompt},
                                      {"role": "user", "content": "请表态。"}],
                                     max_tokens=400)
        except BudgetExceeded:
            log.append("[budget] 反对轮预算耗尽，转入决定")
            objections = {}
            break
        obj = parse_objection(text)
        if obj is None:
            log.append(f"[dept{i}] {DEPARTMENTS[i]}反对轮解析失败")
            continue
        alt_valid = valid_action(obj["alternative"])
        objections[i] = obj
        tag = "反对" if obj["stance"] == "object" else "附议"
        alt_s = f"，替代 {sorted(obj['alternative'])}" if (obj["stance"] == "object" and alt_valid) else ""
        log.append(f"[dept{i}] {DEPARTMENTS[i]}{tag}: {obj['argument']}{alt_s}")

    # ---------- 轮 3：决定 ----------
    def coord(force_final=False):
        nonlocal parse_failures
        facts, catalog = current_prompt_parts()
        prompt = SYS_PROMPT.format(h=episode["h"], weights=episode["weights"],
                                   facts=facts, cost=cost, catalog=catalog)
        blocks = []
        for i, v in exchange.items():
            rec_s = f"（推荐 {v['recommend']}）" if v["recommend"] else ""
            blocks.append(f"  交换轮·{DEPARTMENTS[i]}: {v['argument']}{rec_s}")
        for i, v in objections.items():
            tag = "反对" if v["stance"] == "object" else "附议"
            blocks.append(f"  反对轮·{DEPARTMENTS[i]}: {tag} {v['argument']}")
        prompt += ("\n\n【协商记录】\n" + "\n".join(blocks) +
                   f"\n当前领先选项: {leader}\n你主持决定：可以参考部门意见，但只信已核验事实与算分器。")
        if force_final:
            prompt += "\n\n【本轮必须拍板】直接输出 action，不要再查询。"
        try:
            text, _, _ = client.chat([{"role": "system", "content": prompt},
                                      {"role": "user", "content": "请决策。"}])
        except BudgetExceeded:
            log.append("[budget] 决定轮预算耗尽，转入兜底")
            return ("budget", None)
        log.append(f"[coord] {text[:200]}")
        decision = parse_decision(text)
        if decision is None:
            parse_failures += 1
            log.append("[parse] 协调者 JSON 解析失败")
            return (None, None)
        if "action" in decision:
            action = sorted(decision["action"])
            if valid_action(action):
                return ("action", action)
            log.append("[invalid] 协调者行动不合法，重试")
            return (None, None)
        return ("tool", decision)

    def execute_queries(decision):
        nonlocal cost
        for q in decision.get("queries", []):
            m = re.fullmatch(r"q_(\d)_(\d)", q or "")
            if not m:
                continue
            j = int(m.group(1)) * 3 + int(m.group(2))
            if core.revealed[j]:
                note = f"[系统] {q} 已核验（{DEPARTMENTS[j % 3]}·群组{j // 3} = {z[j] / 10000:.4f}），不要重复查询"
                facts_lines.append(note)
                log.append(f"[skip] {q} 已核验")
                continue
            if cost >= query_budget:
                facts_lines.append(f"[系统] 查询预算已用完（{cost}/{query_budget}），必须拍板")
                log.append("[budget] 查询预算耗尽")
                return False
            core.apply_receipt(j, z[j])
            cost += 1
            k, i = j // 3, j % 3
            line = f"  {DEPARTMENTS[i]}·群组{k} = {z[j] / 10000:.4f}"
            facts_lines.append(line)
            queries_made.append(line)
            log.append(f"[query] {q} -> {z[j] / 10000:.4f} (成本 {cost})")
        return True

    def score_tool(decision):
        notes = []
        for act in decision["score"]:
            try:
                act_t = tuple(sorted(int(k) for k in act))
            except (TypeError, ValueError):
                notes.append(f"  行动{act}不合法")
                continue
            if act_t not in core.actions or len(act_t) != episode["h"]:
                notes.append(f"  行动{act}不合法")
                continue
            t = core.theta[core.actions.index(act_t)]
            partial, n_unknown, full_neutral = 0.0, 0, 0.0
            for j in range(24):
                if core.revealed[j]:
                    partial += t[j] * z[j]
                    full_neutral += t[j] * z[j]
                elif t[j] != 0:
                    n_unknown += 1
                    full_neutral += t[j] * 5000
            partial /= 100000 * episode["h"]
            neutral = full_neutral / (100000 * episode["h"])
            notes.append(f"  行动{list(act_t)}: 已披露部分得分={partial:.4f} "
                         f"({n_unknown}项未披露, 按0.5中性估计总分≈{neutral:.4f})")
        facts_lines.append("[算分器]\n" + "\n".join(notes))
        log.append("[score] 算分器反馈 " + "; ".join(notes))

    queried_in_decision = False
    for attempt in range(4):   # 决定轮：允许 1 次查询批 + 必须拍板 + 1 次重试
        kind, payload = coord(force_final=queried_in_decision and attempt > 0)
        if kind == "action":
            return {"action": payload, "log": log, "queries": queries_made,
                    "tokens": {"in": client.total_in, "out": client.total_out},
                    "parse_failures": parse_failures, "cost": cost,
                    "leader": leader, "deliberation_rounds": 3}
        if kind == "budget":
            break
        if kind == "tool":
            if "score" in payload:
                score_tool(payload)
            elif "queries" in payload and not queried_in_decision:
                if not execute_queries(payload):
                    break
                queried_in_decision = True
            # 已查过一次还想要查询 -> force_final 会在下一轮回合生效

    # ---------- 兜底 ----------
    z_hat = [z[j] if core.revealed[j] else 5000 for j in range(24)]
    a_idx, _ = core.best_action_with(z_hat) if hasattr(core, "best_action_with") \
        else _point_estimate(core, z_hat)
    action = list(core.actions[a_idx])
    log.append(f"[fallback] 点估计兜底 -> {action}")
    return {"action": action, "log": log, "queries": queries_made,
            "tokens": {"in": client.total_in, "out": client.total_out},
            "parse_failures": parse_failures, "cost": cost, "fallback": True,
            "leader": leader, "deliberation_rounds": 3}


# ============================================================
# 三、主入口
# ============================================================
def main():
    live = "--live" in sys.argv
    n_runs = 1
    if "--n" in sys.argv:
        n_runs = int(sys.argv[sys.argv.index("--n") + 1])
    episodes = compile_dev_episodes()
    run_seed = run_seed_from(sys.argv)

    client_cls = LLMClient if live else MockLLM
    print(f"[{'live' if live else 'mock'}] Exchange-then-Decide，开发任务 x{n_runs}")

    os.makedirs(os.path.join(DATA_DIR, "artifacts"), exist_ok=True)
    summary = []
    for ep in episodes[:n_runs]:
        init = resolve_init(sys.argv, ep)
        eval_core = CertificateCore(ep)
        optimal = [list(a) for a in ep["eval_only_optimal_actions"]]
        client = client_cls(seed=run_seed)
        result = run_etd(ep, init, client)
        idx = eval_core.actions.index(tuple(result["action"]))
        tr = eval_core.true_regret(idx, ep["z_int"]) / 100000
        summary.append({"episode": ep["episode_id"], "regret": round(tr, 4),
                        "cost": result["cost"], "parse_failures": result["parse_failures"],
                        "fallback": result.get("fallback", False),
                        "tokens_out": result["tokens"]["out"],
                        "leader": result.get("leader")})
        print(f"\n===== {ep['episode_id']} | Exchange-then-Decide =====")
        print(f"行动: {result['action']}  真实后悔: {tr:.4f}  最优: {optimal}")
        print(f"成本: {result['cost']}  解析失败: {result['parse_failures']}  "
              f"兜底: {result.get('fallback', False)}  token_out: {result['tokens']['out']}  "
              f"领先: {result.get('leader')}")
        for line in result["log"]:
            print("  " + line)
        with open(os.path.join(DATA_DIR, "artifacts",
                               f"Exchange-then-Decide_{ep['episode_id']}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"episode": ep["episode_id"], "method": "Exchange-then-Decide",
                       "result": result, "true_regret": tr, "optimal": optimal},
                      f, ensure_ascii=False, indent=2)

    if summary:
        import statistics
        print("\n===== 汇总 =====")
        for s in summary:
            print(f"  {s['episode']:22s} 后悔={s['regret']:.4f} 成本={s['cost']:2d} "
                  f"解析失败={s['parse_failures']} 兜底={s['fallback']} "
                  f"tok_out={s['tokens_out']} 领先={s['leader']}")
        print(f"  平均后悔={statistics.mean(s['regret'] for s in summary):.4f} (n={len(summary)})")


if __name__ == "__main__":
    main()
