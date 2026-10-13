# -*- coding: utf-8 -*-
"""
第十一步：MAS-Deliberate 方法（方案 3.md 第 15.1 节）

MAS-Deliberate 是什么：三个部门智能体 + 一个协调者，靠【语言协商】做决策。
  - 三个部门各持有自己职能的 8 项指标，轮流发言：论证、指出缺失、提议核验、推荐行动
  - 协调者主持：参考部门意见，自己决定付费查询和最终行动
  - 事实纪律：聊天里只能引用"已核验事实表"的数字，未核验数值不许进对话
    （事实通道=收据，聊天=论证；这正是"真实性受控的结构化事实通道"）
  - 决策结束主要依靠语言协商，不做后悔证书 —— 与 ODC 的核心区别

公平性（方案 14.2 / 15.3 / 15.4）：
  - 与 SAS-Tool/SAS-SC 共享：同一初始证据 E0、同一查询目录与成本、同一算分器
  - token 上限跨【全部参与模型的所有调用】累计：输出 ≤1536、输入 ≤12288
  - 部门发言也花 token（共享同一个记账客户端）

协议红线：
  - 提示词只在开发集上编写调优；测试集碰不得
  - 解析失败有限重试；持续失败按冻结的确定性规则兜底，不悄悄丢样本

用法：
  1. 确保 Bionic 服务在跑：lms server start --port 8000
  2. python step11_mas.py            # mock 自测（只验证管路，mock 不懂角色）
  3. python step11_mas.py --live     # 真模型试跑（开发任务）
  4. python step11_mas.py --live --n 4
"""

import itertools
import json
import os
import re
import sys

from step10_llm import (DATA_DIR, DEPARTMENTS, MAX_OUT_TOKENS,  # noqa: F401
                        LLMClient, MockLLM, BudgetExceeded,
                        compile_dev_episodes, parse_decision, CertificateCore,
                        SYS_PROMPT, _point_estimate, resolve_init, run_seed_from)


# ============================================================
# 一、部门智能体提示词（三个部门共用模板，身份不同）
# ============================================================
DEPT_PROMPT = """你是{dept}的代表，参加组织决策协商会议。组织要在 8 个候选群组中选 {h} 个，
组织权重（增长/运营/治理）= {weights}。

【事实纪律】会议有严格的事实纪律：你只能引用下方"已核验事实表"中的数字发言，
不得陈述任何未核验的指标数值，不得编造数字。你的私有印象可以定性表达
（例如"我对群组3有信心"），但不能当数字引用。

已核验事实表：
{facts}

缺失事实目录（可提议协调者付费核验，每个成本 1，总预算 24，已花 {cost}）：
{catalog}

你的任务：
1. 从{dept}的角度论证哪个群组更值得选（引用表中数字）；
2. 指出哪些缺失事实影响判断，提议核验（不一定被采纳）；
3. 给出你的推荐行动（恰好 {h} 个群组）。

回复 JSON（【不要输出 JSON 以外的任何文字】）：
{{"think": "简短理由", "propose_queries": ["q_群组_指标", ...],
  "argument": "一句话论证（给其他与会者看）", "recommend": [群组编号...]}}"""


def parse_dept_speech(text):
    """解析部门发言；只取需要的字段，缺失不致命。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    rec = d.get("recommend")
    try:
        rec = [int(x) for x in rec] if isinstance(rec, list) else []
    except (TypeError, ValueError):
        rec = []   # 非数字群组号（如"数据治理部·群组3"）视为未给出合法推荐（EXP-015）
    pq = d.get("propose_queries")
    pq = [str(q) for q in pq] if isinstance(pq, list) else []
    return {"think": str(d.get("think", ""))[:100],
            "argument": str(d.get("argument", ""))[:200],
            "propose_queries": pq, "recommend": rec}


# ============================================================
# 二、MAS-Deliberate 主流程
# ============================================================
def run_mas(episode, initial_facts, client, query_budget=24, max_cycles=2):
    """返回与 run_sas_tool 同构的结果字典（多 dept_recommends / deliberation_rounds）。

    流程：E0 入账 ->（部门发言 x3 -> 协调者决策）x max_cycles ->
    协调者最终拍板（1 次重试）-> 兜底。任何时刻协调者给出合法 action 即结束。
    """
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

    def dept_call(i, rnd):
        """第 i 个部门在第 rnd 轮发言；失败返回 None（不致命）。"""
        facts, catalog = current_prompt_parts()
        prompt = DEPT_PROMPT.format(dept=DEPARTMENTS[i], h=episode["h"],
                                    weights=episode["weights"], facts=facts,
                                    cost=cost, catalog=catalog)
        try:
            text, _, _ = client.chat([{"role": "system", "content": prompt},
                                      {"role": "user", "content": "请发言。"}],
                                     max_tokens=400)
        except BudgetExceeded:
            log.append(f"[budget] 部门{DEPARTMENTS[i]}发言时预算耗尽")
            return None
        speech = parse_dept_speech(text)
        if speech is None:
            log.append(f"[dept{i}] {DEPARTMENTS[i]}发言解析失败")
            return None
        rec = speech["recommend"]
        valid = (len(rec) == episode["h"] and len(set(rec)) == episode["h"]
                 and all(0 <= k < 8 for k in rec))
        rec_s = f"（推荐 {sorted(rec)}）" if valid else "（未给出合法推荐）"
        log.append(f"[dept{i}] {DEPARTMENTS[i]}: {speech['argument']}{rec_s}")
        if speech["propose_queries"]:
            log.append(f"[dept{i}] 提议核验: {speech['propose_queries']}")
        return {"speech": speech, "valid_rec": sorted(rec) if valid else None}

    def coordinator_call(rnd, force_final=False):
        """协调者决策。返回 (kind, payload)；kind ∈ action/queries/score/None。"""
        nonlocal parse_failures
        facts, catalog = current_prompt_parts()
        prompt = SYS_PROMPT.format(h=episode["h"], weights=episode["weights"],
                                   facts=facts, cost=cost, catalog=catalog)
        if dept_recs:
            block = "\n".join(f"  {DEPARTMENTS[i]}: {a}" for i, a in dept_recs.items())
            prompt += ("\n\n【本轮部门意见】\n" + block +
                       "\n你主持协商：可以参考部门意见，但只信已核验事实与算分器。")
        if force_final:
            prompt += "\n\n【本轮必须拍板】直接输出 action，不要再查询。"
        try:
            text, _, _ = client.chat([{"role": "system", "content": prompt},
                                      {"role": "user", "content": "请决策。"}])
        except BudgetExceeded:
            log.append("[budget] 协调者预算耗尽，转入兜底")
            return ("budget", None)
        log.append(f"[coord] {text[:200]}")
        decision = parse_decision(text)
        if decision is None:
            parse_failures += 1
            log.append("[parse] 协调者 JSON 解析失败")
            return (None, None)
        if "action" in decision:
            action = sorted(decision["action"])
            if len(action) == episode["h"] and all(0 <= k < 8 for k in action) \
                    and len(set(action)) == episode["h"]:
                return ("action", action)
            log.append("[invalid] 协调者行动不合法，重试")
            return (None, None)
        return ("tool", decision)

    def execute_queries(decision):
        """与 step10 完全相同的查询记账规则。"""
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
        """确定性算分器（与 step10 相同规则：未披露按公开中性 0.5）。"""
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

    # ---------- 主循环：部门发言 -> 协调者 ----------
    dept_recs = {}
    cycles_run = 0
    for rnd in range(1, max_cycles + 1):
        cycles_run = rnd
        dept_recs = {}
        for i in range(3):
            r = dept_call(i, rnd)
            if r and r["valid_rec"] is not None:
                dept_recs[i] = r["speech"]["argument"] + f"（推荐 {r['valid_rec']}）"
        kind, payload = coordinator_call(rnd)
        if kind == "action":
            return {"action": payload, "log": log, "queries": queries_made,
                    "tokens": {"in": client.total_in, "out": client.total_out},
                    "parse_failures": parse_failures, "cost": cost,
                    "dept_recommends": {DEPARTMENTS[i]: v for i, v in dept_recs.items()},
                    "deliberation_rounds": rnd}
        if kind == "budget":
            break
        if kind == "tool":
            if "score" in payload:
                score_tool(payload)
            elif "queries" in payload:
                if not execute_queries(payload):
                    break  # 查询预算耗尽 -> 进入最终拍板

    # ---------- 最终拍板（两轮协商后仍未行动） ----------
    for _ in range(2):
        kind, payload = coordinator_call(cycles_run, force_final=True)
        if kind == "action":
            return {"action": payload, "log": log, "queries": queries_made,
                    "tokens": {"in": client.total_in, "out": client.total_out},
                    "parse_failures": parse_failures, "cost": cost,
                    "dept_recommends": {DEPARTMENTS[i]: v for i, v in dept_recs.items()},
                    "deliberation_rounds": cycles_run}
        if kind == "budget":
            break

    # ---------- 兜底：确定性点估计（冻结规则，与 step10 相同） ----------
    z_hat = [z[j] if core.revealed[j] else 5000 for j in range(24)]
    a_idx, _ = core.best_action_with(z_hat) if hasattr(core, "best_action_with") \
        else _point_estimate(core, z_hat)
    action = list(core.actions[a_idx])
    log.append(f"[fallback] 点估计兜底 -> {action}")
    return {"action": action, "log": log, "queries": queries_made,
            "tokens": {"in": client.total_in, "out": client.total_out},
            "parse_failures": parse_failures, "cost": cost, "fallback": True,
            "dept_recommends": {DEPARTMENTS[i]: v for i, v in dept_recs.items()},
            "deliberation_rounds": cycles_run}


# ============================================================
# 三、主入口（与 step10 同构）
# ============================================================
def main():
    live = "--live" in sys.argv
    n_runs = 1
    if "--n" in sys.argv:
        n_runs = int(sys.argv[sys.argv.index("--n") + 1])
    episodes = compile_dev_episodes()
    run_seed = run_seed_from(sys.argv)

    client_cls = LLMClient if live else MockLLM
    print(f"[{'live' if live else 'mock'}] MAS-Deliberate，开发任务 x{n_runs}")

    os.makedirs(os.path.join(DATA_DIR, "artifacts"), exist_ok=True)
    summary = []
    for ep in episodes[:n_runs]:
        init = resolve_init(sys.argv, ep)
        eval_core = CertificateCore(ep)
        optimal = [list(a) for a in ep["eval_only_optimal_actions"]]
        client = client_cls(seed=run_seed)   # 每条轨迹独立记账（含全部部门+协调者调用）
        result = run_mas(ep, init, client)
        idx = eval_core.actions.index(tuple(result["action"]))
        tr = eval_core.true_regret(idx, ep["z_int"]) / 100000
        summary.append({"episode": ep["episode_id"], "regret": round(tr, 4),
                        "cost": result["cost"], "parse_failures": result["parse_failures"],
                        "fallback": result.get("fallback", False),
                        "tokens_out": result["tokens"]["out"],
                        "rounds": result.get("deliberation_rounds", 0)})
        print(f"\n===== {ep['episode_id']} | MAS-Deliberate =====")
        print(f"行动: {result['action']}  真实后悔: {tr:.4f}  最优: {optimal}")
        print(f"成本: {result['cost']}  解析失败: {result['parse_failures']}  "
              f"兜底: {result.get('fallback', False)}  token_out: {result['tokens']['out']}  "
              f"协商轮数: {result.get('deliberation_rounds', 0)}")
        for line in result["log"]:
            print("  " + line)
        with open(os.path.join(DATA_DIR, "artifacts",
                               f"MAS-Deliberate_{ep['episode_id']}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"episode": ep["episode_id"], "method": "MAS-Deliberate",
                       "result": result, "true_regret": tr, "optimal": optimal},
                      f, ensure_ascii=False, indent=2)

    if summary:
        import statistics
        print("\n===== 汇总 =====")
        for s in summary:
            print(f"  {s['episode']:22s} 后悔={s['regret']:.4f} 成本={s['cost']:2d} "
                  f"解析失败={s['parse_failures']} 兜底={s['fallback']} "
                  f"tok_out={s['tokens_out']} 轮数={s['rounds']}")
        print(f"  平均后悔={statistics.mean(s['regret'] for s in summary):.4f} (n={len(summary)})")


if __name__ == "__main__":
    main()
