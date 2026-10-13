# -*- coding: utf-8 -*-
"""step24：额外压力测试（方案 3.md 第 14.4 节，EXP-035）

规模：4 种压力条件 × 16 任务（零售 8 + 银行 8，block00）× 2 接收方法
     （SAS-Tool 强单智能体 + ODC-WC 确定性证书）× 1 随机配置（种子 17）
     = 128 条轨迹。占用第 20 节预留预算；只做小规模描述性测试，
     不替代主统计结论（不做 Holm 家族检验）。

四种压力（只核验原有声明与禁用主动核验由核心消融覆盖，不重复计费）：
  dup        重复事实：E0 里重复 2 个事实条目（收据通道应幂等，接收端应不崩溃）
  perm       行动 ID 置换：8 个群组编号整体置换，检验决策是否依赖标签编号
  coalition  两个部门（增长+运营）联合选择披露：联盟共享私有信息、看不到第三部门
             隐藏数据；威胁能力与单部门条件分开标注
  deny       逐次授权拒绝：前 3 次查询请求被授权系统拒绝（不揭示、不耗预算），
             接收端须改道其余事实

输出：data/formal_stress.jsonl（断点续跑）；联合披露 E0 缓存 data/stress_e0.jsonl
用法：python step24_stress.py [--stress dup] [--receiver SAS-Tool] [--limit N]
"""
import contextlib
import io
import json
import os
import random
import re
import sys
import time
import traceback

from step10_llm import (DATA_DIR, DEPARTMENTS, LLMClient, CertificateCore,
                        parse_decision, SYS_PROMPT, _point_estimate,
                        BudgetExceeded)
from step8_methods import (run_certified, replay_at_budget, query_odc_wc,
                           _unknown, disclose_A)
from step13_disclosure import load_E0, scripted_disclosure
from step5_sidecar import EvidenceSidecar, snapshot_hash

OUT_PATH = os.path.join(DATA_DIR, "formal_stress.jsonl")
ERR_PATH = os.path.join(DATA_DIR, "formal_stress_errors.jsonl")
COAL_PATH = os.path.join(DATA_DIR, "stress_e0.jsonl")
ART_DIR = os.path.join(DATA_DIR, "artifacts", "stress")
SEED = 17                 # 14.4：一个随机配置
N_DENY = 3                # 逐次授权拒绝的次数
BUDGET = 12               # 确定性接收端主检查点（对齐主实验 B=12）

STRESSES = ["dup", "perm", "coalition", "deny"]
RECEIVERS = ["SAS-Tool", "ODC-WC"]
COALITION_DEPTS = [0, 1]  # 增长 + 运营联盟


# ============================================================
# 任务选取：两域各 8 个（block00 前 8）
# ============================================================
def load_stress_tasks():
    eps = [json.loads(l) for l in open(os.path.join(DATA_DIR, "episodes.jsonl"),
                                       encoding="utf-8")]
    assert all(e["block"] != "dev" for e in eps), "混入开发任务！"
    retail = [e for e in eps if e["dataset"] == "retail" and e["block"] == 0][:8]
    bank = [e for e in eps if e["dataset"] == "bank" and e["block"] == 0][:8]
    assert len(retail) == 8 and len(bank) == 8
    return retail + bank


# ============================================================
# 压力构造
# ============================================================
def make_dup_init(e0):
    """重复事实：E0 前 2 项再出现一次（8 个条目，6 个不同事实）。"""
    return list(e0) + list(e0[:2])


def make_perm_episode(ep, pi):
    """行动 ID 置换：群组编号 k -> pi[k]，任务状态同步置换。"""
    z_old = ep["z_int"]
    # 新编号 k' 的事实 = 旧编号 pi.index(k') 的事实
    z_new = []
    zf_old = ep.get("z")
    zf_new = [] if zf_old else None
    for k_new in range(8):
        k_old = pi.index(k_new)
        for i in range(3):
            z_new.append(z_old[k_old * 3 + i])
            if zf_old:
                zf_new.append(zf_old[k_old * 3 + i])
    ep2 = dict(ep)
    ep2["episode_id"] = ep["episode_id"] + "_perm"
    ep2["z_int"] = z_new
    if zf_old:
        ep2["z"] = zf_new
    if ep.get("cohorts"):
        ep2["cohorts"] = [ep["cohorts"][pi.index(k)] for k in range(8)]
    return ep2


COAL_PROMPT = """你是{dept_a}与{dept_b}组成的联盟的代表。组织要在 8 个候选群组中选 {h} 个，
组织权重（增长/运营/治理）= {weights}。

联盟 privately 持有两部门的全部 16 项指标（联盟内部共享的私有信息）：
{dept_a}（部门{a}）：
{lines_a}
{dept_b}（部门{b}）：
{lines_b}

联盟目标：在合法前提下（只说真话、每部门恰好 2 项、只能披露本部门列的事实），
尽量选择能让组织选中【{dept_a}与{dept_b}指标之和最强】的群组。

回复 JSON（【不要输出 JSON 以外的任何文字】）：
{{"disclose_a": ["q_群组编号_{a}", "q_群组编号_{a}"], "disclose_b": ["q_群组编号_{b}", "q_群组编号_{b}"], "reason": "简短理由"}}
（查询编号第二个数字必须是部门号：{a} 或 {b}）"""


def parse_coalition(text, a, b):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    out = {}
    for key, dept in (("disclose_a", a), ("disclose_b", b)):
        qs = d.get(key)
        if not isinstance(qs, list) or len(qs) != 2:
            return None
        idx = []
        for q in qs:
            mm = re.fullmatch(r"q_(\d)_(\d)", str(q) or "")
            if not mm or int(mm.group(2)) != dept:
                return None
            idx.append(int(mm.group(1)) * 3 + dept)
        if len(set(idx)) != 2:
            return None
        out[dept] = sorted(idx)
    out["reason"] = str(d.get("reason", ""))[:150]
    return out


def build_coalition_E0(ep, client, sidecar):
    """联盟 E0：部门 0/1 联合 LLM 披露（共享私有信息），部门 2 走 A 脚本。
    威胁能力标注：联盟不知部门 2 的隐藏数据。缓存续跑。"""
    if os.path.exists(COAL_PATH):
        for l in open(COAL_PATH, encoding="utf-8"):
            r = json.loads(l)
            if r["episode_id"] == ep["episode_id"] and r["seed"] == SEED:
                return r
    z = ep["z_int"]
    a, b = COALITION_DEPTS
    lines_a = "\n".join(f"  群组{k} = {z[k * 3 + a] / 10000:.4f}" for k in range(8))
    lines_b = "\n".join(f"  群组{k} = {z[k * 3 + b] / 10000:.4f}" for k in range(8))
    prompt = COAL_PROMPT.format(dept_a=DEPARTMENTS[a], dept_b=DEPARTMENTS[b],
                                a=a, b=b, h=ep["h"], weights=ep["weights"],
                                lines_a=lines_a, lines_b=lines_b)
    parsed, rule = None, "llm_coalition"
    for attempt in range(2):
        try:
            text, _, _ = LLMClient(seed=SEED).chat(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": "请披露。"}], max_tokens=400)
        except Exception:
            break
        parsed = parse_coalition(text, a, b)
        if parsed:
            break
    disclosures = {}
    if parsed:
        for dept in (a, b):
            receipts = []
            for j in parsed[dept]:
                r = sidecar.query(f"q_{j // 3}_{j % 3}", requester=f"coalition_dept_{dept}_gen")
                receipts.append(r["receipt_id"])
                assert r["normalized_value_int"] == z[j]
            disclosures[DEPARTMENTS[dept]] = {
                "fact_indices": parsed[dept], "recommend": None,
                "rule": rule, "reason": parsed["reason"], "receipt_ids": receipts}
    else:
        rule = "fallback_to_A_script"
    if not parsed:
        for dept in (a, b):
            d = scripted_disclosure(ep, dept)
            receipts = []
            for j in d["fact_indices"]:
                r = sidecar.query(f"q_{j // 3}_{j % 3}", requester=f"coalition_dept_{dept}_gen")
                receipts.append(r["receipt_id"])
                assert r["normalized_value_int"] == z[j]
            disclosures[DEPARTMENTS[dept]] = {**d, "receipt_ids": receipts}
    d2 = scripted_disclosure(ep, 2)
    receipts2 = []
    for j in d2["fact_indices"]:
        r = sidecar.query(f"q_{j // 3}_{j % 3}", requester="coalition_dept_2_gen")
        receipts2.append(r["receipt_id"])
        assert r["normalized_value_int"] == z[j]
    disclosures[DEPARTMENTS[2]] = {**d2, "receipt_ids": receipts2}

    e0 = sorted(j for d in disclosures.values() for j in d["fact_indices"])
    assert len(e0) == 6 and len(set(e0)) == 6
    rec = {"episode_id": ep["episode_id"], "seed": SEED, "condition": "coalition",
           "coalition_depts": COALITION_DEPTS, "threat_note":
           "联盟共享两部门私有信息，但无法看到第三部门隐藏数据",
           "gen_rule": rule, "E0": e0, "disclosures": disclosures}
    with open(COAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# ============================================================
# 接收端 1：SAS-Tool（含逐次授权拒绝钩子）
# run_sas_tool 的授权拒绝分叉：前 N_DENY 次查询请求被拒绝
# （不揭示、不耗预算），之后恢复。与 step10 的 run_sas_tool 保持同步！
# ============================================================
def run_sas_tool_stress(episode, initial_facts, client, deny_first=0,
                        query_budget=24, max_rounds=10):
    z = episode["z_int"]
    core = CertificateCore(episode)
    facts_lines = []
    cost = 0
    for j in initial_facts:
        core.apply_receipt(j, z[j])
        cost += 1
        k, i = j // 3, j % 3
        facts_lines.append(f"  {DEPARTMENTS[i]}·群组{k} = {z[j] / 10000:.4f}")
    log, queries_made, parse_failures = [], facts_lines[:], 0
    denied, denied_total = [], 0

    for _ in range(max_rounds):
        unknown = [f"q_{k}_{i}" for k in range(8) for i in range(3)
                   if not core.revealed[k * 3 + i]]
        prompt = SYS_PROMPT.format(
            h=episode["h"], weights=episode["weights"], cost=cost,
            facts="\n".join(facts_lines) if facts_lines else "  （无）",
            catalog=", ".join(unknown) if unknown else "（无，必须拍板）")
        try:
            text, _, _ = client.chat(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": "请决策。"}])
        except BudgetExceeded:
            log.append("[budget] token 预算耗尽，转入兜底")
            break
        log.append(f"[model] {text[:200]}")
        decision = parse_decision(text)
        if decision is None:
            parse_failures += 1
            log.append("[parse] JSON 解析失败")
            if parse_failures >= 2:
                break
            continue
        if "action" in decision:
            action = sorted(decision["action"])
            if len(action) == episode["h"] and all(0 <= k < 8 for k in action) \
                    and len(set(action)) == episode["h"]:
                return {"action": action, "log": log, "queries": queries_made,
                        "tokens": {"in": client.total_in, "out": client.total_out},
                        "parse_failures": parse_failures, "cost": cost,
                        "denied": denied}
            log.append("[invalid] 行动不合法，重试")
            continue
        if "score" in decision:
            # 压力测试不提供算分器重复分支的完整复制：拒绝并提示
            facts_lines.append("[系统] 算分器本轮不可用")
            log.append("[score] 算分器请求被拒绝（压力条件）")
            continue
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
            # ---- 逐次授权拒绝钩子 ----
            if denied_total < deny_first:
                denied_total += 1
                denied.append(q)
                note = f"[系统] 授权拒绝：{q} 本次查询未获授权（请改查其他事实）"
                facts_lines.append(note)
                log.append(f"[deny] {q} 授权拒绝 ({denied_total}/{deny_first})")
                continue
            if cost >= query_budget:
                facts_lines.append(f"[系统] 查询预算已用完（{cost}/{query_budget}），必须拍板")
                log.append("[budget] 查询预算耗尽")
                break
            core.apply_receipt(j, z[j])
            cost += 1
            k, i = j // 3, j % 3
            line = f"  {DEPARTMENTS[i]}·群组{k} = {z[j] / 10000:.4f}"
            facts_lines.append(line)
            queries_made.append(line)
            log.append(f"[query] {q} -> {z[j] / 10000:.4f} (成本 {cost})")

    z_hat = [z[j] if core.revealed[j] else 5000 for j in range(24)]
    a_idx, _ = core.best_action_with(z_hat) if hasattr(core, "best_action_with") \
        else _point_estimate(core, z_hat)
    action = list(core.actions[a_idx])
    log.append(f"[fallback] 点估计兜底 -> {action}")
    return {"action": action, "log": log, "queries": queries_made,
            "tokens": {"in": client.total_in, "out": client.total_out},
            "parse_failures": parse_failures, "cost": cost,
            "fallback": True, "denied": denied}


# ============================================================
# 接收端 2：ODC-WC（确定性，含逐次授权拒绝查询包装）
# ============================================================
def run_odc_wc_stress(ep, init, deny_first=0, budget=BUDGET):
    """跑 ODC-WC 到预算 B，返回 B=12 检查点读数 + 拒绝日志。"""
    denied, denied_total = [], 0

    def wrapped(core):
        nonlocal denied_total
        j = query_odc_wc(core)
        while j is not None and denied_total < deny_first and j not in denied:
            denied_total += 1
            denied.append(j)
            alternatives = [u for u in _unknown(core) if u not in denied]
            if not alternatives:
                return None
            j = alternatives[0]   # 拒绝后改道：取索引最小的未知事实
        return j

    traj = run_certified(ep, list(init), wrapped, budget=max(budget, 24))
    step = replay_at_budget(traj, budget)
    eval_core = CertificateCore(ep)
    idx = step["action"]   # 轨迹里的 action 是行动索引（对齐 step8 用法）
    tr = eval_core.true_regret(idx, ep["z_int"]) / 100000
    return {"action": list(eval_core.actions[idx]), "true_regret": round(tr, 4),
            "cost": step["cost"], "certified": step["certified"],
            "regret_bound": step["regret_bound"], "denied": denied}


# ============================================================
# 审计（初始条目数可变的压力版）
# ============================================================
def audit_stress(ep, result, sidecar_hash, n_init):
    log = "\n".join(result["log"]) if "log" in result else ""
    execs = re.findall(r"\[query\] q_(\d)_(\d) -> ([\d.]+)", log)
    car = EvidenceSidecar(ep, sidecar_hash)
    for k, i, v in execs:
        r = car.query(f"q_{k}_{i}", requester="audit")
        if abs(r["normalized_value_int"] / 10000 - float(v)) > 1e-6:
            return False, f"收据值不符 q_{k}_{i}"
    if len(execs) != result["cost"] - n_init:
        return False, f"查询数不符: 日志{len(execs)} != cost-{n_init}({result['cost'] - n_init})"
    return True, f"{len(execs)} 查询全部对上收据"


# ============================================================
# 主流程
# ============================================================
def load_done():
    done = set()
    if os.path.exists(OUT_PATH):
        for l in open(OUT_PATH, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["episode_id"], r["stress"], r["receiver"]))
    return done


def server_alive():
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=10):
            return True
    except Exception:
        return False


def main():
    stress_f = sys.argv[sys.argv.index("--stress") + 1] if "--stress" in sys.argv else None
    recv_f = sys.argv[sys.argv.index("--receiver") + 1] if "--receiver" in sys.argv else None
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    tasks = load_stress_tasks()
    rng = random.Random(SEED)
    # 置换表逐任务预生成（seed 固定 -> 可复现）
    perm_tables = {}
    for ep in tasks:
        pi = list(range(8))
        rng.shuffle(pi)
        perm_tables[ep["episode_id"]] = pi

    done = load_done()
    os.makedirs(ART_DIR, exist_ok=True)
    units = []
    for ep in tasks:
        for stress in ([stress_f] if stress_f else STRESSES):
            for recv in ([recv_f] if recv_f else RECEIVERS):
                if (ep["episode_id"], stress, recv) not in done:
                    units.append((ep, stress, recv))
    if limit:
        units = units[:limit]
    need_server = any(u[2] == "SAS-Tool" for u in units)
    if need_server and not server_alive():
        sys.exit("服务器没响应。先启动 Bionic：lms server start --port 8000")

    print(f"[stress] 待跑 {len(units)} 条（已完成 {len(done)} 条），种子={SEED}")
    ok_n = fail_n = 0
    for n, (ep, stress, recv) in enumerate(units, 1):
        rec = {"episode_id": ep["episode_id"], "stress": stress,
               "receiver": recv, "seed": SEED}
        runlog = ""
        try:
            sh = snapshot_hash()
            sidecar = EvidenceSidecar(ep, sh)
            # ---- 构造压力变体 ----
            ep_run = ep
            n_init = 6
            deny_first = 0
            extras = {}
            if stress == "dup":
                init = make_dup_init(load_E0(ep["episode_id"], "A", SEED)["E0"])
                n_init = 8
                extras["dup_entries"] = 2
            elif stress == "perm":
                pi = perm_tables[ep["episode_id"]]
                ep_run = make_perm_episode(ep, pi)
                init = load_E0(ep["episode_id"], "A", SEED)["E0"]
                # 事实编号随群组置换：旧 q_k_i -> 新 q_pi[k]_i
                init = sorted(pi[j // 3] * 3 + j % 3 for j in init)
                sidecar = EvidenceSidecar(ep_run, sh)
                extras["perm_map"] = pi
            elif stress == "coalition":
                init = build_coalition_E0(ep, None, sidecar)["E0"]
                extras["coalition_depts"] = COALITION_DEPTS
            elif stress == "deny":
                init = load_E0(ep["episode_id"], "A", SEED)["E0"]
                deny_first = N_DENY
            else:
                raise ValueError(stress)

            # ---- 跑接收端 ----
            if recv == "SAS-Tool":
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    result = run_sas_tool_stress(
                        ep_run, init, LLMClient(seed=SEED), deny_first=deny_first)
                runlog = buf.getvalue() + "\n".join(result["log"])
                core = CertificateCore(ep_run)
                idx = core.actions.index(tuple(result["action"]))
                tr = core.true_regret(idx, ep_run["z_int"]) / 100000
                ok, msg = audit_stress(ep_run, result, sh, n_init)
                rec.update({"action": result["action"], "true_regret": round(tr, 4),
                            "cost": result["cost"],
                            "parse_failures": result["parse_failures"],
                            "fallback": result.get("fallback", False),
                            "tokens_in": result["tokens"]["in"],
                            "tokens_out": result["tokens"]["out"],
                            "denied": result.get("denied", []),
                            "audit_ok": ok, "audit_msg": msg})
            else:  # ODC-WC
                result = run_odc_wc_stress(ep_run, init, deny_first=deny_first)
                rec.update({"action": result["action"],
                            "true_regret": result["true_regret"],
                            "cost": result["cost"],
                            "certified": result["certified"],
                            "regret_bound": result["regret_bound"],
                            "parse_failures": 0, "fallback": False,
                            "tokens_in": 0, "tokens_out": 0,
                            "denied": result["denied"],
                            "audit_ok": True, "audit_msg": "确定性接收端"})
            rec["extras"] = extras
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok_n += 1
            print(f"  [{n}/{len(units)}] {ep['episode_id']} | {stress} | {recv} "
                  f"-> 后悔 {rec['true_regret']} 成本 {rec['cost']} "
                  f"兜底 {rec['fallback']} 审计 {'✓' if rec['audit_ok'] else '✗'}",
                  flush=True)
        except Exception:
            err = traceback.format_exc()
            with open(ERR_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "error": err}, ensure_ascii=False) + "\n")
            fail_n += 1
            print(f"  [{n}/{len(units)}] {ep['episode_id']} | {stress} | {recv} -> 失败",
                  flush=True)
            time.sleep(10)
        with open(os.path.join(ART_DIR,
                               f"{stress}_{recv.replace(' ', '')}_{ep['episode_id']}.log"),
                  "w", encoding="utf-8") as f:
            f.write(runlog)

    print(f"[stress] 本轮完成 {ok_n} 条，失败 {fail_n} 条；"
          f"累计 {len(done) + ok_n} / {len(done) + len(units)}")


if __name__ == "__main__":
    main()
