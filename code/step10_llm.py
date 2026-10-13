# -*- coding: utf-8 -*-
"""
第十步：LLM 接入层 + 开发任务编译 + SAS-Tool 基线（方案 3.md 第 15.1 节）

SAS-Tool 是什么：一个强单智能体协调者。
它能看公共目标、已披露事实和查询目录，用工具花钱查数据，最后拍板选行动。
它【没有】后悔证书保护 —— 它是本文方法要击败的基线。

协议红线（方案 8、15、20 节）：
  - 提示词只在【开发集】上编写调优；测试集碰不得
  - 所有方法共享：同一目录、同一查询成本、同一初始证据、同一 token 上限
  - token 上限：输出 ≤1536、输入 ≤12288（跨全部调用累计）
  - 解析失败允许有限重试；持续失败按冻结的确定性规则兜底，不许悄悄丢样本

用法：
  1. 确保 Bionic 服务在跑：lms server start --port 8000
  2. python step10_llm.py              # mock 自测（不需要 GPU）
  3. python step10_llm.py --live      # 真模型试跑（开发任务）
"""

import itertools
import json
import os
import re
import sys
import urllib.request

import pandas as pd

from step4_compile import (best_action, clip01, to_int,  # noqa: F401
                           retail_cohorts, retail_metrics,
                           bank_cohorts, bank_metrics)
from step6_certificate import CertificateCore

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SPLIT_DIR = os.path.join(DATA_DIR, "split")
SEED = 17
MAX_OUT_TOKENS = 1536    # 每条轨迹输出上限（方案 20.2）
MAX_IN_TOKENS = 12288    # 每条轨迹输入上限
TEMPERATURE = 0.2        # 预注册建议值（开发集确认后冻结）
API = "http://localhost:8000/v1/chat/completions"

DEPARTMENTS = {0: "增长部", 1: "运营部", 2: "数据治理部"}


# ============================================================
# 一、LLM 客户端（记账：每次调用的输入/输出 token 都累计）
# ============================================================
class LLMClient:
    def __init__(self, model=None, seed=None):
        self.model = model
        self.seed = seed   # 重复轨迹的解码种子（预注册 [17,29,43]）；None=不指定
        self.total_in = 0
        self.total_out = 0
        self.n_calls = 0

    def _served_model(self):
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=30) as r:
            models = [m["id"] for m in json.loads(r.read())["data"]
                      if "embed" not in m["id"]]
        return models[0]

    def chat(self, messages, max_tokens=600, temperature=None, seed=None):
        """非流式调用；返回 (文本, 本轮输入token, 本轮输出token)。
        seed 用于重复轨迹（预注册 decoding_seeds [17,29,43]）；后端忽略时无害。
        EXP-015 起两个上限都是硬上限：单调用 max_tokens 被裁剪到剩余输出预算，
        输出/输入累计达上限即拒发新调用（方案 20.2：≤1536 / ≤12288，跨调用累计）。"""
        if self.total_out >= MAX_OUT_TOKENS or self.total_in >= MAX_IN_TOKENS:
            raise BudgetExceeded("token 预算耗尽")
        max_tokens = min(max_tokens, MAX_OUT_TOKENS - self.total_out)
        model = self.model or self._served_model()
        payload = {"model": model, "messages": messages,
                   "max_tokens": max_tokens,
                   "temperature": TEMPERATURE if temperature is None else temperature,
                   "stream": False}
        if seed is not None:
            payload["seed"] = seed
        elif self.seed is not None:
            payload["seed"] = self.seed
        req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
        self.n_calls += 1
        usage = resp.get("usage", {})
        tin = usage.get("prompt_tokens", 0)
        tout = usage.get("completion_tokens", 0)
        self.total_in += tin
        self.total_out += tout
        text = resp["choices"][0]["message"].get("content", "")
        return text, tin, tout


class BudgetExceeded(Exception):
    pass


class MockLLM(LLMClient):
    """离线自测用：先查两个事实，然后拍板。"""
    def __init__(self, seed=None):
        super().__init__(model="mock", seed=seed)
        self.step = 0

    def chat(self, messages, max_tokens=600, temperature=None, seed=None):
        self.n_calls += 1
        self.total_in += 100
        self.total_out += 30
        self.step += 1
        if self.step == 1:
            return json.dumps({"think": "先看看增长部第二个群组的指标", "queries": ["q_1_0", "q_5_1"]}, ensure_ascii=False), 100, 30
        return json.dumps({"think": "够了", "action": [1]}, ensure_ascii=False), 100, 30


# ============================================================
# 二、开发任务编译（只用 dev 分割！提示词调优的唯一合法场所）
# ============================================================
def compile_dev_episodes():
    """从 dev 分割各取一块，编译 12 个开发任务（与 step4 同规则同尺度）。"""
    path = os.path.join(DATA_DIR, "dev_episodes.jsonl")
    if os.path.exists(path):
        return [json.loads(l) for l in open(path, encoding="utf-8")]
    print("[dev] 从开发集编译开发任务 ...")
    # 尺度参数必须与 step4 冻结值一致（读数据卡）
    card = json.load(open(os.path.join(DATA_DIR, "data_card.json"), encoding="utf-8"))
    price_bands = [0] + card["retail_price_bands_pence"][1:-1] + [10**12]

    episodes = []
    for name in ("retail", "bank"):
        df = pd.read_csv(os.path.join(SPLIT_DIR, f"{name}_dev.csv.gz"))
        df = df[df["_split"] == "dev"] if "_split" in df.columns else df
        if name == "retail":
            cohorts, _ = retail_cohorts(df, price_bands)
            metric, scale = retail_metrics, card["retail_scale"]
        else:
            cohorts, _ = bank_cohorts(df, card["bank_job_top4"], card["bank_balance_bands"])
            metric, scale = bank_metrics, card["bank_duration_scale_sD"]
        if len(cohorts) < 8:
            raise RuntimeError(f"开发集 {name} 群组不足 8 个，检查数据卡")
        z_int, info = [], []
        for c in cohorts:
            z1, z2, z3, detail = metric(c["mask"], scale)
            z_int += [to_int(z1), to_int(z2), to_int(z3)]
            info.append({"name": c["name"], **detail})
        WEIGHTS = sorted(set(itertools.permutations([0.5, 0.3, 0.2])))
        for h in (1, 2):
            for wi, w in enumerate(WEIGHTS):
                best, bs = best_action(z_int, w, h)
                episodes.append({
                    "episode_id": f"dev_{name}_h{h}_w{wi}",
                    "dataset": name, "block": "dev", "h": h, "weights": list(w),
                    "cohorts": info, "z_int": z_int,
                    "z": [v / 10000 for v in z_int],
                    "n_actions": 8 if h == 1 else 28,
                    "eval_only_optimal_actions": best,
                    "eval_only_optimal_score": bs,
                    "used_fallback_cohorts": False,
                })
    with open(path, "w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[dev] 已编译 {len(episodes)} 个开发任务 -> {path}")
    return episodes


# ============================================================
# 三、SAS-Tool 方法
# ============================================================
SYS_PROMPT = """你是组织决策协调者。组织要在候选业务群组中选 {h} 个，组织权重为 {weights}。
每个群组有三项指标（增长/运营/治理），完整状态共 24 个事实。

当前已核验的事实：
{facts}

你还可以付费核验其他事实（每个成本 1，总预算 24，已花 {cost}）。
剩余可查询目录：{catalog}

规则：
1. 回复必须是 JSON，三种之一（【不要输出 JSON 以外的任何文字】）：
   {{"think": "简短理由", "queries": ["q_群组_指标", ...]}}      # 查新事实，可批量
   {{"think": "简短理由", "score": [[群组编号...], ...]}}          # 用算分器比较候选行动（免费）
   {{"think": "简短理由", "action": [群组编号...]}}               # 拍板，编号0-7，恰好{h}个
2. think 字段务必精炼，否则会超出输出长度导致 JSON 不完整。
3. 【只能】从"剩余可查询目录"里选查询；已核验的事实在上方事实表里。
4. 拍板前建议先用 score 比较 2-3 个候选行动；你的算术可能出错，以算分器为准。
5. 不确定就多查，但最终必须给出 action。"""

FALLBACK_ACTION_H = {1: [0], 2: [0, 1]}  # 解析持续失败时的兜底（冻结规则）


def resolve_init(argv, episode):
    """初始证据索引：默认开发用中性披露；--e0 时从 disclosures.jsonl 按
    （--cond 条件, --seed 种子）取——所有接收方法共享同一 E0（方案 14.2）。"""
    if "--e0" in argv:
        from step13_disclosure import load_E0
        cond = argv[argv.index("--cond") + 1] if "--cond" in argv else "A"
        seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 17
        return load_E0(episode["episode_id"], cond, seed)["E0"]
    return [k * 3 + i for i in range(3) for k in range(2)]


def run_seed_from(argv):
    return int(argv[argv.index("--seed") + 1]) if "--seed" in argv else None


def parse_decision(text):
    """从模型输出提取决策 JSON；失败返回 None。（EXP-015：非数字行动号
    如 "群组3"/"数据治理部·群组3" 视为解析失败，走既有重试路径，不许崩溃。）"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(d.get("action"), list) and len(d["action"]) >= 1:
        try:
            return {"action": [int(x) for x in d["action"]]}
        except (TypeError, ValueError):
            return None
    if isinstance(d.get("score"), list):
        return {"score": d["score"]}
    if isinstance(d.get("queries"), list):
        return {"queries": [str(q) for q in d["queries"]]}
    return None


def run_sas_tool(episode, initial_facts, client, query_budget=24, max_rounds=10,
                 sc_k=0, sc_temp=0.7):
    """返回 {"action", "log", "tokens", "queries", "parse_failures", "cost", ...}。

    sc_k > 1 时进入 SAS-SC 模式（方案 15.1 节）：信息收集只跑一条轨迹，
    模型首次给出行动时冻结上下文，用温度 sc_temp 独立采样 sc_k-1 个额外决策，
    多数投票（并列按冻结的群组编号次序）。查询成本不翻倍 —— 这是与"跑 k 遍"
    不同的公平设计：SC 只在对答案没信心的最后一步多次掷硬币。
    """
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

    for _ in range(max_rounds):
        unknown = [f"q_{k}_{i}" for k in range(8) for i in range(3) if not core.revealed[k * 3 + i]]
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
            if len(action) == episode["h"] and all(0 <= k < 8 for k in action) and len(set(action)) == episode["h"]:
                if sc_k <= 1:
                    return {"action": action, "log": log, "queries": queries_made,
                            "tokens": {"in": client.total_in, "out": client.total_out},
                            "parse_failures": parse_failures, "cost": cost}
                # ---- SAS-SC：冻结上下文，多次采样投票 ----
                votes = [tuple(action)]
                for si in range(sc_k - 1):
                    try:
                        t2, _, _ = client.chat(
                            [{"role": "system", "content": prompt},
                             {"role": "user", "content": "请决策。"}],
                            max_tokens=400, temperature=sc_temp)
                    except BudgetExceeded:
                        log.append(f"[sc] 采样{si+2}: 预算耗尽，停止采样")
                        break
                    d2 = parse_decision(t2)
                    if d2 is None:
                        log.append(f"[sc] 采样{si+2}: 解析失败 ({t2[:80]!r})")
                        continue
                    if "action" not in d2:
                        log.append(f"[sc] 采样{si+2}: 未给行动 (keys={list(d2)})")
                        continue
                    a2 = sorted(d2["action"])
                    if len(a2) == episode["h"] and len(set(a2)) == episode["h"] and all(0 <= k < 8 for k in a2):
                        votes.append(tuple(a2))
                        log.append(f"[sc] 采样{si+2}: 投票 {a2}")
                    else:
                        log.append(f"[sc] 采样{si+2}: 行动不合法 {a2}")
                from collections import Counter
                tally = Counter(votes)
                # 并列按冻结次序（群组编号字典序）
                winner = min(tally, key=lambda a: (-tally[a], a))
                log.append(f"[sc] 投票 {dict((str(list(k)), v) for k, v in tally.items())} -> {list(winner)}")
                return {"action": list(winner), "log": log, "queries": queries_made,
                        "tokens": {"in": client.total_in, "out": client.total_out},
                        "parse_failures": parse_failures, "cost": cost,
                        "sc_votes": {str(list(k)): v for k, v in tally.items()}}
            log.append("[invalid] 行动不合法，重试")
            continue
        # 算分器工具（方案 15.3：基线也配确定性算分器，避免算术错误干扰结论）
        if "score" in decision:
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
                partial, n_unknown = 0.0, 0
                full_neutral = 0.0
                for j in range(24):
                    if core.revealed[j]:
                        partial += t[j] * z[j]
                        full_neutral += t[j] * z[j]
                    elif t[j] != 0:
                        n_unknown += 1
                        full_neutral += t[j] * 5000   # 未披露按公开中性值 0.5
                partial /= 100000 * episode["h"]
                neutral = full_neutral / (100000 * episode["h"])
                notes.append(f"  行动{list(act_t)}: 已披露部分得分={partial:.4f} "
                             f"({n_unknown}项未披露, 按0.5中性估计总分≈{neutral:.4f})")
            facts_lines.append("[算分器]\n" + "\n".join(notes))
            log.append("[score] 算分器反馈 " + "; ".join(notes))
            continue
        # 执行批量查询（已查过的明确反馈，不许静默跳过）
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
                break
            core.apply_receipt(j, z[j])
            cost += 1
            k, i = j // 3, j % 3
            line = f"  {DEPARTMENTS[i]}·群组{k} = {z[j] / 10000:.4f}"
            facts_lines.append(line)
            queries_made.append(line)
            log.append(f"[query] {q} -> {z[j] / 10000:.4f} (成本 {cost})")

    # 兜底：用确定性算分器，未知事实按 0.5 中位点估计（冻结规则）
    z_hat = [z[j] if core.revealed[j] else 5000 for j in range(24)]
    a_idx, _ = core.best_action_with(z_hat) if hasattr(core, "best_action_with") else _point_estimate(core, z_hat)
    action = list(core.actions[a_idx])
    log.append(f"[fallback] 点估计兜底 -> {action}")
    return {"action": action, "log": log, "queries": queries_made,
            "tokens": {"in": client.total_in, "out": client.total_out},
            "parse_failures": parse_failures, "cost": cost, "fallback": True}


def _point_estimate(core, z_hat):
    """未知=0.5 的点估计下选最优行动（确定性算分器）。"""
    scores = []
    for t in core.theta:
        scores.append(sum(tj * zj for tj, zj in zip(t, z_hat)))
    return max(range(len(scores)), key=lambda i: (scores[i], -i)), None


# ============================================================
# 四、入口：mock 自测 / 真模型试跑
# ============================================================
def main():
    live = "--live" in sys.argv
    # --n 5 表示跑前 5 个开发任务（稳定性验证）；默认 1 个
    n_runs = 1
    if "--n" in sys.argv:
        n_runs = int(sys.argv[sys.argv.index("--n") + 1])
    episodes = compile_dev_episodes()
    # 开发任务用中性披露做初始证据（固定规则，便于对比）；--e0 时改从披露库读取
    run_seed = run_seed_from(sys.argv)

    client_cls = LLMClient if live else MockLLM
    if live:
        print(f"[live] 使用服务器模型，开发任务 x{n_runs}")
    else:
        print(f"[mock] 离线逻辑自测")

    os.makedirs(os.path.join(DATA_DIR, "artifacts"), exist_ok=True)
    sc_mode = "--sc" in sys.argv
    summary = []
    for ep in episodes[:n_runs]:
        init = resolve_init(sys.argv, ep)
        eval_core = CertificateCore(ep)
        optimal = [list(a) for a in ep["eval_only_optimal_actions"]]
        variants = [("SAS-Tool", dict(sc_k=0))]
        if sc_mode:
            variants.append(("SAS-SC(k=5)", dict(sc_k=5)))
        for vname, vkw in variants:
            client = client_cls(seed=run_seed)   # 每个任务×方法新客户端，token 独立记账
            result = run_sas_tool(ep, init, client, **vkw)
            idx = eval_core.actions.index(tuple(result["action"]))
            tr = eval_core.true_regret(idx, ep["z_int"]) / 100000
            summary.append({"episode": ep["episode_id"], "method": vname, "regret": round(tr, 4),
                            "cost": result["cost"], "parse_failures": result["parse_failures"],
                            "fallback": result.get("fallback", False),
                            "tokens_out": result["tokens"]["out"]})
            print(f"\n===== {ep['episode_id']} | {vname} =====")
            print(f"行动: {result['action']}  真实后悔: {tr:.4f}  最优: {optimal}")
            print(f"成本: {result['cost']}  解析失败: {result['parse_failures']}  "
                  f"兜底: {result.get('fallback', False)}  token_out: {result['tokens']['out']}")
            if result.get("sc_votes"):
                print(f"投票: {result['sc_votes']}")
            for line in result["log"]:
                print("  " + line)
            with open(os.path.join(DATA_DIR, "artifacts",
                                   f"{vname.split('(')[0]}_{ep['episode_id']}.json"),
                      "w", encoding="utf-8") as f:
                json.dump({"episode": ep["episode_id"], "method": vname, "result": result,
                           "true_regret": tr, "optimal": optimal}, f, ensure_ascii=False, indent=2)

    if len(summary) > 1:
        print("\n===== 汇总 =====")
        for s in summary:
            print(f"  {s['episode']:22s} {s['method']:12s} 后悔={s['regret']:.4f} 成本={s['cost']:2d} "
                  f"解析失败={s['parse_failures']} 兜底={s['fallback']} tok_out={s['tokens_out']}")
        import statistics
        for vname in dict.fromkeys(s["method"] for s in summary):
            regs = [s["regret"] for s in summary if s["method"] == vname]
            print(f"  {vname:12s} 平均后悔={statistics.mean(regs):.4f} (n={len(regs)})")


if __name__ == "__main__":
    main()
