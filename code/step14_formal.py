# -*- coding: utf-8 -*-
"""
第十四步：正式实验批量驱动器（方案 3.md 第 17、20 节）

干什么：把 4 个 LLM 方法 × 3 条件 × 3 种子 × 240 测试任务 的全部轨迹
        一格一格跑完，结果落盘、可断点续跑、可独立审计。

工作单元 = (episode_id, condition, method, seed)
  方法：SAS-Tool / SAS-SC / MAS-Deliberate / Exchange-then-Decide
  条件：C / S（LLM 生成披露）/ A（脚本披露）
  种子：17 / 29 / 43（预注册 decoding_seeds）

可靠性设计（昨晚 LM Studio 中途崩溃的教训）：
  - 每条轨迹独立：异常 -> 重试 1 次 -> 记入 formal_errors.jsonl（不算完成，续跑会重试）
  - 连续 5 条失败 -> 熔断中止（多半是服务器挂了，别硬刷）
  - 已完成的单元跳过（formal_results.jsonl 即事实清单）
  - E0 缺失时按（任务,条件,种子）现场生成并落盘 disclosures.jsonl

诚实性设计：
  - 测试任务来自 data/episodes.jsonl；开发任务绝不进入本脚本
  - 每条轨迹跑完做侧车审计：日志里的每个查询值必须与侧车收据一致
  - 成本审计：实际执行查询数必须等于 cost - 6（初始证据 6 项）

用法（先确保 Bionic 在跑：lms server start --port 8000）：
  # 小分片验证（条件 A 不需要 LLM 生成披露，最便宜）：
  python step14_formal.py --cond A --method SAS-Tool --seed 17 --limit 4
  # 正式全量（分多次跑也行，随时中断随时续）：
  python step14_formal.py
"""

import contextlib
import io
import json
import os
import re
import sys
import time
import traceback

from step10_llm import (DATA_DIR, LLMClient, CertificateCore,
                        run_sas_tool)
from step11_mas import run_mas
from step12_exchange import run_etd
from step13_disclosure import build_E0, load_E0, OUT_PATH as DISC_PATH
from step5_sidecar import EvidenceSidecar, snapshot_hash

RESULTS_PATH = os.path.join(DATA_DIR, "formal_results.jsonl")
ERRORS_PATH = os.path.join(DATA_DIR, "formal_errors.jsonl")
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts", "formal")

METHODS = {
    "SAS-Tool": lambda ep, init, client: run_sas_tool(ep, init, client),
    "SAS-SC": lambda ep, init, client: run_sas_tool(ep, init, client, sc_k=5),
    "MAS-Deliberate": run_mas,
    "Exchange-then-Decide": run_etd,
}


def load_test_episodes():
    """正式实验只用测试任务（240 个）。开发任务禁止进入（协议红线）。"""
    path = os.path.join(DATA_DIR, "episodes.jsonl")
    eps = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert all(e["block"] != "dev" for e in eps), "混入开发任务！"
    return eps


def load_done():
    done = set()
    if os.path.exists(RESULTS_PATH):
        for l in open(RESULTS_PATH, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["episode_id"], r["condition"], r["method"], r["seed"]))
    return done


def server_alive():
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=10):
            return True
    except Exception:
        return False


def ensure_E0(ep, cond, seed, client, sidecar):
    """E0 缺失就生成（A 脚本免费；C/S 用 LLM），落盘 disclosures.jsonl。"""
    try:
        return load_E0(ep["episode_id"], cond, seed)["E0"]
    except KeyError:
        pass
    rec = build_E0(ep, cond, client if cond != "A" else None, sidecar, seed=seed)
    with open(DISC_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec["E0"]


def audit_trajectory(ep, result, sidecar_hash):
    """侧车审计：日志中的每个查询值必须等于收据值；查询数必须对上成本。"""
    log = "\n".join(result["log"])
    execs = re.findall(r"\[query\] q_(\d)_(\d) -> ([\d.]+)", log)
    car = EvidenceSidecar(ep, sidecar_hash)
    for k, i, v in execs:
        r = car.query(f"q_{k}_{i}", requester="audit")
        if abs(r["normalized_value_int"] / 10000 - float(v)) > 1e-6:
            return False, f"收据值不符 q_{k}_{i}: 日志{v} != 收据{r['normalized_value_int']/10000}"
    if len(execs) != result["cost"] - 6:
        return False, f"查询数不符: 日志{len(execs)} != cost-6({result['cost'] - 6})"
    return True, f"{len(execs)} 查询全部对上收据"


def run_one(ep, cond, method, seed, client):
    """跑一条轨迹（抑制打印，返回结果+日志文本）。"""
    sh = snapshot_hash()
    sidecar = EvidenceSidecar(ep, sh)
    init = ensure_E0(ep, cond, seed, client, sidecar)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = METHODS[method](ep, init, client)
    eval_core = CertificateCore(ep)
    idx = eval_core.actions.index(tuple(result["action"]))
    tr = eval_core.true_regret(idx, ep["z_int"]) / 100000
    ok, msg = audit_trajectory(ep, result, sh)
    return result, tr, ok, msg, buf.getvalue()


def main():
    cond_f = sys.argv[sys.argv.index("--cond") + 1] if "--cond" in sys.argv else None
    meth_f = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else None
    seed_f = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else None
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    if not server_alive():
        sys.exit("服务器没响应。先启动 Bionic：lms server start --port 8000")

    episodes = load_test_episodes()
    done = load_done()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    conds = [cond_f] if cond_f else ["A", "C", "S"]
    methods = [meth_f] if meth_f else list(METHODS)
    seeds = [seed_f] if seed_f else [17, 29, 43]

    units = [(ep, c, m, s) for ep in episodes for c in conds
             for m in methods for s in seeds
             if (ep["episode_id"], c, m, s) not in done]
    if limit:
        units = units[:limit]
    total, ok_n, fail_n = len(units), 0, 0
    print(f"[formal] 待跑 {total} 条轨迹（已完成 {len(done)} 条），"
          f"条件={conds} 方法={methods} 种子={seeds}")

    consec_err = 0
    for n, (ep, cond, method, seed) in enumerate(units, 1):
        rec = {"episode_id": ep["episode_id"], "condition": cond,
               "method": method, "seed": seed}
        runlog = ""
        try:
            last_err = None
            for attempt in range(2):   # 异常重试 1 次（两次之间换新客户端）
                try:
                    result, tr, audit_ok, audit_msg, runlog = run_one(
                        ep, cond, method, seed, LLMClient(seed=seed))
                    last_err = None
                    break
                except Exception:
                    last_err = traceback.format_exc()
                    time.sleep(15)
            if last_err is not None:
                raise RuntimeError(last_err)
            rec.update({"action": result["action"], "true_regret": round(tr, 4),
                        "cost": result["cost"],
                        "parse_failures": result["parse_failures"],
                        "fallback": result.get("fallback", False),
                        "tokens_in": result["tokens"]["in"],
                        "tokens_out": result["tokens"]["out"],
                        "sc_votes": result.get("sc_votes"),
                        "leader": result.get("leader"),
                        "audit_ok": audit_ok, "audit_msg": audit_msg})
            with open(RESULTS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok_n += 1
            consec_err = 0
            print(f"  [{n}/{total}] {rec['episode_id']} | {cond} | {method} | s{seed} "
                  f"-> 后悔 {tr:.4f} 成本 {rec['cost']} 兜底 {rec['fallback']} "
                  f"审计 {'✓' if audit_ok else '✗ ' + audit_msg}")
        except Exception:
            err = traceback.format_exc()
            with open(ERRORS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "error": err}, ensure_ascii=False) + "\n")
            fail_n += 1
            consec_err += 1
            print(f"  [{n}/{total}] {rec['episode_id']} | {cond} | {method} | s{seed} "
                  f"-> 失败（已记 formal_errors.jsonl，续跑会重试）")
            if consec_err >= 5:
                sys.exit("连续 5 条失败，熔断中止。检查服务器后重新运行本命令即可续跑。")
            time.sleep(20)
        # 每条轨迹的完整日志另存（审计与论文附录素材）
        with open(os.path.join(ARTIFACT_DIR,
                               f"{cond}_{method.replace(' ', '')}_s{seed}_{ep['episode_id']}.log"),
                  "w", encoding="utf-8") as f:
            f.write(runlog)

    print(f"[formal] 本轮完成 {ok_n} 条，失败 {fail_n} 条；"
          f"累计完成 {len(done) + ok_n} / {len(done) + total}")


if __name__ == "__main__":
    main()
