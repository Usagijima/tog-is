# -*- coding: utf-8 -*-
"""
第二十二步：LLM 接收端的中性披露臂（方案 3.md 第 16.3 节遗漏差距 TOG 的缺失一臂）

背景：确定性接收端的收据替换已由 step20 覆盖（三条件对比即干预）；对 LLM 接收端，
每个方法在三种条件下的对比本身已冻结接收端/提示词/种子，唯一变化是 E0 内容
（推荐不进入 LLM 接收端提示词），因此条件对比即"内容 × LLM 推理"的通道检验
（主表 paired_tests_final.md 已完成此分析：SAS-SC C−A 显著、ETD C−A 未复现）。

真正的预注册缺口是 16.3 节的遗漏差距 TOG：Reg(E_selective) − Reg(E_neutral)。
中性披露（step8.disclose_neutral，每部门按冻结编号披露前 2 个事实，与种子无关）
此前只在确定性端跑过；本脚本为 4 个 LLM 方法 × 3 解码种子补齐中性 E0 臂。

工作单元 = (episode_id, method, seed)，条件恒为 "neutral"：
  每条轨迹独立：异常 -> 重试 1 次（换新客户端，1536/12288 预算铁律）
  -> 记入 formal_llm_neutral_errors.jsonl（不算完成，续跑会重试）
  连续 5 条失败 -> 熔断中止
  已完成单元跳过（formal_llm_neutral.jsonl 即事实清单）
  每条轨迹跑完做侧车审计（与 step14 同一规则）

用法（先确保 Bionic 在跑：lms server start --port 8000）：
  python step22_llm_neutral.py --method SAS-Tool --seed 17 --limit 4   # 烟测
  python step22_llm_neutral.py                                        # 全量
输出：data/formal_llm_neutral.jsonl；日志 data/artifacts/neutral/
"""

import contextlib
import io
import json
import os
import sys
import time
import traceback

from step10_llm import DATA_DIR, LLMClient, CertificateCore
from step14_formal import (load_test_episodes, server_alive,
                           audit_trajectory, METHODS)
from step5_sidecar import EvidenceSidecar, snapshot_hash
from step8_methods import disclose_neutral

RESULTS_PATH = os.path.join(DATA_DIR, "formal_llm_neutral.jsonl")
ERRORS_PATH = os.path.join(DATA_DIR, "formal_llm_neutral_errors.jsonl")
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts", "neutral")


def load_done():
    done = set()
    if os.path.exists(RESULTS_PATH):
        for l in open(RESULTS_PATH, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["episode_id"], r["method"], r["seed"]))
    return done


def run_one(ep, method, seed, client):
    """跑一条中性臂轨迹：init 为确定性中性披露（与种子无关）。"""
    sh = snapshot_hash()
    sidecar = EvidenceSidecar(ep, sh)
    init, _ = disclose_neutral(ep)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = METHODS[method](ep, init, client)
    eval_core = CertificateCore(ep)
    idx = eval_core.actions.index(tuple(result["action"]))
    tr = eval_core.true_regret(idx, ep["z_int"]) / 100000
    ok, msg = audit_trajectory(ep, result, sh)
    return result, tr, ok, msg, buf.getvalue()


def main():
    meth_f = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else None
    seed_f = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else None
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    meths_f = None
    if "--methods" in sys.argv:
        i = sys.argv.index("--methods") + 1
        meths_f = []
        while i < len(sys.argv) and not sys.argv[i].startswith("--"):
            meths_f.append(sys.argv[i])
            i += 1

    if not server_alive():
        sys.exit("服务器没响应。先启动 Bionic：lms server start --port 8000")

    episodes = load_test_episodes()
    done = load_done()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    methods = meths_f if meths_f else ([meth_f] if meth_f else list(METHODS))
    seeds = [seed_f] if seed_f else [17, 29, 43]

    units = [(ep, m, s) for ep in episodes for m in methods for s in seeds
             if (ep["episode_id"], m, s) not in done]
    if limit:
        units = units[:limit]
    total, ok_n, fail_n = len(units), 0, 0
    print(f"[neutral] 待跑 {total} 条轨迹（已完成 {len(done)} 条），"
          f"方法={methods} 种子={seeds}")

    consec_err = 0
    for n, (ep, method, seed) in enumerate(units, 1):
        rec = {"episode_id": ep["episode_id"], "condition": "neutral",
               "method": method, "seed": seed}
        runlog = ""
        try:
            last_err = None
            for attempt in range(2):   # 异常重试 1 次（两次之间换新客户端）
                try:
                    result, tr, audit_ok, audit_msg, runlog = run_one(
                        ep, method, seed, LLMClient(seed=seed))
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
            print(f"  [{n}/{total}] {rec['episode_id']} | neutral | {method} | s{seed} "
                  f"-> 后悔 {tr:.4f} 成本 {rec['cost']} 兜底 {rec['fallback']} "
                  f"审计 {'OK' if audit_ok else 'X ' + audit_msg}")
        except Exception:
            err = traceback.format_exc()
            with open(ERRORS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "error": err}, ensure_ascii=False) + "\n")
            fail_n += 1
            consec_err += 1
            print(f"  [{n}/{total}] {rec['episode_id']} | neutral | {method} | s{seed} "
                  f"-> 失败（已记 formal_llm_neutral_errors.jsonl，续跑会重试）")
            if consec_err >= 5:
                sys.exit("连续 5 条失败，熔断中止。检查服务器后重新运行本命令即可续跑。")
            time.sleep(20)
        with open(os.path.join(ARTIFACT_DIR,
                               f"neutral_{method.replace(' ', '')}_s{seed}_"
                               f"{ep['episode_id']}.log"),
                  "w", encoding="utf-8") as f:
            f.write(runlog)

    print(f"[neutral] 本轮完成 {ok_n} 条，失败 {fail_n} 条；"
          f"累计完成 {len(done) + ok_n} / {len(done) + total}")


if __name__ == "__main__":
    main()
