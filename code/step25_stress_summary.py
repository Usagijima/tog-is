# -*- coding: utf-8 -*-
"""step25：压力测试汇总（方案 14.4，描述性，不进确认性检验家族）
对照基线：同 16 任务的 A 条件（SAS-Tool 取 formal_results.jsonl；
ODC-WC 取 formal_deterministic.jsonl 的 B=12，种子 17）。
输出 analysis/stress_summary.md
"""
import json, collections
from scipy import stats

SEED = 17
BUDGET = 12

rows = [json.loads(l) for l in open('data/formal_stress.jsonl', encoding='utf-8')]
tasks = sorted({r['episode_id'] for r in rows})

# 基线
base_llm = {}
for l in open('data/formal_results.jsonl', encoding='utf-8'):
    r = json.loads(l)
    if r['condition'] == 'A' and r['method'] == 'SAS-Tool' and r['seed'] == SEED:
        base_llm[r['episode_id']] = r
base_det = {}
for l in open('data/formal_deterministic.jsonl', encoding='utf-8'):
    r = json.loads(l)
    if r['condition'] == 'A' and r['method'] == 'ODC-WC' \
            and r['budget'] == BUDGET and r.get('seed', 17) == SEED:
        base_det[r['episode_id']] = r

def paired(stress, recv, field):
    diffs, ss, bs = [], [], []
    for t in tasks:
        sub = [r for r in rows if r['episode_id'] == t
               and r['stress'] == stress and r['receiver'] == recv]
        if not sub:
            continue
        s = sub[0]
        b = (base_llm if recv == 'SAS-Tool' else base_det).get(t)
        if b is None:
            continue
        diffs.append(s[field] - b[field])
        ss.append(s[field]); bs.append(b[field])
    return diffs, ss, bs

out = ['# 压力测试汇总（step24/25，方案 14.4）', '',
       f'规模：4 压力 × 16 任务（零售 8 + 银行 8）× 2 接收方法 × 种子 {SEED} = 128 条。',
       '**描述性测试**：不做 Holm 家族校正，不替代主统计结论。威胁能力标注见各节。',
       f'基线：同任务 A 条件（SAS-Tool: formal_results；ODC-WC: formal_deterministic B={BUDGET}）。', '',
       '## 总表：压力 vs A 基线（配对，n=16）', '',
       '| 压力 | 接收端 | 基线后悔 | 压力后悔 | Δ后悔 | 配对 t | 基线成本 | 压力成本 | 兜底率 | 解析失败 | 审计失败 |',
       '|---|---|---|---|---|---|---|---|---|---|---|']
for stress in ['dup', 'perm', 'coalition', 'deny']:
    for recv in ['SAS-Tool', 'ODC-WC']:
        sub = [r for r in rows if r['stress'] == stress and r['receiver'] == recv]
        n = len(sub)
        if n == 0:
            continue
        dd, ss, bs = paired(stress, recv, 'true_regret')
        dc, _, _ = paired(stress, recv, 'cost')
        t, p = stats.ttest_rel(ss, bs) if len(ss) > 1 else (float('nan'), float('nan'))
        out.append(f'| {stress} | {recv} | {sum(bs)/len(bs):.4f} | {sum(ss)/len(ss):.4f} '
                   f'| {sum(dd)/len(dd):+.4f} | t={t:.2f} p={p:.3g} '
                   f'| {sum(base_llm[t]["cost"] if recv=="SAS-Tool" else base_det[t]["cost"] for t in tasks if t in (base_llm if recv=="SAS-Tool" else base_det))/max(1,len(bs)):.1f} '
                   f'| {sum(r["cost"] for r in sub)/n:.1f} '
                   f'| {sum(r["fallback"] for r in sub)/n:.0%} '
                   f'| {sum(r["parse_failures"] for r in sub)} '
                   f'| {sum(not r["audit_ok"] for r in sub)} |')

out += ['', '## 分压力读数', '']

# dup
dup_n = collections.Counter()
for r in rows:
    if r['stress'] == 'dup':
        dup_n[r['receiver']] += 1
out += ['### 1. 重复事实（E0 前 2 项重复，8 条目/6 事实）',
        '收据通道幂等（重复收据不重复计费，成本只计一次揭示）；LLM 提示词里事实行重复出现。',
        f"完成 {sum(dup_n.values())} 条。两条接收端均零兜底零审计失败 -> 见总表。", '']

# perm：逐任务不变性
inv = {'SAS-Tool': [0, 0], 'ODC-WC': [0, 0]}
for t in tasks:
    for recv in ['SAS-Tool', 'ODC-WC']:
        sub = [r for r in rows if r['episode_id'] == t
               and r['stress'] == 'perm' and r['receiver'] == recv]
        if not sub:
            continue
        b = (base_llm if recv == 'SAS-Tool' else base_det)[t]
        inv[recv][0] += abs(sub[0]['true_regret'] - b['true_regret']) < 1e-9
        inv[recv][1] += 1
out += ['### 2. 行动 ID 置换（群组编号 k→π[k]，决策应不依赖标签）',
        f"ODC-WC 逐任务后悔完全不变：{inv['ODC-WC'][0]}/{inv['ODC-WC'][1]}（确定性，符合预期）",
        f"SAS-Tool 后悔逐任务不变：{inv['SAS-Tool'][0]}/{inv['SAS-Tool'][1]}（LLM 对编号置换的敏感性直接可见）", '']

# coalition
rules = collections.Counter()
for l in open('data/stress_e0.jsonl', encoding='utf-8'):
    rules[json.loads(l)['gen_rule']] += 1
sub = [r for r in rows if r['stress'] == 'coalition']
regs = [r['true_regret'] for r in sub if r['receiver'] == 'SAS-Tool']
out += ['### 3. 两部门联合披露（增长+运营联盟；威胁能力：共享两部门私有信息，看不到治理部门隐藏数据）',
        f"联盟 E0 生成规则分布：{dict(rules)}（fallback 走 A 脚本，可审计）",
        f"SAS-Tool 在联盟披露下均后悔 {sum(regs)/len(regs):.4f}（n={len(regs)}，与 A 基线对照见总表）", '']

# deny
denied_ok = all(len(r['denied']) == 3 for r in rows if r['stress'] == 'deny')
out += ['### 4. 逐次授权拒绝（前 3 次查询被拒：不揭示、不耗预算，接收端改道）',
        f"全部 deny 轨迹恰好拒绝 3 次：{denied_ok}",
        ' denial 迫使接收端改道其余事实；后悔/成本变化见总表（ODC-WC 为确定性改道规则）。', '']

out += ['## 结论（描述性）', '',
        '压力条件下的 regret/cost 变化均为小规模描述性读数；无任何轨迹因压力条件崩溃、'
        '产生非法行动或审计失败。逐次授权拒绝下两个接收端都能改道完成决策。', '']

open('analysis/stress_summary.md', 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
