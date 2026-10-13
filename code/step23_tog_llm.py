# -*- coding: utf-8 -*-
"""step23: LLM 端遗漏差距 (TOG) 分析
TOG = Reg(E_selective) - Reg(E_neutral)，按 (episode_id, method, seed) 配对。
配对源：data/formal_results.jsonl（条件 A/C/S）+ data/formal_llm_neutral.jsonl（条件 neutral）。
输出：analysis/tog_llm.md
"""
import json, itertools, collections
from scipy import stats

W = 'data/formal_results.jsonl'
N = 'data/formal_llm_neutral.jsonl'
CONDS = ['A', 'C', 'S']
METHODS = ['SAS-Tool', 'SAS-SC', 'MAS-Deliberate', 'Exchange-then-Decide']
SEEDS = [17, 29, 43]

def load(path, with_cond):
    d = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        key = (r['episode_id'], r['method'], r['seed'], r['condition']) if with_cond \
              else (r['episode_id'], r['method'], r['seed'])
        d[key] = r
    return d

formal = load(W, True)      # 键含条件：4 方法 x 3 种子 x 3 条件 x 240 = 8640
neutral = load(N, False)    # 条件恒 neutral：2880

# 覆盖检查
missing = [k for k in formal if k[1] in METHODS and k[2] in SEEDS
           and (k[0], k[1], k[2]) not in neutral]
print(f'formal 键数 {len(formal)}（应 8640），neutral 键数 {len(neutral)}（应 2880），缺配对 {len(missing)}')

# 汇总每方法每条件（全体种子合并 + 多种子核对）
def tog_for(method, cond, seed=None):
    diffs, regs_c, regs_n = [], [], []
    for (ep, m, s, c), rc in formal.items():
        if m != method or c != cond:
            continue
        if seed is not None and s != seed:
            continue
        rn = neutral.get((ep, m, s))
        if rn is None:
            continue
        diffs.append(rc['true_regret'] - rn['true_regret'])
        regs_c.append(rc['true_regret'])
        regs_n.append(rn['true_regret'])
    return diffs, regs_c, regs_n

# Holm 校正
def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj

rows = []
for method, cond in itertools.product(METHODS, CONDS):
    diffs, regs_c, regs_n = tog_for(method, cond)
    t, p = stats.ttest_rel(regs_c, regs_n)
    rows.append(dict(method=method, cond=cond, n=len(diffs),
                     mean_c=sum(regs_c)/len(regs_c), mean_n=sum(regs_n)/len(regs_n),
                     tog=sum(diffs)/len(diffs), sd=stats.tstd(diffs), t=t, p=p,
                     per_seed={s: (lambda d: sum(d)/len(d) if d else None)(tog_for(method, cond, s)[0]) for s in SEEDS}))

adjs = holm([r['p'] for r in rows])
for r, a in zip(rows, adjs):
    r['p_holm'] = a

# 成本通道（顺便记录，正文主要用后悔）
cost_rows = []
for method in METHODS:
    dn = [r['cost'] for (ep, m, s), r in neutral.items() if m == method]
    cost_rows.append((method, sum(dn)/len(dn)))

out = ['# LLM 端遗漏差距（TOG）分析 — step23', '',
       '配对设计：同 (episode, method, seed) 下 TOG = Reg(E_selective) − Reg(E_neutral)。',
       f'配对数：每方法每条件 {rows[0]["n"]} 对（240 任务 × 3 种子）。', '',
       '## 主表：TOG = Reg(条件) − Reg(中性)', '',
       '| 方法 | 条件 | Reg(条件) | Reg(中性) | TOG | SD | t | p | Holm p | 每种子 TOG (17/29/43) |',
       '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    seeds_str = '/'.join(f'{r["per_seed"][s]:+.4f}' for s in SEEDS)
    out.append(f'| {r["method"]} | {r["cond"]} | {r["mean_c"]:.4f} | {r["mean_n"]:.4f} | '
               f'{r["tog"]:+.4f} | {r["sd"]:.4f} | {r["t"]:+.2f} | {r["p"]:.4g} | {r["p_holm"]:.4g} | {seeds_str} |')
out += ['', '## 中性臂平均成本（LLM 查询数）', '',
        '| 方法 | 平均成本 |', '|---|---|']
for m, c in cost_rows:
    out.append(f'| {m} | {c:.1f} |')
out += ['', f'注：中性披露的 E0 不依赖种子，三种子差异仅来自解码；Holm 校正覆盖 4 方法 × 3 条件 = 12 项检验。']

open('analysis/tog_llm.md', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('\n'.join(out))
