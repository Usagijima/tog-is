# -*- coding: utf-8 -*-
"""图9：LLM 端遗漏差距（TOG）—— 每方法 x 条件的配对后悔差（95% CI）"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['SAS-Tool', 'SAS-SC', 'MAS-Deliberate', 'Exchange-then-Decide']
MLAB = {'SAS-Tool': 'SAS-Tool', 'SAS-SC': 'SAS-SC',
        'MAS-Deliberate': 'MAS-Delib.', 'Exchange-then-Decide': 'ETD'}
CONDS = ['A', 'C', 'S']
CLAB = {'A': 'A (script)', 'C': 'C (coop.)', 'S': 'S (self-int.)'}
CCOL = {'A': '#4C72B0', 'C': '#DD8452', 'S': '#55A868'}

def load(path, with_cond):
    d = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        key = (r['episode_id'], r['method'], r['seed'], r['condition']) if with_cond \
              else (r['episode_id'], r['method'], r['seed'])
        d[key] = r['true_regret']
    return d

formal = load('data/formal_results.jsonl', True)
neutral = load('data/formal_llm_neutral.jsonl', False)

means, ses = {}, {}
for m in METHODS:
    for c in CONDS:
        diffs = [rc - neutral[(ep, m, s)]
                 for (ep, mm, s, cc), rc in formal.items()
                 if mm == m and cc == c]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        means[(m, c)] = mean
        ses[(m, c)] = (var / n) ** 0.5

fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
x = range(len(METHODS))
w = 0.26
for i, c in enumerate(CONDS):
    xs = [xx + (i - 1) * w for xx in x]
    ys = [means[(m, c)] for m in METHODS]
    es = [1.96 * ses[(m, c)] for m in METHODS]
    ax.bar(xs, ys, w, yerr=es, capsize=3, label=CLAB[c], color=CCOL[c],
           error_kw=dict(lw=0.8, ecolor='#333333'))
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels([MLAB[m] for m in METHODS])
ax.set_ylabel(r'TOG $=\mathrm{Reg}(E_{\mathrm{selective}})-\mathrm{Reg}(E_{\mathrm{neutral}})$')
ax.set_title('LLM-side omission gap: all 12 cells negative\n(paired $n{=}720$; Holm-adjusted $p\\leq1.9\\times10^{-42}$)', fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig('analysis/fig_tog_llm.png', bbox_inches='tight')
print('saved analysis/fig_tog_llm.png')
