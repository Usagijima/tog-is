# 参考文献核查表（2026-10-11，经学术搜索引擎逐条核实）

> 核查方式：Scholar 学术搜索；原始 CSV 存于 `analysis/refs/`。
> 标注 *[待补]* 的条目信息不完整，写作引用前需人工点开链接再确认。

## 核心相关工作

1. **HiddenBench** — Li, Y., Naito, A., & Shirado, H. *HiddenBench: Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks.* arXiv:2505.11556, 2025. OpenReview 版本同年。https://arxiv.org/abs/2505.11556
   - 方案 3.md 称"2026 年修订并标注 ICMC 2026 接收"——本次检索未独立确认 ICML 2026 接收信息 *[待补：投稿前核实]*。

2. **The Illusion of Multi-Agent Advantage** — Jwalapuram, P., Lin, H., Li, C., Jiao, F., Wang, S., et al. arXiv:2606.13003, 2026. https://arxiv.org/abs/2606.13003

3. **When Truth Is Distributed** — Yan, C., Yue, Z., Zhao, F., Lin, E., Jia, L., Tong, H., et al. *When Truth Is Distributed: Misinformation Derails Collective Fact Recovery in LLM-Based Multi-Agent Systems.* arXiv:2608.03421, 2026.（框架名 ForesightSafety-TIDE）https://arxiv.org/abs/2608.03421

4. **MAS 失败分类** — Cemri, M., Pan, M. Z., Yang, S., et al. *Why Do Multi-Agent LLM Systems Fail?* NeurIPS 2025 Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html
   - 早期版：Pan, M. Z., Cemri, M., Agrawal, L. A., Yang, S., et al. ICLR 2025 Workshop.

## 数学基础

5. **选择性核验机制设计** — Fotakis, D., Tzamos, C., & Zampetakis, M. *Mechanism Design with Selective Verification.* EC 2016. DOI: 10.1145/2940716.2940743

6. **最小最大后悔效用询询** — Wang, T., & Boutilier, C. *Incremental Utility Elicitation with the Minimax Regret Decision Criterion.* IJCAI 2003.

7. — Boutilier, C., Patrascu, R., Poupart, P., & Schuurmans, D. *Constraint-based optimization and utility elicitation using the minimax decision criterion.* Artificial Intelligence, 2006.

8. — Braziunas, D., & Boutilier, C. *Minimax Regret Based Elicitation of Generalized Additive Utilities.* UAI 2007.

## 竞争风险（必须引用以划清边界）

9. **SSRN 撞题预警** — Hu, J., & Miao, C. *Truthful Adaptive Information Procurement for Exact Decision Certificates.* SSRN 7529186（未注明日期，检索时 0 引用）。https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7529186
   - 与我们区别：该文研究"查询成本私有已知"下的机制设计（支付、诚实性），我们是"披露动机 ρ 变化下的 LLM 行为评测 + 固定成本事实通道下的治理协议"。**论文相关工作必须主动引用此文并写明区别**。

## 引用时的诚实备注

- 检索显示的"年份"以 arXiv/会议页为准；引用数会变动，论文中不写引用数。
- 所有 arXiv 编号已逐条点开核实存在；[待补] 项在投稿前完成。
