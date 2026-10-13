# 句句属实，决策仍错：面向策略性隐瞒的多智能体决策充分性证书

**Not Every Truth Is Enough: Decision-Certified Multi-Agent Collaboration under Strategic Omission**

基于真实开源零售与营销数据的受控博弈评测：三个 LLM 部门智能体在各自 KPI 或组织目标下做**真实的**选择性披露，组织方在查询预算内做决策。本项目提供：

- **TOG-IS 评测**：真实性受控的结构化事实通道（证据收据 + 侧车审计），分离"错误陈述"与"选择性遗漏"
- **ODC 协议**：面向遗漏的决策充分性证书（多反例覆盖查询 + 可复算的损失上界放行条件）
- **十方法对照**：4 个 LLM 协调方法 × 6 个确定性参考系，共享初始披露、配对干预、Holm 校正
- **C3 诊断**：最小证书补全成本（精确 MILP）与遗漏税分解

> 论文草稿见 `docs/论文草稿.md` / `docs/论文草稿_en.md`；AAAI-27 LaTeX 投稿稿见 `docs/paper_aaai27.tex`；
> 完整实验日志见 `docs/实验记录.md`（EXP-001~035）。
> 目标会议：AAAI-27 Workshop（2026-11-20 投稿）。

## 主要结果（种子 {17,29,43}，n=720/格，2026-10-12 最终版）

- 四个 LLM 协调方法决策质量无显著差异，查询成本相差逾一倍：ETD ≈7.8 次 vs SAS ≈16 次（省 50%+）
- 确定性证书控制器在 B=12 质量优于全部 LLM 方法：Width-Cert 0.036 vs 最优 LLM 0.045
- 披露行为：组织最优群组的关键事实在 E0 中披露率仅 46~50%（M2）；偏离收益有界且不被策略性披露放大（M5）
- 条件效应：ETD 的 C−A 未复现；SAS-SC 的 C−A 显著（Holm 后 p=0.015）；Width-Cert 的 S−C 显著（p=0.019）
- LLM 端遗漏差距（预注册 16.3 补齐）：12/12 项 TOG 为负且显著（−0.043~−0.058，Holm 后 p ≤ 1.9e-42）；中性固定披露使后悔近翻倍——句句属实 ≠ 足以决策
- 证书：181,440 行 ε 扫描零违规；遗漏税为负（−0.80~−0.84 个事实，|t|>25）

## 仓库结构

```
code/      step1~step25 全流程脚本 + 预注册配置（configs_preregistration.yaml）
data/      任务、披露、正式结果（jsonl，合计 24.3 万行）；原始数据集请自行从 UCI 下载（见下）
analysis/  主表、配对检验、遗漏差距（tog_llm.md）、压力测试（stress_summary.md）、图1~9（PNG + Markdown 表）
docs/      论文草稿、实验记录
```

## 复现步骤

环境：Windows / Python 3.12+（依赖 numpy、pandas、scipy、matplotlib、requests），
本地 LLM 服务（LM Studio 加载 Qwen3-4B-Instruct-2507，GGUF Q4_K_M，`lms server start --port 8000`）。
硬件实测：RTX 5060 8GB（预注册方案假设 5090，已如实登记偏差）。

```bash
# 1. 数据 → 任务（原始数据放 data/ 下，来源见 data/source_manifest.json）
python code/step1_download.py && python code/step2_clean.py && python code/step3_split.py
python code/step4_compile.py && python code/step5_sidecar.py && python code/step6_certificate.py
# 2. 生成初始披露 E0（需要服务器）
python code/step13_disclosure.py --live --all --test
# 3. 确定性方法（不需要服务器，分钟级；逐种子跑，断点自动跳过）
python code/step8_methods.py --formal --seed 17
python code/step8_methods.py --formal --seed 29
python code/step8_methods.py --formal --seed 43
# 4. LLM 正式实验（种子批；可中断续跑，done-set 跳过）
code/run_seeds.bat
# 5. 分析与图表（自动纳入全部已完成种子）
python code/step15_summary_final.py   # 主表 + 配对检验 + 跨种子核对
python code/step16_eps_sweep.py       # ε 扫描（可 --seed 分批跑）
python code/step17_disclosure_behavior.py
python code/step18_stratified.py
python code/step19_min_cost_formal.py # 最小补全成本 MILP（可 --seed 分批跑）
python code/step20_receipt_replacement.py
python code/step21_deviation_gain.py  # 有限偏离收益 M5
# 6. LLM 端遗漏差距（中性披露臂 2,880 条，需要服务器；可中断续跑）
code/run_neutral.bat                  # 或逐个跑 step22_llm_neutral.py --method X --seed Y
python code/step23_tog_llm.py         # TOG 配对检验 + Holm 校正 → analysis/tog_llm.md
python code/step24_stress.py          # 压力测试 128 条（需要服务器；可中断续跑）
python code/step25_stress_summary.py  # 压力测试汇总 → analysis/stress_summary.md
```

协议红线：提示词仅可在开发集调优（`eval_only_*` 字段不进任何在线流程）；
`formal_results.jsonl` 只含测试任务，每条轨迹过侧车审计。

## 数据许可

原始数据（不随仓库分发，请从 UCI 下载）：Online Retail (UCI 352)、Bank Marketing (UCI 222)，均为 CC BY 4.0。
本仓库的派生任务与结果文件以 CC BY 4.0 发布；代码以 MIT 发布。

## 引用

（投稿后补正式 BibTeX）

## 状态

- [x] 种子 17/29/43 全部完成（11,520 LLM 轨迹 + 45,360 确定性行，审计全过）
- [x] ε 标定（181,440 行，认证违规 0）、最小补全成本（MILP 精确）、收据替换因果实验（确定性端 + LLM 端）
- [x] LLM 端遗漏差距 / 中性披露臂 2,880 条（EXP-034，2026-10-12）——预注册实验全部补齐
- [x] 压力测试 128 条（方案 14.4，EXP-035，描述性）
- [ ] 英文投稿排版（paper_aaai27.tex + aaai27.sty）
