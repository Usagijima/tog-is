# Not Every Truth Is Enough: Decision-Certified Multi-Agent Collaboration under Strategic Omission

> Draft status: Introduction / Related Work / Setup / Method / Experimental Design finalized (2026-10-11).
> All result sections have been updated to the final three-seed statistics (2026-10-12; seeds {17, 29, 43}; LLM and deterministic methods alike at n=720 per cell).
> Only remaining experiment: the LLM-side receipt swap (the deterministic-receiver version is complete across three seeds).
> Target: AAAI-27 Workshop (submission ~2026-11-20).

---

## Abstract

Multi-agent systems typically verify exchanged information to improve collaboration reliability, yet truthful reports may still fail to contain the information a decision requires. We study a semi-synthetic organizational decision game, TOG-IS, built from real open-source retail and marketing data: three LLM agents each hold one department's business metrics and disclose facts under either a shared organizational objective (cooperative condition) or departmental KPIs (self-interested condition). We design a truthfulness-controlled structured fact channel with evidence receipts to separate *false statements* from *selective omission*, and propose Omission-aware Decision Certificates (ODC): the protocol proactively verifies the facts that could still overturn the current action and, under a trusted data contract and valid-range assumptions, attaches a verifiable upper bound on organizational loss to every certified action. The evaluation covers strong single-agent coordinators, structured multi-agent deliberation, and six deterministic controllers, measuring true regret, certification coverage, verification cost, and bounded deviation gain. Across the preregistered three-seed replication (n=720 per cell): the four LLM coordinators show no significant quality differences while differing by more than 2× in query cost (ETD ≈7.8 queries vs SAS ≈16, a 50%+ reduction); deterministic certificate controllers consistently outperform all LLM methods at equal budget (Width-Cert mean regret 0.036 vs best LLM 0.045); LLM departments omit roughly half of decision-critical facts even under fully truthful disclosure; and selective disclosure raises regret only modestly and significantly only for individual coordinators (SAS-SC contrast C−A, Holm-adjusted p=0.015). Finally, relative to a neutral fixed-index truthful disclosure, selective disclosure nearly halves receiver regret (omission gap −0.043 to −0.058, 12/12 tests significant after Holm correction) — every statement being true is far from being enough to decide.

**Keywords:** multi-agent systems; strategic omission; selective disclosure; decision certificates; minimax regret

---

## 1 Introduction

### 1.1 A structural problem that actually occurs

Organizational decisions depend on data scattered across departments, each of which sees only part of the whole: the growth team emphasizes conversion rates, the operations team emphasizes completed workload, and the data-governance team emphasizes compliance and field completeness. Every number each department reports can be **true** — and the organization can still choose the wrong action.

The distinction at the core of this paper is:

$$
\text{factuality} \neq \text{disclosure sufficiency}.
$$

The first asks whether the stated facts are correct; the second asks whether unstated facts could still change the action that should be taken. When LLM agents play the role of departments holding private information, this distinction becomes urgent: an agent can tell the truth while strategically selecting which truths to tell. Remaining silent on decision-critical facts requires fabricating nothing.

### 1.2 From "everyone agrees" to "no unknown fact can overturn the decision"

Most multi-agent collaboration research sets the stopping condition at "consensus reached" or "discussion converged." We advocate a stopping condition that can be machine-checked:

> In every state still compatible with the valid evidence, the loss of the chosen action does not exceed $\varepsilon$.

This condition relies on no LLM self-reported confidence, no LLM-as-judge, and no human annotation. When it holds, the system attaches a **decision-sufficiency certificate**: a loss upper bound that anyone can recompute. When it does not hold, the protocol keeps verifying the facts that "could still overturn the current action" — because most missing facts, even if known, would not change the decision.

### 1.3 Research questions

- **RQ1:** When every disclosed fact is true, can selective disclosure still systematically increase organizational decision regret?
- **RQ2:** Can linguistic collaboration (longer discussion, structured objections, self-consistency sampling) substitute for a disclosure-sufficiency check at equal cost?
- **RQ3:** Is decision-relevant verification (multi-counterexample coverage queries) more effective than indiscriminate completion (uniform, max-width, single-counterexample)?

### 1.4 Contributions

- **C1: A truthfulness-fixed benchmark for strategic omission (TOG-IS).** Real business aggregate values, known organizational objectives, computable departmental utilities, controlled private views, fully automated verification of structured facts, paired interventions, and exact decision ground truth — distinguishing "the facts were wrong," "the facts were not disclosed," and "disclosed facts were not used correctly."
- **C2: An omission-aware decision-certificate protocol (ODC).** A minimal governance protocol combining an evidence ledger, explicit uncertainty over undisclosed information, counterexample-driven supplementary verification, and an action-release condition. Its mathematical skeleton comes from robust decision-making and preference elicitation; we do not claim to invent these foundations.
- **C3: A decomposition of manipulation cost.** We simultaneously measure wrong decisions, extra verification burden, certification coverage, and bounded deviation gain, separating "necessary information cost" from "redundancy caused by poor coordination."

---

## 2 Related Work

**Collective reasoning under distributed information.** HiddenBench [1] (Li et al., 2026, ICML) studies collective reasoning, unexplored information, and premature consensus in multi-agent LLMs, and proposes structured communication protocols. We keep its "premature consensus" problem awareness but add computable departmental utilities, strictly truthful structured disclosure, priced supplementary verification, and executable decision certificates: the question is not whether agents can integrate hidden information, but whether the stopping condition of integration is verifiable.

**Are multi-agent systems really stronger?** The Illusion of Multi-Agent Advantage [2] (Jwalapuram et al., 2026) shows that comparing multi-agent systems only against weak single agents leads to wrong conclusions. All of our main comparisons match permissions, available evidence, query cost, and token budget, and explicitly include strong single-agent baselines and a deterministic reference family.

**Propagation of unreliable information.** When Truth Is Distributed [3] (Yan et al., 2026) studies the propagation of false testimony in distributed fact recovery; its controlled framework ForesightSafety-TIDE is a close cousin of our fact channel. Our main experiment **forbids false facts from entering the shared fact channel**, narrowing the object of study to selective omission under truthful facts — a mechanistic isolation of the "unreliable information" problem.

**Failure taxonomies of multi-agent LLM systems.** Cemri et al. [4] (NeurIPS 2025) provide the first empirically grounded taxonomy of multi-agent LLM failures. We mechanize one specific failure mode — information-integration failure — into a controllable, auditable, and quantifiable business game.

**Mathematical foundations.** Minimax regret and incremental utility elicitation [5, 6, 7] supply the mathematical skeleton of our certificates; mechanism design with selective verification [8] offers the related perspective of "checking only critical reports." We claim no priority over these foundations; our novelty is a new instantiation of these ideas for LLM strategic-omission evaluation and governance.

**Proactive statement on competing work.** We note that Hu & Miao [9] have independently studied a problem close to ours in title (adaptive information procurement for exact decision certificates). Their work focuses on incentive-compatible mechanisms (payment design) when query costs are privately known; ours evaluates LLM departmental disclosure behavior under controlled motivation shifts and provides a governance protocol under a fixed-cost fact channel. The two are complementary. We do not claim priority over the broad concept of "decision certificates"; our contribution lies in the evaluation of strategic omission under truthful facts and its governance analysis.

---

## 3 Task and Game: TOG-IS

### 3.1 Data and task generation

Tasks derive from two open datasets: **Online Retail** (UK e-commerce transactions, mapped to three benefit-direction metric columns: growth / operations / governance) and **Bank Marketing** (Portuguese bank telemarketing, mapped identically). Each task generates $K=8$ candidate business groups from a data block; group $k$ carries three metrics $\boldsymbol z_k=(z_{k1},z_{k2},z_{k3})\in[0,1]^3$, flattened as $\boldsymbol z\in[0,1]^{24}$. The fixed split yields **240 formal test tasks** (a dev set is reserved for prompt tuning; dev tasks are protocol-barred from formal results).

### 3.2 Players, actions, and objectives

Three department agents $\mathcal N=\{1,2,3\}$ (growth / operations / governance). Department $i$ directly sees only its own column $\{z_{ki}\}_{k=1}^8$ — data silos are enforced by session isolation and prompt-level permissions. The organization selects $h$ of the 8 groups ($h\in\{1,2\}$, $|\mathcal A_1|=8$, $|\mathcal A_2|=28$); the public weight $\boldsymbol w$ takes the six permutations of $(0.5,0.3,0.2)$. Organizational score

$$
W(a;\boldsymbol z)=\frac{1}{h}\sum_{k} x_{ak}\sum_i w_i z_{ki}
$$

is affine in $\boldsymbol z$, i.e., $W(a;\boldsymbol z)=\beta_a+\boldsymbol\theta_a^\top\boldsymbol z$ (with $\beta_a=0$ in the main task), the key to exact interval regret bounds. Departmental KPI is $V_i(a;\boldsymbol z)$ and departmental reward is $U_i=(1-\rho)W+\rho V_i$.

### 3.3 Fact channel and evidence receipts

Free text never becomes fact in the main experiment: all facts are obtained as structured query IDs (`q_group_department`) through an **evidence sidecar**, and every query returns a **receipt** binding the task-state hash and the value. Every query value appearing in a log must equal its receipt value — this audit runs automatically after every trajectory; all 8,640 trajectories (3 seeds × 2,880) passed without a single violation. Queries are billed per independent fact; batched responses accumulate per-fact cost.

### 3.4 Three disclosure conditions (Protocol 14.1)

Under each (task, condition), the three departments each disclose exactly 2 facts of their own column (6 total), forming the initial evidence $E_0$:

- **C (cooperative, $\rho=0$):** LLM departments are asked to disclose the 2 facts "most useful for the organizational decision."
- **S (departmental, $\rho=1$):** LLM departments advance their own KPI within legality (true facts, exactly 2, own column); whether strategic omission emerges is a measured behavior, not an assumption.
- **A (scripted selective disclosure):** a model-independent script — each department discloses its two highest-metric facts and recommends its own best action; fixed, interpretable, and blind to the global optimum.

For each (task, condition, seed), $E_0$ is generated once and shared by **all** receiving methods (Protocol 14.2), ensuring we compare "facing the same disclosure, how does each protocol complete and decide." Note (Protocol 14.3) that C/S differ from A in both disclosed content and recommendation mechanism; the contrast is not a pure causal effect of omission. We therefore additionally run a receipt-swap intervention (Section 6.6).

---

## 4 Omission-aware Decision Certificates (ODC)

### 4.1 Evidence-compatible states and regret bounds

Given receipt values of revealed facts, unrevealed facts keep their valid ranges $[\ell_j,u_j]$, inducing the evidence-compatible set $\mathcal Z$. For candidate action $a$ and competitor $b$, the pairwise worst-case advantage is attained at a vertex under box constraints (the objective is affine in $\boldsymbol z$) and computed in exact integer arithmetic. The regret upper bound of action $a$ is

$$
\overline{\mathrm{Reg}}(a)=\max_{b\in\mathcal A}\ \mathrm{PW}(a,b),
$$

the selection rule takes the minimax-bound action with deterministic tie-breaking, and an action is **certified** when its bound $\le\varepsilon$ (valid boundary). Correctness relies on receipt authenticity (fact channel) and boundary validity (unknown values within ranges); we make no safety claims outside this class.

### 4.2 Multi-counterexample coverage queries

Each step selects the unknown fact that simultaneously compresses the most not-yet-excluded counterexamples (multi-counterexample coverage score), rather than re-checking what departments already said. This heuristic shares the same certifier with three query baselines, swapping only the query selector (fixed-controller layer, Protocol 15.3): single-counterexample (ODC-CS, worst competitor only), uniform random (Uniform-Cert), and max interval width (Width-Cert).

### 4.3 Proven properties (validated on 1,000 random CPU instances)

1. Conditional decision-loss guarantee: a certified action's true regret does not exceed $\varepsilon$;
2. Monotonicity: new valid evidence never increases the minimum attainable regret;
3. **Fact verification alone does not guarantee correct decisions** (there exist instances where all disclosed facts are true yet the disclosure set cannot support the optimal decision);
4. Exact optimality after full legitimate disclosure;
5. Organizational damage from strategic disclosure is bounded, but private gain need not vanish.

---

## 5 Experimental Design

### 5.1 Ten methods and three fairness layers

**LLM coordination methods (4):** SAS-Tool (a strong single coordinator with read-only query tools and a deterministic scorer); SAS-SC (single coordinator with self-consistency; sample count fixed by budget); MAS-Deliberate (three departments exchange evidence and discuss; the coordinator decides further queries and the action); Exchange-then-Decide (ETD, a tool-constrained adaptation of HiddenBench's "exchange–object–decide" structure).

**Deterministic references (6):** ODC-WC (our full query), ODC-CS, Uniform-Cert, Width-Cert, NoLLM-E2E (fully scripted departments, all-algorithm reference), and FullInfo (reads all 24 facts; an oracle for solvability checking, not an equal-information competitor).

Fairness is enforced at three layers: end-to-end (same catalog, permissions, objective, $E_0$, resource caps); fixed-evidence (identical evidence sequences isolate acquisition from integration); fixed-controller (only the query selector changes).

### 5.2 Budget and metrics

Unified query budget $B=24$ (including the 6 initial disclosed facts) with checkpoints at 6/12/18/24 under an anytime protocol. Token budget hard cap: 12,288 in / 1,536 out per trajectory, accumulated across calls, with circuit-breaking on exceedance. Primary metric: organizational regret (macro-averaged, stratified by dataset and condition). Secondary: certification coverage, query cost, fallback rate, and bounded deviation gain (M5, Section 6.5).

### 5.3 Models, randomness, and statistics

The main model is a locally deployed Qwen3-4B-Instruct-2507 (GGUF Q4_K_M, RTX 5060 8GB; offline inference — zero API cost, GPU time still counted). Decode seeds are preregistered and frozen at {17, 29, 43}; all prompts were frozen after dev-set tuning (registry: preregistration config) and LLM methods repeat each (task × condition × method) per seed. Deterministic methods are not pseudo-replicated; they are evaluated separately per seed-generated $E_0$ (Protocol 15.2). Statistics: paired tests (same task, method, and seed) with Holm correction for multiple comparisons.

---

## 6 Results (seeds {17, 29, 43}, n=720 per cell)

### 6.1 Main table (budget B=12; LLM at natural stopping cost)

Full table: `analysis/main_table_final.md`. Key readings:

- **The four LLM methods are quality-equivalent:** under condition A, mean regret SAS-Tool 0.0490 / SAS-SC 0.0454 / MAS 0.0541 / ETD 0.0498, with no significant pairwise differences; **cost structure differs enormously** — SAS ≈ 16 queries, MAS ≈ 9.7, ETD ≈ 7.8 (50%+ savings), and per-cell means are nearly identical across seeds. Multi-agent deliberation reaches similar quality with far fewer queries; this is the most robust finding of the study.
- **Deterministic controllers beat all LLM methods at B=12:** Width-Cert 0.0359, ODC-WC / NoLLM-E2E 0.0377, ODC-CS 0.0423, versus the best LLM 0.0454. In this 24-fact state space, structured querying still dominates on decision quality; the value of LLM methods lies in cost structure and (expected) generality, not quality on this benchmark. Uniform-Cert (0.0603) is significantly worse than every decision-relevant strategy — supporting RQ3: *what* you ask matters more than *how much* you ask. These deterministic readings are nearly seed-invariant (e.g., Width-Cert under A is 0.0359 in all three seeds).
- **FullInfo oracle regret is 0**; task solvability verified.

### 6.2 Effect of disclosure condition (Figure 1)

The ranking remains roughly A ≲ S < C, directionally consistent with the seed-17 preview, but **the significance pattern has changed**:

- **ETD's C−A difference did not replicate.** It was the only uncorrected-significant contrast at seed 17 (+0.0071, p=0.013), but neither seed 29 (p=0.37) nor seed 43 (p=0.22) reproduces it; merged, +0.0028, p=0.13, not significant after Holm correction. The seed-17 "significance" was small-sample fluctuation; for ETD, "cooperative disclosure is harmful" must be reported as a directional trend only.
- **SAS-SC's C−A difference is significant in the merged statistics** (+0.0068, t=3.46, p=0.0006, Holm-adjusted p=0.015), direction-consistent per seed (+0.0010 / +0.0122 / +0.0074 at seeds 17/29/43, two of three individually significant). It is the only surviving LLM-side condition effect among all 27 tests — single-coordinator methods are more vulnerable to cooperatively disclosed content.
- **On the deterministic side, Width-Cert's S−C difference is significant** (−0.0026, p=0.0007, Holm-adjusted p=0.019, n=720): regret under S (departmental self-interest) is lower than under C (cooperative), consistent with the direction of the disclosure-behavior evidence M2 in 6.5.
- ETD's S−A reaches p=0.049 uncorrected but does not survive Holm. Cost structure is condition-robust (MAS ~9.5, ETD ~7.7, SAS ~16): condition effects, where present, live on the quality side, not the behavior side.

Full tests: `analysis/paired_tests_final.md`; per-seed checks: `analysis/cross_seed_check.md`.

### 6.3 The anytime view (Figure 3)

Deterministic regret decreases monotonically with budget; LLM natural-stopping points lie **above** the curves: at comparable cost, LLM methods systematically incur more regret than the curve readings of Width-Cert/ODC-WC; MAS has the lowest median regret but a higher mean — its distribution shifts left with a fatter tail (when premature consensus anchors on the wrong option, it loses bigger).

### 6.4 Certification coverage and the ε-calibration sweep (Figure 5)

At the preregistered ε=0.05, B=12 certification tops out at 2.8% — the certificate almost never releases. We re-ran all deterministic trajectories at 7 ε values (0.005–0.20) × 3 certifier-family methods × 240 tasks × 3 conditions × 3 seeds (181,440 rows):

1. **The guarantee holds empirically**: across 181,440 rows, the number of certified actions whose true regret exceeds ε is **zero** — Proposition 1 validated at empirical scale.
2. **Budget, not threshold, is the binding constraint:** at B=24 (complete information) certification is 100% at every ε (bounds collapse to zero); at B=12, even ε=0.20 certifies only 11%–22%. At half budget, what blocks release is residual uncertainty itself.
3. **Bounds are conservative:** certified actions' mean true regret sits far below ε (e.g., 0.035 at ε=0.10).

Implication: at this task scale, ODC's practical function is not fast release but **steering verification budget toward unknown facts that can still change the action**. Together with Uniform-Cert's significantly worse performance (6.1), this answers RQ3 affirmatively — and honestly bounds C2: the "release" function needs larger fact spaces to become visible.

### 6.5 Departmental disclosure behavior: behavioral evidence of omission (Figure 6)

Measuring the 6,480 real departmental disclosures (240 tasks × 3 conditions × 3 seeds):

- **M2 (core evidence):** the departmental facts of the organizationally optimal group appear in $E_0$ only A 0.499 / C 0.457 / S 0.476 of the time — even when every disclosure is true and departments are asked to cooperate "most usefully," more than half of the decision-critical facts are undisclosed. The direction C < A matches the regret trend in 6.2, offering a mechanistic clue for why LLM cooperative disclosure is not better: **facts LLMs consider "useful" are not the facts that are decision-relevant.**
- **M1:** departments disclose their own top-KPI group ≥98% of the time in all conditions — self-interest does not change "showcasing" behavior; omission concentrates in lower ranks, not at the top.
- **M3 / M4:** Jaccard overlap between LLM and scripted-A $E_0$ is 0.70–0.72 — LLMs mostly follow a "pick the top two" pattern with ~30% deviation; C has zero fallback generations, S falls back 5.9% (under the S prompt the model more often produces illegal disclosures — self-interest increases "strategic expression" attempts, which we analyze as behavioral evidence of strategic omission).
- **M5 (deviation gain, the fourth measurement of C3):** define department $i$'s deviation gain as $\Delta_i = V_i(\hat a) - V_i(a^*)$ — the change the chosen action brings to the department's KPI relative to the organizationally optimal action (theoretically bounded by 1) — and the trajectory-level gain as $\max_i \Delta_i$. Mean deviation gain is small and nearly identical across conditions (A +0.0035 / C +0.0034 / S +0.0036; no condition contrast survives Holm correction); in 67–70% of trajectories at least one department benefits from the deviation, and deviation gain correlates significantly with organizational regret (r=0.444, n=8,640). **Private dividends are real but bounded, and are not amplified by strategic-disclosure conditions** — they look like a by-product of decision errors rather than a payoff to omission. Full numbers: `analysis/deviation_gain.md`.

### 6.6 Receipt-swap intervention: the causal read (Figure 8)

Following Protocol 14.3, we freeze the receiver, the true state, the fact count, and the format, and swap only the disclosed fact set. Deterministic receivers satisfy the freeze by construction (their decisions ignore recommendations; prompts are constant), so the three-condition contrast in `formal_deterministic.jsonl` *is* this intervention. Across 3 methods × 3 contrasts (9 paired tests, three seeds merged, n=720 per test), all effects are ≤0.003 in magnitude; **only Width-Cert's S−C contrast survives Holm correction** (−0.0026, adjusted p=0.007) — the same contrast significant in 6.2 — while the remaining eight are non-significant. Causal reading: swapping the disclosed fact set alone moves a deterministic receiver's decision only marginally, with the single exception of the S-versus-C content difference under max-width querying; if the LLM-side condition trend is real, it must act through the **interaction between disclosure content and LLM reasoning**. This narrows the space of mechanistic explanations. The LLM-receiver version of the swap experiment has been completed; see Section 6.7.

### 6.7 The LLM-side omission gap: the neutral-disclosure arm (Figure 9)

The omission gap required by the pre-registration (Protocol 16.3) is now closed on the LLM receiver side: pairing on (task, method, seed), we compare the receiver's regret when facing the selective $E_0$ (the three-condition content of the main batch) versus a neutral $E_0$ (a fixed set of the first two fact indices per department, independent of the seed), $\mathrm{TOG} = \mathrm{Reg}(E_{\mathrm{selective}}) - \mathrm{Reg}(E_{\mathrm{neutral}})$, n=720 per cell. All 2,880 neutral-arm trajectories (4 methods × 3 seeds × 240 tasks) pass the audit with zero fallbacks.

- **All 12 TOG values are negative and highly significant**: −0.043 to −0.058 (|t| = 14.6–19.2, Holm-adjusted p ≤ 1.9×10⁻⁴²), and every one of the 12 tests is directionally consistent across all three seeds.
- **Magnitude**: the neutral arm nearly doubles LLM regret — Reg(neutral) 0.095–0.108 vs Reg(selective) 0.045–0.056. The inter-condition differences of Section 6.2 (~0.005) are only about 1/10 of the "selective vs neutral" gap: **whether the disclosed content is decision-relevant matters far more than which selective strategy produced it**.
- **Mechanistic placement**: against the deterministic-receiver swap effect of ≤0.003 (Section 6.6), the same manipulation is an order of magnitude larger on the LLM receiver — confirming Section 6.6's inference that condition effects act through the interaction between disclosure content and LLM reasoning; it also shows that under truthfulness fixed at 1, the largest single-factor source of regret is not strategic motive but the **informational poverty of the neutral fixed disclosure itself**.
- **Convergent with the negative omission tax on the deterministic side (Section 6.8)**: at an identical fact count (6 each), the selective disclosure set is both cheaper (0.80–0.84 fewer facts to complete) and lower-regret than the neutral fixed index — "every statement true" is far from "enough to decide." This is the core experimental evidence for the paper's title claim.

### 6.8 Minimum certificate completion cost and the omission tax (Figure 7)

We compute, per (task, condition, seed), the theoretical minimum number of additional facts required to certify an action (exact MILP via HiGHS; 2,880 cells across three seeds, 100% optimally solved; ε=0.05):

- Certifying *some* certifiable action costs C* ≈ 9.9 further facts on average; certifying the *true optimal* action costs ≈10.0–10.2. Certificates are intrinsically expensive at this ε (6 + 10 ≈ 16 of the 24 budget), explaining the 2.8% B=12 certification rate.
- **The omission tax is negative and highly significant:** all three disclosure conditions are 0.80–0.84 facts *cheaper* to certify than neutral frozen-index disclosure (|t| > 25, n=720 per condition, paired). Strategic "showcase" disclosure leaks more decision-relevant information than neutral indexing — counterintuitive but coherent, and we discuss it head-on.
- **CostRatio** (ODC-WC actual spend / theoretical minimum) ≈ 1.17–1.18: the multi-counterexample coverage query spends only ~18% above the lower bound — it asks nearly optimally.

### 6.9 Stratified robustness

Stratifying by dataset (Retail/Bank) and action size (h=1 vs h=2): LLM mean regret is 0.0388 at h=1 versus 0.0604 at h=2 (action space 8 vs 28), consistent with earlier observations; all three qualitative conclusions (deterministic controllers dominate on quality, multi-agent saves queries, condition differences are small) hold in both datasets. Meanwhile, the regret ranking of the four LLM methods is **unstable across all 4 strata × 3 conditions** — ranking shuffling is independent evidence that quality differences among LLM methods are noise.

### 6.10 Robustness Under Stress Conditions (Protocol 14.4, descriptive)

Following the preregistered four stress conditions (duplicate facts, action-ID permutation, two-department coalition disclosure, sequential authorization denial) × 8 tasks per domain × 2 receivers (SAS-Tool and ODC-WC) × 1 random configuration (seed 17), we run 128 descriptive trajectories (`analysis/stress_summary.md`):

- **Robustness floor:** across all 128 trajectories there are zero fallbacks, zero illegal actions, and zero audit failures — the receipt channel is idempotent under duplicated facts (duplicated entries are not double-charged), and under sequential authorization denial (the first 3 query requests are refused at no budget cost) both receivers reroute and still complete their decisions.
- **Action-ID permutation:** ODC-WC's per-task regret is exactly invariant (16/16, as expected for a deterministic receiver); SAS-Tool's regret changes under relabeling in 10/16 tasks — LLM coordinators exhibit label sensitivity, another behavioral difference from deterministic receivers.
- **Two-department coalition disclosure:** for the growth+operations alliance (sharing both departments' private information but unable to see the governance department's hidden data; threat capability labeled separately from the single-department conditions), all 16 E0 sets are LLM-generated with zero fallbacks, and receiver regret stays close to the A baseline (SAS-Tool 0.0492 vs 0.0449, descriptive).
- All regret/cost shifts are small-scale descriptive readings; they enter no confirmatory test family and do not replace the main statistical conclusions.

---

## 7 Discussion and Limitations

1. **External validity of the omission gap:** the neutral fixed-index disclosure is one specific uninformative-but-truthful baseline; the roughly 2× regret amplification it induces (Section 6.7) may not transfer to other neutral baselines, and we do not report it as a universal constant.
2. **Small state space:** 24 facts let deterministic methods dominate; LLM comparative advantages are expected at larger, more heterogeneous fact spaces (future work).
3. **4B local model:** sensitivity to model scale is untested (small-scale transfer tests are preregistered).
4. **Condition contrasts are not pure causation** (Protocol 14.3); the receipt-swap intervention (6.6) localizes the causal component for deterministic receivers.
5. **Semi-synthetic setting:** departments are modeling roles; we do not claim the source organizations used these departments, and we report no causal ROI or real future profit.
6. **The guarantee is conditional:** certificates assume receipt authenticity and valid ranges; we make no safety claims beyond this class.

## 8 Conclusion

Truth is not sufficiency. On a truthfulness-controlled fact channel, selective disclosure lets organizations make suboptimal decisions while every sentence remains literally true; linguistic negotiation reduces query cost but cannot replace the check of whether undisclosed facts could still overturn the decision. Decision-sufficiency certificates rewrite the stopping condition of collaboration from consensus to a recomputable loss bound, and convert the cost of manipulation into a measurable verification burden. Three-seed statistics support this qualitative picture: multi-agent saves queries, deterministic verifiers dominate on quality, and literal truthfulness still falls short of decision sufficiency — while the harm from disclosure conditions, though real, is modest and uneven across coordinators.

---

## Figures and Tables (from `week1/analysis/`)

- Figure 1 `fig_regret_by_condition_final.png`: mean true regret by method × condition (95% CI, three seeds)
- Figure 2 `fig_cost_final.png`: query-cost comparison (three seeds)
- Figure 3 `fig_anytime_final.png`: anytime curves + LLM natural-stopping points (key figure; three seeds)
- Figure 4 `fig_regret_dist_final.png`: regret-distribution box plots (three seeds)
- Figure 5 `fig_eps_sweep.png`: certification rate vs ε (B=12 / B=24; three seeds, 181,440 rows)
- Figure 6 `fig_disclosure.png`: departmental behavior across conditions (M1/M2/M3; three seeds, 6,480 disclosures)
- Figure 7 `fig_min_cost.png`: minimum completion-cost distribution and omission tax (three seeds)
- Figure 8 `fig_receipt_replacement.png`: receipt-swap causal effects (deterministic receivers; three seeds)
- Table 1 `main_table_final.md`: main table (final; LLM n=720, deterministic n=720)
- Table 2 `paired_tests_final.md`: condition-difference paired tests (Holm-corrected, final)
- Table 2b `cross_seed_check.md`: per-cell cross-seed stability check (LLM methods)
- Table 3 `eps_sweep.md`: full ε-calibration sweep (three seeds)
- Table 4 `disclosure_behavior.md`: departmental disclosure behavior metrics (three seeds)
- Table 4b `deviation_gain.md`: bounded deviation gain M5 (three seeds, 8,640 trajectories)
- Table 5 `main_table_stratified.md`: stratified main table (dataset × h, three seeds)
- Table 6 `min_cost.md`: minimum completion cost and omission tax (three seeds)
- Table 7 `receipt_replacement.md`: receipt-swap paired tests (three seeds)

## References (verified 2026-10-12)

[1] Li, Y., Naito, A., & Shirado, H. Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs (HiddenBench). ICML 2026; arXiv:2505.11556. (Acceptance independently confirmed 2026-10-12; the ICML version carries an updated title.)

[2] Jwalapuram, P., Lin, H., Li, C., Jiao, F., Wang, S., et al. The Illusion of Multi-Agent Advantage. arXiv:2606.13003, 2026.

[3] Yan, C., Yue, Z., Zhao, F., Lin, E., Jia, L., Tong, H., et al. When Truth Is Distributed: Misinformation Derails Collective Fact Recovery in LLM-Based Multi-Agent Systems. arXiv:2608.03421, 2026.

[4] Cemri, M., Pan, M. Z., Yang, S., et al. Why Do Multi-Agent LLM Systems Fail? NeurIPS 2025 Datasets and Benchmarks Track.

[5] Wang, T., & Boutilier, C. Incremental Utility Elicitation with the Minimax Regret Decision Criterion. IJCAI 2003.

[6] Boutilier, C., Patrascu, R., Poupart, P., & Schuurmans, D. Constraint-based optimization and utility elicitation using the minimax decision criterion. Artificial Intelligence, 2006.

[7] Braziunas, D., & Boutilier, C. Minimax Regret Based Elicitation of Generalized Additive Utilities. UAI 2007.

[8] Fotakis, D., Tzamos, C., & Zampetakis, M. Mechanism Design with Selective Verification. EC 2016.

[9] Hu, J., & Miao, C. Truthful Adaptive Information Procurement for Exact Decision Certificates. SSRN 7529186.
