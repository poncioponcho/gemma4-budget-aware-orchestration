## Abstract

Tool-calling agents for repository-level issue repair operate under hard budgets on tool calls and time. Whether orchestration built on top of a fixed tool set converts budget into resolved issues, rather than into exploration churn, is an empirical question that leaderboard-style evaluation rarely isolates. We present a budget-controlled four-arm ablation on the official Gemma 4 Developer Agent harness: a fixed-order single-agent baseline at 10 tool calls (B0), the same agent at 20 calls (B0′), a planner→coder two-phase arm (B1-A), and a writable-root topology repair (B1-A′), all under one model, one 128-task set, and a pre-registered decision line (≥10 percentage points, McNemar p<0.05). On the 65-task development split, resolution rates were 3.1%, 6.2%, 10.8%, and 9.2%; no arm pair met the line. A frozen 63-task held-out run (B0′ 12.7% vs B1-A′ 15.9%, +3.2 pp, p=0.73) did not change the verdict. Mechanistic forensics explain the negative result: the planner consumed a median of 14 of 20 billed calls against a prompt-stated share of 8; 29 of 65 tasks submitted empty patches; and 13 tasks submitted without any edit although their budget was spent. The repair cut empty submissions from 45% to 32% and handoff delay from 14 to 0 calls; strict B1-A′ abandonment was still 30.8% (20/65), while B0′ had 22% production no-patch cases (14/65), so the structural gains did not meet the pre-registered thresholds. We conclude that on this model, prompt-level quantitative constraints and behavioral prohibitions are non-binding, and that a session-level shared budget provides no per-agent enforcement — a boundary result for multi-agent designs, not a claim that division of labor is useless. All results come from self-executed runs of the official evaluation harness.

## 1. Introduction

Agents that fix repository issues by calling tools — search, read, edit, run tests, submit — have become a standard LLM application, and their evaluation is dominated by aggregate pass-rate leaderboards. That format hides a structural question: given a fixed tool set and a fixed budget, *who spends the budget, and on what*. When a budget is small, the dominant failure mode is truncation; when it is large, the dominant failure mode shifts rather than disappears. Orchestration — planner/coder division of labor, handoff timing, context compression — is supposed to decide how budget is converted into patches. But orchestration claims are typically confounded with budget changes, because adding structure usually comes with more calls or more tokens.

We isolate the two factors on the official Gemma 4 Developer Agent harness. We keep the model, the official tool universe, the task set, and the harness fixed; B0→B0′ measures the frozen budget-envelope change, while B0′→B1-A/B1-A′ compares orchestration at the same 20-call envelope. Four arms are compared on a 65-task development split, and the final comparison is executed once on a frozen 63-task held-out split under a protocol registered before the run.

The baseline motivates the study. At the default 10-call budget, 57% of tasks exhaust their tool budget; 36 of those 37 truncated tasks submit nothing at all — and fastapi issues are solved at 0/34. Yet the two baseline successes are rich issues resolved in six calls with small, early-submitted patches: the budget is not a capability ceiling but an allocation failure. Doubling the budget removes truncation (57%→2%) but converts it into context-overflow deaths (5%→35%) and yields only +3.1 pp (not significant). Orchestration should therefore be judged on whether it spends context more economically, not on whether it spends more.

This paper makes four contributions. (i) A budget-controlled, four-arm ablation that separates budget from orchestration on a real agent harness, with a pre-registered decision line and a frozen held-out protocol. (ii) Trace-level forensics that locate *why* the orchestration arm underperforms its design: a prompt-level budget share that the model violates by 75%, and empty patch submissions that survive an explicit prohibition. (iii) A documented, hash-pinned method freeze and one-shot held-out run, so every reported number is reproducible from archived artifacts. (iv) An honest negative result with a failure taxonomy and two trace-level case studies, plus a transferable warning: under a session-level shared budget with no per-agent accounting, prompt-level constraints on sub-agents are advisory, not enforced.

## 2. Methods and Experiments

**Task set and split.** 128 multi-repository issue-fixing tasks (fastapi 67, rich 48, requests 13) from the competition, excluding one singleton httpx task. We split within each repository by `sha256(instance_id)`: the first ⌈n/2⌉ tasks form development (fastapi 34, rich 24, requests 7; n=65) and the remainder held-out (33/24/6; n=63). Development was used for prompt and orchestration iteration; held-out pass/fail was not read until the method was frozen. The frozen split hashes are development `0cd2371a9a4bd5af3280e03948a719af04fed5ef4c7f72811d00a08d1997a3b5` and held-out `4600b8c12aa64d3739e94101625960f006c79fb253cf2a6715568a52d4f148db`.

**Evaluation.** We use the official `swegemma` evaluation harness. A task is *resolved* iff pytest exits 0, the JUnit XML is valid with `passed>0`, `failures=0`, `errors=0`, and all required tests (extracted from `test_patch`) explicitly pass without skipping. All runs were executed by us with the official harness on Kaggle (4× NVIDIA L4); no competition-side verifier was invoked. The frozen pre-held-out inventory contains 195 task records, all with `official_pass=null`; the held-out artifacts likewise carry null, so every number here is a self-run result, not a leaderboard score.

**Arms.** Table 1 defines the four arms. All arms use the same nine-tool universe, the same model (`gemma-4-31b-it-qat-w4a16-ct`), and identical sampling (`thinking_budget=0`, `include_thoughts=false`, `max_output_tokens=512`). The frozen evaluation limits are B0: 10 calls/1 minute/50 turns; B0′, B1-A, and B1-A′: 20 calls/3 minutes/60 turns. Thus the primary 20-call contrasts share the full evaluation envelope; B1-A and B1-A′ assign the shared tools across root and sub-agent roles as shown. B1-A implements three mechanisms on top of B0′: (A) static planner→coder division of labor, (C) a guaranteed handoff with a submission floor, and (D) token economy via plan handoff instead of raw transcript forwarding; mechanism B (dynamic budget adaptation) is deliberately absent from both orchestration arms. B1-A′ keeps mechanisms A/C/D but moves the agent that holds `submit_patch` to the root, so the terminal agent can never receive zero remaining budget.

**Table 1. Frozen arm definitions.** Tool assignments and evaluation values are the archived `b0prime`, `b1`, and `b1a2` bundles plus the B0 sample freeze manifest.

| Arm | Topology | Tool calls | Mechanisms | Role |
|---|---|---|---|---|
| B0 | single agent | 10 | none (fixed order) | official-style baseline |
| B0′ | single agent | 20 | none | budget control |
| B1-A | planner→coder | 20 | A+C+D | orchestration |
| B1-A′ | root agent + read-only planner sub-agent | 20 | A+C+D | topology repair |
<!-- -->

**Statistics.** Resolution rate is a per-task binary outcome; we report Wilson 95% intervals and same-task paired McNemar exact tests. The decision line was pre-registered on 2026-10-15 and never changed: an orchestration arm beats B0′ only if the resolution-rate gain is ≥10 pp *and* McNemar p<0.05 (the exact conditional-binomial p-values under both doubling and standard two-sided conventions must agree). On n=65 one task flip is ≈1.5 pp, so the line is deliberately conservative. Mode shares, token medians, and per-repository breakdowns are reported alongside point estimates.

**Freeze and held-out protocol.** After development iterations, the task files, per-arm bundles (prompts, tool lists, eval configs), budgets, sampling overrides, analysis scripts, and the decision line were frozen with SHA-256 manifests (`freeze.py` verifies; exit=0 required before any run). The held-out comparison (B0′ vs B1-A′) was then executed once per arm, 63 tasks each, with no intermediate inspection. Integrity gates before analysis: 63/63 completed task runs and traces, correct batch index, 4×L4 hardware guard, and zero files corrupted by proxy 502 pages. The B1-A′ kernel ended in a harness `ERROR` produced by a summary-cell classifier meeting an unregistered error string *after* all 63 task runs had been written; because the per-task data were complete, we did not re-run it. Code, per-task records, the frozen manifests, and a reproduction notebook that re-derives every number in this paper are public: https://github.com/poncioponcho/gemma4-budget-aware-orchestration and https://www.kaggle.com/code/daniel1547/gemma-4-budget-aware-orchestration-reproduction.

## 3. Results and Analysis

**Development split (Table 2).** Doubling the budget (B0→B0′) buys +3.1 pp (p=0.50) while context overflow rises from 5% to 35% and the prompt-token median grows 57%: budget is converted into context death, not resolution. Two-phase orchestration (B1-A) reaches 10.8% but is not significant against B0′ (+4.6 pp, p=0.4531), and its abandonment share climbs to 43%. The topology repair (B1-A′) lands at 9.2% (+3.1 pp, p=0.6250). No arm pair meets the pre-registered line.

**Table 2. Development outcomes and failure modes (n=65).** The B0′ row is compared with B0; both orchestration rows are compared with B0′. “Abandoned / no patch” here uses the production classifier; Table 4 reports the strict no-error/no-patch definition. Sources: `c3_three_arm_n65_20261016.log` and `analyze_ablation_B0p_vs_B1A2_20261017.log`.

| Arm | Resolved | Wilson 95% CI | Δ vs comparator | McNemar p (paired comparator) | Context overflow | Abandoned / no patch (production) |
|---|---|---|---|---|---|---|
| B0 | 2/65 (3.1%) | [0.8, 10.5] | — | — | 3 (5%) | 5 (8%) |
| B0′ | 4/65 (6.2%) | [2.4, 14.8] | +3.1 pp vs B0 | 0.50 vs B0 | 23 (35%) | 14 (22%) |
| B1-A | 7/65 (10.8%) | [5.3, 20.6] | +4.6 pp vs B0′ | 0.4531 vs B0′ | 15 (23%) | 28 (43%) |
| B1-A′ | 6/65 (9.2%) | [4.3, 18.7] | +3.1 pp vs B0′ | 0.6250 vs B0′ | 16 (25%) | 22 (34%) |
<!-- -->

Per repository (development), B1-A resolves fastapi 3/34, rich 3/24, requests 1/7; B1-A′ resolves 2/34, 3/24, 1/7 — no arm escapes the fastapi difficulty. B1-A is also internally uneven (run A 1/10, run B 3/20, run C 3/35 resolved; abandonment 4/7/17 per run), so the 10.8% total must not be read as a stable capability.

**Why B1-A underperforms its design (forensics).** The 65 traces show the two-phase split was triggered on 52 tasks but never delivered as budgeted: the planner consumed a median of **14 of 20** billed calls against a prompt-stated share of **8** (hard cap 12), leaving the coder a median of 6 (design 12); 24 of 52 handoffs arrived with ≤4 calls left. Patch production collapses monotonically with handoff delay (Table 3). Separately, 29/65 tasks called `submit_patch` with `agent_patch_size=0` — empty submissions against an unmodified tree.

**Table 3. B1-A handoff timing and patch production.** Values are from the 52 tasks that called the coder; billed calls exclude `submit_patch` and `get_status`.

| Handoff timing (billed calls) | n | Tasks with a patch | Resolved |
|---|---|---|---|
| 0–7 | 10 | 70% | 20% |
| 8–13 (design band) | 14 | 64% | 21% |
| 14–17 | 17 | 18% | 12% |
| 18–20 | 11 | 0% | 0% |
<!-- -->

**What the repair fixed, and what it did not (Table 4).** B1-A′ eliminates the zero-remaining-budget path: the first planner call now happens at a median of 0 billed calls (was 14), empty submissions fall 45%→32%, and the prompt-token median falls from 128,526 to 99,443. Two pre-registered structural thresholds are still missed (empty ≤5/65; strict abandonment ≤15%), and resolution does not improve (6/65; paired vs B0′: p=0.6250). Decomposing the 21 empty submissions is the paper's key finding: 13 of them **never called any edit tool**, directly violating the root prompt's prohibition ("never submit before editing at least one file") — and these 13 tasks did not lack budget (a median of 21 chargeable calls against a 20-call cap; a step may batch several calls, so per-call counts can exceed the cap; tasks with patches spent a median of 19). The failure is behavioral, not architectural and not budgetary.

**Table 4. Development structural diagnostics (n=65).** Empty means `submit_patch` was called with a zero-byte patch; strict abandonment means no error, no patch, and unresolved. Sources: `AUDIT.md` §§11.43–11.46 and the archived B1-A/B1-A′ traces.

| Metric (development, n=65) | B1-A | B1-A′ | Threshold | Met? |
|---|---|---|---|---|
| First planner at billed-call median | 14 | 0 | ≤5 | yes |
| Empty `submit_patch` | 29/65 (45.0%) | 21/65 (32.3%) | ≤5/65 | no |
| Strict `abandoned_no_patch` | 28/65 (43.1%) | 20/65 (30.8%) | ≤15% | no |
| Non-empty patch | 19/65 | 22/65 (33.8%) | — | — |
| Prompt-token median | 116,911 | 99,443 | — | — |
<!-- -->

**Held-out validation (Table 5).** Executed once after freezing, blind to intermediate results: B0′ 8/63 (12.7%) vs B1-A′ 10/63 (15.9%), +3.2 pp, McNemar 3:5, exact two-sided p=0.7266. The direction matches development; the magnitude does not meet the pre-registered line. The structural gains replicate: context overflow 24→8 and prompt-token median 120,473→102,138 (−15%), while no-patch abandonment rises 9→20 and the per-repository picture is mixed: rich improves (3/24→6/24), fastapi decreases (3/33→2/33), and requests stays at 2/6 in both arms.

**Table 5. Frozen held-out comparison (n=63 per arm).** All values come from `heldout_stats_B0p_vs_B1A2.json`; “Abandoned / no patch” is the archived production failure-mode count; the run was analyzed only after both task sets were complete.

| Metric (held-out, n=63) | B0′ | B1-A′ |
|---|---|---|
| Resolved (Wilson 95% CI) | 8/63 (12.7%) [6.6, 23.1] | 10/63 (15.9%) [8.9, 26.8] |
| Non-empty patch | 25/63 | 31/63 |
| Context overflow | 24 | 8 |
| Abandoned / no patch | 9 | 20 |
| Prompt-token median | 120,473 | 102,138 |
<!-- -->

**Failure taxonomy and cases.** Across arms the recurring terminal states are: budget truncation (exploration exhausts calls), patch-fails-tests, context overflow (session-killing at the 16k window), and no-patch abandonment. Two development traces illustrate the extremes. `fastapi_14349`: 14 steps of exploration and localization, zero patch, truncated — with 118,151 prompt tokens against 787 completion tokens, context growth is the binding constraint, exactly what mechanism D targets. `fastapi_14479`: a 1,945-byte patch is produced but fails required tests; the trace also captures a `write_file` tool call truncated by the 512-token output cap, disclosing a real side effect of that frozen runtime value.

**Task-validity floor.** A base-commit probe (official harness, `--skip-agent-patch`) shows three tasks — `fastapi_14077`, `fastapi_14851`, `requests_7427` — pass required tests with *no* agent patch. Hence every resolution rate in this paper is an upper bound, and one B1-A success (`fastapi_14851`) is a confirmed false positive. Pre-registered arm counts are not retroactively adjusted.

## 4. Related Work

Repository-level issue fixing has been standardized by benchmarks such as SWE-bench (Jimenez et al., 2024), which our task format follows. Agent designs on top of it divide into single-agent loops with tool-use scaffolds (SWE-agent, Yang et al., 2024; ReAct-style reasoning-acting loops, Yao et al., 2023) and pipeline or multi-agent decompositions, from verbal-reflection loops (Reflexion, Shinn et al., 2023) to open multi-agent platforms (OpenHands, Wang et al., 2024) and training-free pipeline agents (Agentless, Xia et al., 2024). Most report pass-rate improvements on fixed harnesses; our study instead holds the harness fixed and ablates the *budget policy* dimension that such systems usually change implicitly. Context-window management and retrieval compression are well-known efficiency techniques, and tool-use budgeting appears in agent sandboxes and cost-control systems; less studied is the *allocation of a shared budget across sub-agents at inference time*. Our finding — that a natural-language budget share is exceeded by 75% and that a natural-language submission prohibition is violated on 13 of 65 tasks — connects to evidence on instruction-following unreliability of quantized open models (e.g., the Gemma family, Gemma Team, 2025) and argues for mechanism-level enforcement over prompt-level policy in multi-agent tool budgets. We also position the paper relative to reproducibility practice in agent evaluation: we contributed a hash-pinned freeze and a one-shot held-out protocol rather than best-of-n reporting.

## 5. Limitations and Conclusion

**Limitations.** (1) All results are self-executed harness runs; no competition-side verifier or leaderboard score is claimed. (2) The task set has a validity floor: three tasks pass at base commit, so rates are upper bounds and cross-arm success attribution is bounded. (3) One quantized 31B model (w4a16) on three repositories; requests contributes only 7 development / 6 held-out tasks, so per-repository statements there are directional only. (4) The decision line (≥10 pp, p<0.05 on n=65) is powered only for large, nearly flip-free effects; our estimates carry wide Wilson intervals and we do not claim equivalence. (5) `max_output_tokens=512` was frozen for arm comparability and we observed tool-call truncation as a side effect. (6) Two designed mechanisms were not evaluated: dynamic budget adaptation (B1-ABC) and a planner-lite variant (B1A3) — the latter was deliberately not run because the empty-submitting tasks did not lack calls, so more budget would likely buy more reading, not edits.

**Conclusion.** Under the frozen session envelopes, moving from 10 calls/1 minute/50 turns to 20 calls/3 minutes/60 turns moved deaths from truncation to context overflow without significant gains, and two-phase orchestration — as actually executed — did not reach its pre-registered improvement line on either the development or the frozen held-out split. What the traces do establish is mechanistic and transferable: prompt-level budget shares and behavioral prohibitions were violated exactly where they mattered, the harness offers no per-agent budget or submission hook, and the one structural repair that worked (handoff timing, context economy) still left abandonment intact. For practitioners, the lesson is to enforce budget policy in the framework rather than the prompt; for researchers, the honest reading of this study is a boundary result about *when* orchestration can pay, not evidence that division of labor is useless.

## References

- Jimenez, C. et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024. arXiv:2310.06770.
- Yang, J. et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS 2024. arXiv:2405.15793.
- Yao, S. et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023. arXiv:2210.03629.
- Shinn, N. et al. Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023. arXiv:2303.11366.
- Wang, X. et al. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. 2024. arXiv:2407.16741.
- Xia, C. S. et al. Agentless: Demystifying LLM-based Software Engineering Agents. 2024. arXiv:2407.01489.
- Gemma Team. Gemma 3 Technical Report. 2025. arXiv:2503.19786.
