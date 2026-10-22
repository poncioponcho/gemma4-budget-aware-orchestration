# Prompt-Level Budget Constraints Do Not Bind — Research Artifact

Research artifact for the paper *"Prompt-Level Budget Constraints Do Not Bind: A
Budget-Controlled Ablation of a Gemma 4 Issue-Fixing Agent"* (submission to the
[Google – The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper), 2026).

**One-paragraph summary.** On the official Gemma 4 Developer Agent harness we ran a
budget-controlled four-arm ablation (fixed-order baseline at 10 calls; the same agent at 20
calls; a planner→coder two-phase arm; and a writable-root topology repair) over 128
multi-repository issue-fixing tasks, with a pre-registered decision line (≥10 pp and
McNemar p<0.05) and a frozen 63-task held-out run executed once, blind. Doubling the tool
budget removes truncation but converts it into context-overflow deaths (5%→35%) with no
significant gain (+3.1 pp, p=0.50). Neither orchestration arm reaches the line on
development (10.8%, p=0.4531; 9.2%, p=0.6250) or held-out (12.7% vs 15.9%, +3.2 pp,
p=0.7266). Mechanistic trace forensics explain the negative result: the planner consumed a
median of 14 of 20 billed calls against a prompt-stated share of 8; 29/65 tasks submitted
empty patches; and 13 tasks submitted without any edit despite spending their full budget.
Topology repair cut empty submissions 45%→32% and handoff delay 14→0 calls, but abandonment
grew. The transferable finding: under a session-level shared budget with no per-agent
accounting, prompt-level quantitative constraints and behavioral prohibitions are
non-binding on this model.

## Live links

- Repository: <https://github.com/poncioponcho/gemma4-budget-aware-orchestration>
- Executable reproduction notebook on Kaggle (self-contained, no local files needed):
  <https://www.kaggle.com/code/daniel1547/gemma-4-budget-aware-orchestration-reproduction>

## Contents

| Path | What it is |
|---|---|
| `paper/writeup.md` | The submission writeup — byte-identical to the version submitted on Kaggle (English, 2,963 words incl. tables, ≤3,000 limit; SHA-256 `f07ab276…b5733a`). The body starts at `## Abstract`; title and subtitle live in the competition metadata. |
| `data/per_task_results.csv` | 386 rows (6 arm-runs × tasks): per-task resolved/failure mode, patch size, error, tool calls, prompt/completion tokens, 512-cap truncation flag |
| `data/heldout_stats_B0p_vs_B1A2.json` | Frozen held-out analysis (n=63 per arm) with validation block and run metadata |
| `data/freeze_manifest.json` | SHA-256 freeze manifest of tasks, splits, per-arm bundles, budgets, sampling, scripts, decision line |
| `data/split_*.json` | development (65) / held-out (63) task lists + split manifest |
| `code/` | Frozen analysis scripts (`stats.py`, `analyze_ablation.py`, `handoff_forensics.py`, `freeze.py`, `validate_bundle.py`, `task_record.py`), the CSV builder (`make_per_task_csv.py`), and `reproduce_results.ipynb` |
| `logs/` | Archived analysis logs: development three-arm (n=65), development B0p-vs-B1A2 (n=65), held-out (n=63), and the base-commit validity probe |

## Reproducing every number in the paper

```bash
cd code
jupyter nbconvert --to notebook --execute reproduce_results.ipynb
```

The notebook reads `data/per_task_results.csv`, recomputes resolution rates, Wilson 95%
intervals, failure-mode distributions, per-repository breakdowns, prompt-token medians and
same-task paired exact McNemar tests, and **asserts every value against the numbers printed
in the paper** (development 2/65, 4/65, 7/65, 6/65; held-out 8/63, 10/63; p=0.4531,
0.6250, 0.7266; token medians 82,143 / 128,526 / 116,911 / 99,443 / 120,473 / 102,138).
Any mismatch raises `AssertionError` — the notebook is the verification contract.

The statistical conventions are exactly those of the frozen `stats.py` /
`analyze_ablation.py`:

- Wilson score interval, z=1.96, for proportions.
- Same-task paired McNemar exact (conditional binomial, two-sided, doubled tail);
  the pre-registered line additionally required the standard and doubled variants to agree.
- Failure-mode classification precedence: resolved → context overflow (`ContextWindowExceeded`)
  → budget truncation (`tool call budget` / `turns budget`) → session timeout →
  patch failed tests (non-empty patch, unresolved) → abandoned/no patch.
- Token metrics are trace `final_metrics` totals; medians over tasks with metrics.

## Scope and honesty notes

- All numbers are **self-executed runs of the official harness** (Kaggle, 4× NVIDIA L4).
  No competition-side verifier was invoked; `official_pass` is null for every record.
- Three tasks (`fastapi_14077`, `fastapi_14851`, `requests_7427`) pass required tests at
  the base commit with no agent patch, so **every resolution rate is an upper bound**
  (see `logs/base_commit_probe_20261017.json`).
- Raw task snapshots, repository contents and agent traces are not redistributed here;
  this release contains per-task records, summaries, analysis code and the writeup.
  Task IDs come from the public competition data.

## License

Original code and documentation in this repository are licensed under the
[Apache License 2.0](LICENSE). This satisfies the competition's open-source obligation for
award-winning submissions. Third-party assets retain their own licenses: the official
harness wheels and task data are provided by the competition organizers; see the
competition rules.

## Citation

```
Google - The Gemma 4 Developer Agent Paper Track. Kaggle, 2026.
https://www.kaggle.com/competitions/gemma-4-developer-agent-paper
```
