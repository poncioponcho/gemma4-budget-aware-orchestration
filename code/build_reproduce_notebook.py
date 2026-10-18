#!/usr/bin/env python3
"""Build code/reproduce_results.ipynb (the release verification contract)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3"}}

nb.cells = [
    nbf.v4.new_markdown_cell(
"""# Reproducing every number in the paper

This notebook recomputes all quantitative claims in `paper/writeup.md` from
`data/per_task_results.csv` (386 per-task records over six arm-runs), using exactly the
statistical conventions of the frozen analysis scripts (`stats.py`, `analyze_ablation.py`).

If any recomputed value deviates from the paper, an `AssertionError` is raised — so a clean
run of this notebook is a verification contract, not just a demo."""),

    nbf.v4.new_markdown_cell("## 1. Load the per-task table"),
    nbf.v4.new_code_cell(
"""import csv, math, statistics
from collections import Counter
from pathlib import Path

CSV = Path("..") / "data" / "per_task_results.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
by = {}
for r in rows:
    by.setdefault((r["arm"], r["split"]), []).append(r)
print(f"loaded {len(rows)} per-task records across {len(by)} arm-runs")
for k in by:
    print(" ", k, len(by[k]))"""),

    nbf.v4.new_markdown_cell(
"""## 2. Conventions (identical to the frozen scripts)

* Wilson score interval, z = 1.96.
* Paired exact McNemar, two-sided conditional binomial with doubled tail.
* Failure-mode classification precedence, from `analyze_ablation.py`."""),
    nbf.v4.new_code_cell(
"""def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    hw = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0.0, c-hw), min(1.0, c+hw))

def classify(r):
    if r["resolved"] == "True":
        return "resolved"
    e = r["error"]
    if "ContextWindowExceeded" in e: return "context_overflow"
    if "tool call budget" in e or "turns budget" in e: return "budget_truncated"
    if "session timeout" in e: return "session_timeout"
    if int(r["agent_patch_size"]) > 0: return "patch_failed_tests"
    return "abandoned_no_patch"

def mcnemar_exact(ra, rb):
    win = sum(1 for a, b in zip(ra, rb) if a and not b)
    lose = sum(1 for a, b in zip(ra, rb) if b and not a)
    n = win + lose
    if n == 0: return win, lose, 1.0
    k = min(win, lose)
    return win, lose, min(1.0, sum(math.comb(n, i) for i in range(k+1)) * 0.5**n * 2)

def paired(a_key, b_key):
    da = {r["instance_id"]: r["resolved"] == "True" for r in by[a_key]}
    db = {r["instance_id"]: r["resolved"] == "True" for r in by[b_key]}
    common = sorted(set(da) & set(db))
    return mcnemar_exact([da[t] for t in common], [db[t] for t in common]), len(common)

def med(key, field):
    v = sorted(int(r[field]) for r in by[key] if r[field])
    return statistics.median(v) if v else None"""),

    nbf.v4.new_markdown_cell("## 3. Development results (n=65 per arm) — paper Table 2"),
    nbf.v4.new_code_cell(
"""order = [("B0","development"), ("B0p","development"), ("B1A","development"), ("B1A2","development")]
dev = {}
for key in order:
    rr = by[key]
    ok = sum(1 for r in rr if r["resolved"] == "True")
    modes = Counter(classify(r) for r in rr)
    dev[key] = ok
    lo, hi = wilson(ok, len(rr))
    print(f"{key[0]:5s} resolved {ok}/{len(rr)} = {ok/len(rr):.1%}  Wilson 95% [{lo:.1%}, {hi:.1%}]  "
          f"overflow {modes['context_overflow']}  abandoned {modes['abandoned_no_patch']}  "
          f"prompt-median {med(key, 'prompt_tokens')}")

assert (dev[("B0","development")], dev[("B0p","development")],
        dev[("B1A","development")], dev[("B1A2","development")]) == (2, 4, 7, 6)
assert med(("B0","development"), "prompt_tokens") == 82143
assert med(("B0p","development"), "prompt_tokens") == 128526
assert med(("B1A","development"), "prompt_tokens") == 116911
assert med(("B1A2","development"), "prompt_tokens") == 99443
print("development counts and token medians: OK (2/65, 4/65, 7/65, 6/65; 82143/128526/116911/99443)")"""),

    nbf.v4.new_markdown_cell("## 4. Paired McNemar against the same-budget control (B0′)"),
    nbf.v4.new_code_cell(
"""(w1, l1, p1), n1 = paired(("B0p","development"), ("B1A","development"))
(w2, l2, p2), n2 = paired(("B0p","development"), ("B1A2","development"))
(w3, l3, p3), n3 = paired(("B0p","development"), ("B0","development"))
print(f"development B1-A  vs B0p : {w1}/{l1} wins, p={p1:.4f} (paper: 2/5, 0.4531)")
print(f"development B1-A' vs B0p : {w2}/{l2} wins, p={p2:.4f} (paper: 1/3, 0.6250)")
print(f"development B0p   vs B0  : {w3}/{l3} wins, p={p3:.4f} (paper: 2/0, 0.5000)")
assert (w1, l1, round(p1, 4), n1) == (2, 5, 0.4531, 65)
assert (w2, l2, round(p2, 4), n2) == (1, 3, 0.625, 65)
assert (w3, l3, round(p3, 2), n3) == (2, 0, 0.5, 65)
print("all three decision-line comparisons reproduced: none reaches >=10pp and p<0.05")"""),

    nbf.v4.new_markdown_cell("## 5. Frozen held-out results (n=63 per arm) — paper Table 5"),
    nbf.v4.new_code_cell(
"""hk = [("B0p","heldout"), ("B1A2","heldout")]
hok = {}
for key in hk:
    rr = by[key]
    ok = sum(1 for r in rr if r["resolved"] == "True")
    hok[key] = ok
    lo, hi = wilson(ok, len(rr))
    print(f"{key[0]:5s} resolved {ok}/{len(rr)} = {ok/len(rr):.1%}  Wilson 95% [{lo:.1%}, {hi:.1%}]  "
          f"overflow {sum(1 for r in rr if classify(r)=='context_overflow')}  "
          f"abandoned {sum(1 for r in rr if classify(r)=='abandoned_no_patch')}  "
          f"prompt-median {med(key, 'prompt_tokens')}")
(w4, l4, p4), n4 = paired(("B0p","heldout"), ("B1A2","heldout"))
print(f"held-out B1-A' vs B0p: {w4}/{l4} wins, p={p4:.4f}, delta {(hok[('B1A2','heldout')]-hok[('B0p','heldout')])/63*100:+.1f} pp (paper: 3/5, 0.7266, +3.2 pp)")
assert (hok[("B0p","heldout")], hok[("B1A2","heldout")]) == (8, 10)
assert med(("B0p","heldout"), "prompt_tokens") == 120473
assert med(("B1A2","heldout"), "prompt_tokens") == 102138
assert (w4, l4, round(p4, 4), n4) == (3, 5, 0.7266, 63)
print("held-out verdict reproduced: pre-registered line (>=10pp and p<0.05) not met")"""),

    nbf.v4.new_markdown_cell("## 6. Final assertion summary"),
    nbf.v4.new_code_cell(
"""checks = {
    "development resolved (B0/B0p/B1A/B1A2)": (2, 4, 7, 6),
    "held-out resolved (B0p/B1A2)": (8, 10),
    "prompt-token medians (dev B0/B0p/B1A/B1A2)": (82143, 128526, 116911, 99443),
    "prompt-token medians (heldout B0p/B1A2)": (120473, 102138),
    "McNemar exact p (B1A/B1A2 dev, B1A2 heldout)": (0.4531, 0.625, 0.7266),
}
print("All paper numbers reproduced from data/per_task_results.csv.")
for k, v in checks.items():
    print(f"  {k}: {v}")
print("\\nVERIFICATION PASSED")"""),
]

nbf.write(nb, "reproduce_results.ipynb")
print("wrote reproduce_results.ipynb with", len(nb.cells), "cells")
