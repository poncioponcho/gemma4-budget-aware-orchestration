#!/usr/bin/env python3
"""Build the public per-task results table for the Gemma 4 ablation release.

Reads the archived per-task records (task_results.jsonl) and trace final_metrics
from the six executed arm-runs and emits data/per_task_results.csv with one row
per (arm, split, instance_id).

Conventions mirrored exactly from the frozen analysis scripts:
- same task across multiple run roots: the last root read wins (analyze_ablation.py)
- token metrics: first root with a non-empty trace final_metrics wins
- 512-output-cap truncation flag: any root trace whose steps contain the marker

Usage:  python3 make_per_task_csv.py --archive /path/to/audit_artifacts --out data/per_task_results.csv
"""
import argparse
import csv
import glob
import json
import os

TRUNC_MARKER = "token limit before the tool call"

# (arm, split, [run roots relative to archive], layout)
# layout "task_runs" = task_runs/{task}/attempt_*/eval_results/task_results.jsonl   (development)
# layout "task_results" = task_results/{task}/task_results.jsonl                     (held-out)
RUNS = [
    ("B0",   "development", ["w2_b0_20261013/batch1/output/w2_results",
                             "w2_b0_20261013/batch23/output/w2_results",
                             "w2_b0_20261013/batch4/output/w2_results_b0_20-34_42-48",
                             "w2_b0_20261013/batch5/output/w2_results_b0_48-58_60-65"], "task_runs"),
    ("B0p",  "development", ["w2_b0_20261013/b0p_runA/output/w2_results_b0p_0-10",
                             "w2_b0_20261013/b0p_runB/output/w2_results_b0p_10-20_34-42_58-60",
                             "w2_b0_20261013/b0p_runC/output/w2_results_b0p_20-34_42-48_48-58_60-65"], "task_runs"),
    ("B1A",  "development", ["w2_b0_20261013/b1a_runA/output/w2_results_b1a_0-10",
                             "w2_b0_20261013/b1a_runB/output/w2_results_b1a_10-20_34-42_58-60",
                             "w2_b0_20261013/b1a_runC/output/w2_results_b1a_20-34_42-48_48-58_60-65"], "task_runs"),
    ("B1A2", "development", ["w2_b0_20261013/b1a2_runA/output/w2_results_b1a2_0-10",
                             "w2_b0_20261013/b1a2_runB/output/w2_results_b1a2_10-20_34-42_58-60",
                             "w2_b0_20261013/b1a2_runC/output/w2_results_b1a2_20-34_42-48_48-58_60-65"], "task_runs"),
    ("B0p",  "heldout",     ["heldout_20261017/B0p/output/w2_results_b0p_heldout_0-63"], "task_results"),
    ("B1A2", "heldout",     ["heldout_20261017/B1A2/output/w2_results_b1a2_heldout_0-63"], "task_results"),
]


def task_glob(root, layout, instance_id):
    if layout == "task_runs":
        return glob.glob(os.path.join(root, "task_runs", instance_id, "attempt_*",
                                      "eval_results", "task_results.jsonl"))
    return [os.path.join(root, "task_results", instance_id, "task_results.jsonl")]


def collect(archive, roots, layout):
    rows = {}
    for rel in roots:
        root = os.path.join(archive, rel)
        if not os.path.isdir(root):
            raise SystemExit(f"missing run root: {root}")
        for p in sorted(glob.glob(os.path.join(root, "task_results", "*"))):
            if not os.path.isdir(p):
                continue
            instance_id = os.path.basename(p)
            for f in sorted(task_glob(root, layout, instance_id)):
                for line in open(f, encoding="utf-8"):
                    line = line.strip()
                    if line.startswith("{"):
                        rows[instance_id] = json.loads(line)  # same task: last root wins
    return rows


def trace_glob(root, instance_id):
    return glob.glob(os.path.join(root, "traces", instance_id, "attempt_*", f"trace_{instance_id}.json"))


def tokens_and_trunc(archive, roots, instance_id):
    for rel in roots:  # first root with metrics wins
        root = os.path.join(archive, rel)
        for p in sorted(trace_glob(root, instance_id)):
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            fm = d.get("final_metrics") or {}
            trunc = any(TRUNC_MARKER in (s.get("message") or "")
                        for s in d.get("steps", []) if isinstance(s, dict))
            if fm:
                return fm.get("total_prompt_tokens"), fm.get("total_completion_tokens"), trunc
    return None, None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="path to audit_artifacts/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_rows = []
    for arm, split, roots, layout in RUNS:
        rows = collect(args.archive, roots, layout)
        for instance_id, r in rows.items():
            ptok, ctok, trunc = tokens_and_trunc(args.archive, roots, instance_id)
            out_rows.append({
                "arm": arm, "split": split, "instance_id": instance_id,
                "repo": r.get("repo", ""), "resolved": bool(r.get("resolved")),
                "agent_patch_size": r.get("agent_patch_size") or 0,
                "error": r.get("error") or "", "tool_calls": r.get("tool_calls"),
                "total_llm_calls": r.get("total_llm_calls"),
                "duration_seconds": r.get("duration_seconds"),
                "test_exit_code": r.get("test_exit_code"),
                "prompt_tokens": ptok, "completion_tokens": ctok,
                "truncated_tool_call": bool(trunc),
            })
        print(f"{arm:5s} {split:11s} rows={len(rows)} roots={len(roots)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
