#!/usr/bin/env python3
"""两阶段移交取证（离线，只读，不修改任何产物）。

来源：B1A_HANDOFF_FORENSICS.md 的一次性分析固化为可复跑脚本，
用作 HANDOFF_FIX_DESIGN.md §3.1 预注册指标的**同一把尺子**（B1-A 与 B1-A' 必须同口径）。

输出三行预注册主诊断：
  1. 空补丁提交（调用了 submit_patch 但 agent_patch_size == 0）
  2. abandoned_no_patch 占比
  3. 咨询子代理时的计费调用数（中位数）
外加：工具调用分布、移交时机 × 补丁产出率的剂量-反应表。

用法：
  python3 ablation/handoff_forensics.py --label B1A --root ROOT1 [--root ROOT2 ...]
  ROOT = 某次运行的 w2_results 目录（其下 traces/<task>/attempt_*/trace_<task>.json）

计费口径：官方语义下 `submit_patch` 与 `get_status` 不计入 tool-call budget
（HARNESS_README §6.1），故"计费调用"= 全部 tool_calls 去掉这两个。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

# 官方工具全集（RESEARCH_HYPOTHESIS.md §2）。任何不在此集合中的 tool_call 名
# 即子代理调用（agent_tool）。
OFFICIAL_TOOLS = {
    "run_command", "submit_patch", "get_status", "read_file", "edit_file",
    "write_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph",
}
FREE_TOOLS = {"submit_patch", "get_status"}   # 不计入预算
WRITE_TOOLS = {"edit_file", "write_file", "submit_patch"}


def collect_results(roots: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "task_runs", "*", "attempt_*",
                                               "eval_results", "task_results.jsonl"))):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line.startswith("{"):
                    d = json.loads(line)
                    rows[d["instance_id"]] = d
    return rows


def collect_traces(roots: list[str]) -> dict[str, list[str]]:
    """task_id → 按 attempt 顺序展开的 tool_call 名序列（取 attempt 编号最大者）。"""
    per_task: dict[str, tuple[int, list[str]]] = {}
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "traces", "*", "attempt_*", "trace_*.json"))):
            attempt = int(os.path.basename(os.path.dirname(p)).split("_", 1)[1])
            task_id = os.path.basename(p)[len("trace_"):-len(".json")]
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            names = [tc.get("function_name")
                     for step in d.get("steps", [])
                     for tc in (step.get("tool_calls") or [])]
            prev = per_task.get(task_id)
            if prev is None or attempt >= prev[0]:
                per_task[task_id] = (attempt, names)
    return {k: v[1] for k, v in per_task.items()}


def classify(result: dict) -> str:
    err = result.get("error") or ""
    if result.get("resolved"):
        return "resolved"
    if "ContextWindowExceeded" in err:
        return "context_overflow"
    if "tool call budget" in err or "turns budget" in err:
        return "budget_truncated"
    if "session timeout" in err:
        return "session_timeout"
    if result.get("agent_patch_size", 0) > 0:
        return "patch_failed_tests"
    return "abandoned_no_patch"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--expect-n", type=int, default=None)
    args = ap.parse_args()

    results = collect_results(args.root)
    traces = collect_traces(args.root)
    n = len(results)

    tool_hist: Counter = Counter()
    empty_patch: list[str] = []
    handoffs: list[int] = []
    buckets = defaultdict(lambda: [0, 0, 0])   # key → [n, with_patch, resolved]

    for task_id, result in sorted(results.items()):
        names = traces.get(task_id, [])
        tool_hist.update(names)
        billed = [x for x in names if x not in FREE_TOOLS]
        sub_idx = [i for i, x in enumerate(names) if x and x not in OFFICIAL_TOOLS]
        handoff = (sum(1 for x in names[:sub_idx[0]] if x not in FREE_TOOLS)
                   if sub_idx else None)
        if handoff is not None:
            handoffs.append(handoff)
        if "submit_patch" in names and not result.get("agent_patch_size", 0):
            empty_patch.append(task_id)

        patch = result.get("agent_patch_size", 0) > 0
        key = ("no_subagent" if handoff is None else
               "0-7" if handoff <= 7 else
               "8-13" if handoff <= 13 else
               "14-17" if handoff <= 17 else "18-20")
        buckets[key][0] += 1
        buckets[key][1] += int(patch)
        buckets[key][2] += int(bool(result.get("resolved")))
        del billed  # 仅用于可读性，口径见文件头

    cat = Counter(classify(r) for r in results.values())
    resolved = sum(1 for r in results.values() if r.get("resolved"))
    abandoned = cat.get("abandoned_no_patch", 0)

    print(f"===== {args.label}  n={n} =====")
    if args.expect_n is not None and n != args.expect_n:
        print(f"  !! n={n} != expected {args.expect_n}")
        return 1
    print(f"resolved: {resolved}/{n} = {resolved/n:.1%}" if n else "resolved: 0/0")
    print(f"失败分类: {dict(cat.most_common())}")
    print()
    print("--- 预注册主诊断（HANDOFF_FIX_DESIGN.md §3.1）---")
    print(f"  1. 空补丁提交（submit_patch 被调用但 patch=0）: {len(empty_patch)}/{n}"
          f"  [{len(empty_patch)/n:.0%}]" if n else "")
    print(f"  2. abandoned_no_patch 占比: {abandoned}/{n} = {abandoned/n:.0%}" if n else "")
    print(f"  3. 咨询子代理时的计费调用（中位）: "
          f"{statistics.median(handoffs) if handoffs else 'n/a'}"
          f"  (n={len(handoffs)}, 最大 {max(handoffs) if handoffs else '-'})")
    print()
    print("--- 工具调用分布 ---")
    for k, v in tool_hist.most_common():
        tag = " (子代理)" if k not in OFFICIAL_TOOLS else ""
        print(f"  {v:>5}  {k}{tag}")
    print()
    print("--- 剂量-反应：移交时机 × 补丁产出 ---")
    for key in ("0-7", "8-13", "14-17", "18-20", "no_subagent"):
        if key in buckets:
            b_n, b_p, b_r = buckets[key]
            print(f"  {key:<12} n={b_n:<3} 有补丁={b_p:<3} ({b_p/b_n:.0%})"
                  f"  resolved={b_r} ({b_r/b_n:.0%})")
    if empty_patch:
        print()
        print(f"空补丁任务（前 10）: {empty_patch[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
