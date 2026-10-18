#!/usr/bin/env python3
"""方法冻结清单：生成 / 校验（METHOD_FREEZE.md 的机器可执行部分）。

用法：
  python3 ablation/freeze.py --write    # 生成/刷新 freeze_manifest.json（仅在冻结时执行一次）
  python3 ablation/freeze.py            # 校验当前工作树与冻结清单是否一致（默认，只报告）

冻结后任何一项不匹配都应视为**破坏可复现性**，必须报告而不是"修"清单。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # audit_artifacts/w2_b0_20261013
ART = ROOT.parent                                    # audit_artifacts
MANIFEST = ROOT / "freeze_manifest.json"

BUNDLES = {"B0p": ART / "b0prime_bundle", "B1A": ART / "b1_bundle", "B1A2": ART / "b1a2_bundle"}
B0_SAMPLE_DIRS = [
    ROOT / "batch1/output/w2_results/b0_submission",
    ROOT / "batch23/output/w2_results/b0_submission",
    ROOT / "batch4/output/w2_results_b0_20-34_42-48/submission",
    ROOT / "batch5/output/w2_results_b0_48-58_60-65/submission",
]
FILES = {
    "tasks_jsonl": ART / "kaggle/tasks.jsonl",
    "split_development": ROOT / "split_development.json",
    "split_heldout": ROOT / "split_heldout.json",
    "split_manifest": ROOT / "split_manifest.json",
}
SCRIPTS = {
    "stats.py": HERE / "stats.py",
    "analyze_ablation.py": HERE / "analyze_ablation.py",
    "handoff_forensics.py": HERE / "handoff_forensics.py",
    "build_ablation_notebook.py": HERE / "build_ablation_notebook.py",
    "validate_bundle.py": HERE / "validate_bundle.py",
    "task_record.py": HERE / "task_record.py",
}
RUNTIME = {
    "max_output_tokens": 512,
    "thinking_budget": 0,
    "include_thoughts": False,
    "budgets": {
        "B0": {"max_tool_calls": 10, "max_time_minutes": 1, "max_turns": 50},
        "B0p": {"max_tool_calls": 20, "max_time_minutes": 3, "max_turns": 60},
        "B1A": {"max_tool_calls": 20, "max_time_minutes": 3, "max_turns": 60},
        "B1A2": {"max_tool_calls": 20, "max_time_minutes": 3, "max_turns": 60},
    },
    "runtime_rewrites": [
        "strip `adapter:` lines",
        "remove adapters/ directory",
        "sampling: thinking_budget=0, include_thoughts=false, max_output_tokens=512",
        "models.yaml aliases: gemma-4-31b-it-qat-w4a16-ct / main_lora / tool_lora -> base model",
    ],
}
JUDGMENT = {
    "primary_contrast": "B1A2 vs B0p (paired, same 20-call budget)",
    "min_effect_pp": 10,
    "alpha": 0.05,
    "test": "McNemar exact conditional binomial; both doubling and standard two-sided must agree",
    "note": "判定线自 2026-10-15 预注册后未改动",
}
DEV_RESULTS = {
    "B0": {"n": 65, "resolved": 2, "rate": "3.1%", "wilson": ["0.8%", "10.5%"]},
    "B0p": {"n": 65, "resolved": 4, "rate": "6.2%", "wilson": ["2.4%", "14.8%"]},
    "B1A": {"n": 65, "resolved": 7, "rate": "10.8%", "wilson": ["5.3%", "20.6%"]},
    "B1A2": {"n": 65, "resolved": 6, "rate": "9.2%", "wilson": ["4.3%", "18.7%"]},
    "contrasts": {
        "B0p_vs_B0": {"delta_pp": 3.1, "p": 0.5000, "verdict": "不显著"},
        "B1A_vs_B0p": {"delta_pp": 4.6, "p": 0.4531, "verdict": "预注册判定不成立"},
        "B1A2_vs_B0p": {"delta_pp": 3.1, "p": 0.6250, "verdict": "预注册判定不成立"},
    },
}
NOT_CLAIMED = [
    "B1-ABC（机制 B）——未运行，不得推断",
    "B1A3（planner-lite）——已建成但决定不跑，不得推断",
    "任何 per-agent 预算或提交钩子——框架不支持（adk_submission closed registries）",
    "官方 verify_task / leaderboard 判定——全部为 self-run official harness",
    "held-out 63 的任何结果——冻结前未运行",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dir_hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def build() -> dict:
    return {
        "schema": "w2_method_freeze_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {k: {"path": str(v.relative_to(ART.parent)), "sha256": sha(v)} for k, v in FILES.items()},
        "bundles": {arm: dir_hashes(d) for arm, d in BUNDLES.items()},
        "b0_sample_submission": dir_hashes(B0_SAMPLE_DIRS[0]),
        "runtime": RUNTIME,
        "scripts": {k: sha(v) for k, v in SCRIPTS.items()},
        "judgment": JUDGMENT,
        "development_results": DEV_RESULTS,
        "not_claimed": NOT_CLAIMED,
    }


def verify(man: dict) -> list[str]:
    problems: list[str] = []
    for key, item in man["files"].items():
        p = ART.parent / item["path"]
        if not p.is_file():
            problems.append(f"{key}: missing {item['path']}")
        elif sha(p) != item["sha256"]:
            problems.append(f"{key}: SHA-256 mismatch ({item['path']})")
    for arm, expected in man["bundles"].items():
        cur = dir_hashes(BUNDLES[arm])
        if cur != expected:
            diff = set(cur) ^ set(expected) or {k for k in cur if cur.get(k) != expected.get(k)}
            problems.append(f"bundle {arm}: drift in {sorted(diff)}")
    cur_b0 = dir_hashes(B0_SAMPLE_DIRS[0])
    if cur_b0 != man["b0_sample_submission"]:
        problems.append("b0 sample_submission: drift")
    for name, digest in man["scripts"].items():
        if sha(SCRIPTS[name]) != digest:
            problems.append(f"script {name}: SHA-256 mismatch (analysis code changed after freeze)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if args.write:
        man = build()
        MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"frozen: {MANIFEST}")
        print(f"  files={len(man['files'])} bundles={len(man['bundles'])} scripts={len(man['scripts'])}")
        return 0

    if not MANIFEST.is_file():
        print(f"no freeze manifest at {MANIFEST}; run --write at freeze time")
        return 1
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = verify(man)
    print(f"freeze: {man['schema']}  frozen_at={man['frozen_at_utc']}")
    if problems:
        print(f"\n!! {len(problems)} FROZEN ITEM(S) DRIFTED:")
        for p in problems:
            print(f"   - {p}")
        print("\n冻结后这些项不应改变。若确属有意修改，需重新冻结并说明理由。")
        return 1
    print(f"OK: {len(man['files'])} files, {len(man['bundles'])} bundles, "
          f"{len(man['scripts'])} scripts all match the freeze.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
