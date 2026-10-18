"""Shared analysis definitions for the W2 ablation records.

All consumers use these functions so historical n=30 and final n=65 reports
cannot silently diverge in their labels or statistical conventions.
"""
from __future__ import annotations

import math
import re
from typing import Any

KNOWN_ERROR_MARKERS = (
    ("ContextWindowExceeded", "context_overflow"),
    ("tool call budget", "timeout"),
    ("turns budget", "timeout"),
    ("session timeout", "timeout"),
)

# Harness *notes* that are not failures and must not abort classification.
# "Agent completed execution without calling submit_patch." is informational:
# per HARNESS_README Sec.2.3 the harness still extracts the working-tree diff, so
# the verdict depends on whether a patch exists, not on this message.
# First observed in B1-A' (writable-root) run A; absent from all 195 archived records
# (verified by grep before this change), so adding it cannot alter historical labels.
NON_FATAL_ERROR_MARKERS = ("without calling submit_patch",)


def classify(result: dict[str, Any]) -> str:
    if result.get("resolved") is True:
        return "none"
    error = result.get("error") or ""
    if "ContextWindowExceeded" in error:
        # This is an actual sandbox/service failure caused by the fixed 16k
        # context boundary, and is the only infra category in this dataset.
        return "infra"
    if "tool call budget" in error or "turns budget" in error or "session timeout" in error:
        return "timeout"
    if error and not any(m in error for m in NON_FATAL_ERROR_MARKERS):
        raise ValueError("unmatched non-empty error: " + error)
    if (result.get("agent_patch_size") or 0) > 0:
        return "test_fail"
    return "no_patch"


def unmatched_errors(results: list[dict[str, Any]]) -> list[str]:
    unmatched = []
    for result in results:
        error = result.get("error") or ""
        if not error:
            continue
        try:
            classify(result)
        except ValueError:
            unmatched.append(error)
    return unmatched


def wilson(k: int, n: int) -> list[float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def mcnemar_counts(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], tasks: list[str]) -> tuple[int, int]:
    win = lose = 0
    for task in tasks:
        ra = a.get(task, {}).get("resolved") is True
        rb = b.get(task, {}).get("resolved") is True
        if ra and not rb:
            win += 1
        elif rb and not ra:
            lose += 1
    return win, lose


def mcnemar_p_values(win: int, lose: int) -> tuple[float, float]:
    """Return (doubling, standard exact two-sided) conditional-binomial p's."""
    n = win + lose
    if n == 0:
        return 1.0, 1.0
    probs = [math.comb(n, i) * (0.5 ** n) for i in range(n + 1)]
    k = min(win, lose)
    p_doubling = min(1.0, 2 * sum(probs[: k + 1]))
    observed = probs[k]
    p_standard = min(1.0, sum(p for p in probs if p <= observed + 1e-15))
    return p_doubling, p_standard


def mcnemar_exact(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], tasks: list[str]) -> tuple[int, int, float, float]:
    win, lose = mcnemar_counts(a, b, tasks)
    p_doubling, p_standard = mcnemar_p_values(win, lose)
    return win, lose, p_doubling, p_standard


def pick_attempt(paths: list[str]) -> str | None:
    """Pick the numerically greatest attempt path, never lexicographic last."""
    if not paths:
        return None
    def key(path: str) -> tuple[int, str]:
        match = re.search(r"attempt_(\d+)", path)
        return (int(match.group(1)) if match else -1, path)
    return sorted(paths, key=key)[-1]
