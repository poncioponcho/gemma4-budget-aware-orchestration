"""Machine-readable task-record construction shared by notebook and backfill."""
from __future__ import annotations

from typing import Any


def build_task_record(
    result: dict[str, Any],
    *,
    run_id: str,
    group: str,
    task_position: int | None = None,
    batch_index: Any = None,
    artifact_fields: dict[str, Any] | None = None,
    failure_category: str | None = None,
) -> dict[str, Any]:
    """Build the stable schema; unknown values remain None by design."""
    if failure_category is None:
        # Keep the module usable inside the offline analyzer and defer the
        # import so the same function can be embedded in the Kaggle cell.
        from stats import classify

        failure_category = classify(result)
    row = {
        "run_id": run_id,
        "instance_id": result.get("instance_id"),
        "group": group,
        "task_position": task_position,
        "batch_index": batch_index,
        "patch_validity": None,
        "patch_apply_cmd": None,
        "test_patch_applied": None,
        "pytest_exit_code": result.get("test_exit_code"),
        "local_test_pass": None,
        "junit_valid": None,
        "official_pass": None,
        "fail_to_pass_baseline": None,
        "pass_to_pass_reference": None,
        "tool_calls_used": result.get("tool_calls"),
        "total_llm_calls": result.get("total_llm_calls"),
        "agent_elapsed_seconds": result.get("duration_seconds"),
        "time_seconds_remaining": None,
        "max_tokens_context": None,
        "failure_category": failure_category,
        "submit_patch_called": None,
        "run_status": "completed",
        "resumed_skip": False,
        "error_present": bool(result.get("error")),
        "prompt_tokens_total": None,
        "completion_tokens_total": None,
        "log_paths": artifact_fields or {},
    }
    return row
