#!/usr/bin/env python3
"""离线静态校验消融 bundle（无需 GPU、无需 Kaggle、不修改任何产物）。

校验项（对应 HANDOFF_FIX_DESIGN.md §5 的验收）：
  1. 每个 .yaml 能被 YAML 解析；`!include` 目标存在且可解析（相对本文件路径）。
  2. 每个 agent 的 `tools:` 名称都在官方工具白名单内（拼写错误会被 harness 静默忽略）。
  3. 每个 `agent_tool: config_path` 指向的子代理配置存在，且子代理的工具名同样合法。
  4. 子代理嵌套深度 ≤ 1（本项目的设计约束）。
  5. 每个 agent 都有 `name` / `model` / `instruction` / `generate_content_config`。
  6. 各臂 `eval_config.yaml` 的 evaluation 块（数值）可比较，用于单变量核对。
  7. 无 `adapter:` 行残留（运行时会被剥离，残留说明 bundle 未按 §6 红线准备）。

用法：
  python3 ablation/validate_bundle.py                 # 校验全部已知 bundle
  python3 ablation/validate_bundle.py b1a2_bundle     # 只校验指定 bundle
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ART_ROOT = HERE.parents[1]                      # audit_artifacts/
BUNDLES = ["b0prime_bundle", "b1_bundle", "b1a2_bundle", "b1abc_bundle"]

# 官方工具全集（RESEARCH_HYPOTHESIS.md §2 的 9 个官方工具）
OFFICIAL_TOOLS = {
    "run_command", "submit_patch", "get_status", "read_file", "edit_file",
    "write_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph",
}
AGENT_REQUIRED_KEYS = ["name", "model", "instruction", "generate_content_config"]


class Loader(yaml.SafeLoader):
    """SafeLoader + `!include`（相对当前文件解析，与 harness 语义一致）。"""

    def __init__(self, stream, root: Path):
        super().__init__(stream)
        self._root = root


def _include(loader: Loader, node):
    """与 harness 语义一致（adk_submission/yaml_loader.py:162）：
    .yaml/.yml → 解析后的 YAML；其余（.md/.txt）→ 原文文本。"""
    rel = loader.construct_scalar(node)
    target = (loader._root / rel).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"!include target missing: {rel} -> {target}")
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in (".yaml", ".yml"):
        return yaml.load(text, Loader=_make_loader(target.parent))
    return text


def _make_loader(root: Path):
    class _L(Loader):
        pass

    _L.add_constructor("!include", _include)
    return lambda stream: _L(stream, root)


def load_yaml(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_make_loader(path.parent))


def check_agent(cfg: dict, label: str, errors: list[str], depth: int = 0):
    for key in AGENT_REQUIRED_KEYS:
        if key not in cfg:
            errors.append(f"{label}: missing required key '{key}'")
    if depth > 1:
        errors.append(f"{label}: sub-agent nesting depth {depth} exceeds 1")
    tools = cfg.get("tools") or []
    for t in tools:
        if isinstance(t, dict):
            if "agent_tool" not in t:
                errors.append(f"{label}: unsupported tool entry {t!r}")
                continue
            ref = (t["agent_tool"] or {}).get("config_path")
            if not ref:
                errors.append(f"{label}: agent_tool without config_path")
                continue
            sub = _sub_config_path(label, ref)
            if not sub.is_file():
                errors.append(f"{label}: agent_tool config_path missing: {ref}")
                continue
            check_agent(load_yaml(sub), f"{label} -> {ref}", errors, depth + 1)
        elif t not in OFFICIAL_TOOLS:
            errors.append(f"{label}: unknown tool '{t}' (harness would ignore it silently)")


def _sub_config_path(label: str, ref: str) -> Path:
    """label 形如 '<bundle>/agent.yaml'；config_path 相对 bundle 根。"""
    bundle = label.split("/", 1)[0]
    return ART_ROOT / bundle / ref


def has_adapter_key(text: str) -> bool:
    """真·`adapter:` 键（忽略注释与缩进），与 notebook cell 3 的剥离逻辑同口径。"""
    return any(
        line.strip().startswith("adapter:")
        for line in text.splitlines()
        if not line.strip().startswith("#")
    )


def validate(bundle: str) -> tuple[list[str], list[str], dict]:
    root = ART_ROOT / bundle
    errors: list[str] = []
    warnings: list[str] = []
    info: dict = {"bundle": bundle, "agents": [], "files": []}
    if not root.is_dir():
        return [f"bundle directory missing: {root}"], [], info

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        info["files"].append(str(path.relative_to(root)))
        if path.suffix.lower() not in (".yaml", ".yml"):
            continue
        label = f"{bundle}/{path.relative_to(root)}"
        text = path.read_text(encoding="utf-8")
        try:
            cfg = load_yaml(path)
        except Exception as exc:                     # noqa: BLE001
            errors.append(f"{label}: YAML/include error: {exc}")
            continue
        if not isinstance(cfg, dict):
            errors.append(f"{label}: top level is not a mapping")
            continue
        if has_adapter_key(text):
            # 预期内：b0prime 镜像官方 sample_submission 带 `adapter:` 行，运行时会由
            # notebook cell 3 统一剥离（§6 红线），并有 "adapter: key lines survived" 断言兜底。
            warnings.append(f"{label}: carries an 'adapter:' key (stripped at runtime by cell 3)")
        if "name" in cfg and "tools" in cfg:
            info["agents"].append(cfg.get("name"))
            check_agent(cfg, label, errors)
        elif "evaluation" in cfg:
            info["eval_config"] = cfg["evaluation"]
    return errors, warnings, info


def main() -> int:
    wanted = sys.argv[1:] or BUNDLES
    all_errors: list[str] = []
    summary = []
    for bundle in wanted:
        errors, warnings, info = validate(bundle)
        all_errors.extend(errors)
        summary.append((bundle, info))
        status = "OK" if not errors else f"{len(errors)} ERROR(S)"
        print(f"[{status}] {bundle}: agents={info.get('agents')} eval={info.get('eval_config')}")
        for w in warnings:
            print(f"    ~ warn: {w}")
        for e in errors:
            print(f"    - {e}")

    print("\n=== 单变量核对：evaluation 块数值对比 ===")
    base = None
    for bundle, info in summary:
        ev = info.get("eval_config")
        if ev is None:
            continue
        if base is None:
            base = (bundle, ev)
        same = "同" if ev == base[1] else "**不同**"
        print(f"  {bundle}: {ev}  [{same} {base[0]}]")

    if all_errors:
        print(f"\nFAILED: {len(all_errors)} error(s)")
        return 1
    print("\nPASSED: all bundles valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
