#!/usr/bin/env python3
"""消融矩阵跨臂分析（预注册指标，见 W2_PHASE2_ABLATION_PLAN.md §3）。

用法:
  python3 analyze_ablation.py --arm B0=ROOT1 [--arm B0p=ROOT2 ...]
  ROOT = 某次运行输出的 w2_results 目录（其下 task_runs/{task}/attempt_*/eval_results/task_results.jsonl）
臂标签由命令行显式指定（不从目录名/run_id 猜测——吸取 kernel v25 provenance 教训）。

输出:
1. 每臂 resolution rate + Wilson 95% CI + 失败模式分布 + 逐仓库分解
2. token 经济性（trace final_metrics：prompt/completion tokens 中位数）——机制 D 指标
3. 配对 McNemar（精确二项）逐对比较（同任务配对，缺失任务剔除并报告）

不读 held-out。只统计 development 各臂实跑产物。
"""
import argparse, glob, json, math, os, statistics, sys
from collections import Counter

def collect(root):
    rows = {}
    for p in sorted(glob.glob(os.path.join(root, 'task_runs', '*', 'attempt_*', 'eval_results', 'task_results.jsonl'))):
        for line in open(p):
            line = line.strip()
            if not line.startswith('{'):
                continue
            d = json.loads(line)
            rows[d['instance_id']] = d  # 同任务多次 attempt 取最后一次读到的
    return rows

def load_tokens(root, instance_id):
    pats = glob.glob(os.path.join(root, 'traces', instance_id, 'attempt_*', f'trace_{instance_id}.json'))
    fm = None
    for p in sorted(pats):
        try:
            fm = json.load(open(p)).get('final_metrics') or {}
        except Exception:
            pass
    return fm or {}

TRUNC_MARKER = 'token limit before the tool call'

def load_trunc(root, instance_id):
    """512 输出上限 tool call 截断指示（§6-5）：trace 中是否出现截断 system 提示。"""
    for p in sorted(glob.glob(os.path.join(root, 'traces', instance_id, 'attempt_*', f'trace_{instance_id}.json'))):
        try:
            d = json.load(open(p))
            return any(TRUNC_MARKER in (s.get('message') or '') for s in d.get('steps', []) if isinstance(s, dict))
        except Exception:
            pass
    return False

def classify(r):
    err = r.get('error') or ''
    if r['resolved']:
        return 'resolved'
    if 'ContextWindowExceeded' in err:
        return 'context_overflow'   # §6-6：token 膨胀致 session 级致命故障，机制 D 直接证据
    if 'tool call budget' in err or 'turns budget' in err:
        return 'budget_truncated'
    if 'session timeout' in err:
        return 'session_timeout'   # §6-5：含 512 截断死循环（长 tool call 无法闭合→重试→超时）
    if r.get('agent_patch_size', 0) > 0:
        return 'patch_failed_tests'
    return 'abandoned_no_patch'    # 预算内正常结束但未提交补丁——机制 C（保底提交）的直接靶子

def wilson(k, n):
    z = 1.96
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [max(0, c-hw), min(1, c+hw)]

def mcnemar_exact(a, b, tasks):
    """精确二项 McNemar：a,b = 两臂 per-task resolved dict。返回 (win, lose, p)。"""
    win = lose = 0
    for t in tasks:
        ra, rb = a.get(t, {}).get('resolved'), b.get(t, {}).get('resolved')
        if ra is True and rb is False: win += 1
        elif ra is False and rb is True: lose += 1
    n = win + lose
    if n == 0:
        return win, lose, 1.0
    k = min(win, lose)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return win, lose, min(1.0, p)

ap = argparse.ArgumentParser()
ap.add_argument('--arm', action='append', required=True,
                help='LABEL=ROOT 或 LABEL=ROOT1+ROOT2（多 root 合并，如 B0 全量 5 批次目录）')
args = ap.parse_args()

arms = {}
for spec in args.arm:
    label, rhs = spec.split('=', 1)
    roots = rhs.split('+')
    rows = {}
    for root in roots:
        if not os.path.isdir(root):
            sys.exit(f"root not found: {root}")
        rows.update(collect(root))   # 同任务跨 root 以最后一个为准
    arms[label] = {'roots': roots, 'rows': rows}

def metric_over_roots(roots, instance_id, fn):
    """跨 root 取该任务的首个有效指标（token）。"""
    for root in roots:
        v = fn(root, instance_id)
        if v:
            return v
    return None

for label, a in arms.items():
    rows = a['rows']
    n = len(rows)
    print(f"\n===== {label} (roots={'+'.join(a['roots'])}) n={n} =====")
    if n == 0:
        print("  (no results)")
        continue
    ok = sum(1 for r in rows.values() if r['resolved'])
    print(f"resolved: {ok}/{n} = {ok/n:.1%} | Wilson 95% CI {[f'{x:.1%}' for x in wilson(ok, n)]}")
    dist = Counter(classify(r) for r in rows.values())
    print("失败模式: " + ", ".join(f"{k}={v} ({v/n:.0%})" for k, v in dist.most_common()))
    print("逐仓库:")
    for repo in ('fastapi', 'rich', 'requests'):
        sub = [r for t, r in rows.items() if t.startswith(repo)]
        if not sub:
            continue
        sk = sum(1 for r in sub if r['resolved'])
        sd = Counter(classify(r) for r in sub)
        print(f"  {repo}: {sk}/{len(sub)} resolved | {dict(sd)}")
    prompts, completions = [], []
    for t in rows:
        fm = metric_over_roots(a['roots'], t, load_tokens) or {}
        if fm.get('total_prompt_tokens'):
            prompts.append(fm['total_prompt_tokens'])
        if fm.get('total_completion_tokens'):
            completions.append(fm['total_completion_tokens'])
    trunc_n = sum(1 for t in rows if any(load_trunc(r, t) for r in a['roots']))
    print(f"tool call 截断（512 上限副作用，§6-5）: {trunc_n}/{n} 任务")
    if prompts:
        print(f"token 经济性: prompt 中位数 {statistics.median(prompts):.0f} (n={len(prompts)}) | "
              f"completion 中位数 {statistics.median(completions):.0f}" if completions else
              f"token 经济性: prompt 中位数 {statistics.median(prompts):.0f} (n={len(prompts)})")
    a['tokens'] = {'prompt': prompts, 'completion': completions}

labels = list(arms)
if len(labels) >= 2:
    print("\n===== 配对 McNemar（精确二项，同任务配对）=====")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            ta, tb = set(arms[la]['rows']), set(arms[lb]['rows'])
            common = sorted(ta & tb)
            only_a, only_b = ta - tb, tb - ta
            win, lose, p = mcnemar_exact(arms[la]['rows'], arms[lb]['rows'], common)
            note = ""
            if only_a or only_b:
                note = f" (剔除不配对任务: {la}独有{len(only_a)}, {lb}独有{len(only_b)})"
            print(f"{la} vs {lb}: 配对 n={len(common)} | {la}赢 {win} / {lb}赢 {lose} | p={p:.4f}{note}")
    # token 对比（机制 D 预期：B1 系 prompt tokens 比 B0p 降 ≥50%）
    print("\n===== token 对比（prompt 中位数）=====")
    for label in labels:
        pr = arms[label].get('tokens', {}).get('prompt') or []
        if pr:
            print(f"  {label}: {statistics.median(pr):.0f} (n={len(pr)})")

print("\n预注册判定（W2_PHASE2_ABLATION_PLAN.md §3）: B1A vs B0p 提升 ≥10pp 且 McNemar p<0.05 → 论文主张成立。")
