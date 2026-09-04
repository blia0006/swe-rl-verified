#!/usr/bin/env python3
"""排查 tracing：打印每步的工具、参数与观察，定位 Agent 卡在哪。

用法：
    python3 experiments/inspect_tracing.py <tracing.jsonl> [--from 1] [--to 8]
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--from", dest="lo", type=int, default=1)
    ap.add_argument("--to", dest="hi", type=int, default=8)
    ap.add_argument("--chars", type=int, default=320)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    steps = [r for r in recs if r.get("type") == "step"]
    meta = next((r for r in recs if r.get("type") == "meta"), {})
    summ = next((r for r in recs if r.get("type") == "summary"), {})

    print(f"task={meta.get('task_id')} rollout={meta.get('rollout_id')} 共 {len(steps)} 步")
    print(f"工具序列: {' → '.join(str(s.get('tool')) for s in steps)}\n")

    for s in steps:
        n = s.get("step", 0)
        if not (args.lo <= n <= args.hi):
            continue
        print(f"--- step {n}  tool={s.get('tool')}  ok={s.get('ok')}")
        th = (s.get("thought") or "").replace("\n", " ")
        if th:
            print(f"    thought: {th[:160]}")
        print(f"    args   : {json.dumps(s.get('args'), ensure_ascii=False)[:args.chars]}")
        obs = (s.get("observation") or "").replace("\n", " | ")
        print(f"    obs    : {obs[:args.chars]}")

    if summ:
        print(
            f"\n汇总: 步数={summ.get('steps_used')} 编辑成功={summ.get('edits_applied')} "
            f"格式错误={summ.get('parse_errors')} patch={summ.get('patch_chars')}字符 "
            f"主动完成={summ.get('finished_by_agent')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
