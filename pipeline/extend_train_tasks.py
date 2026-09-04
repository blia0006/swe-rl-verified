#!/usr/bin/env python3
"""
在不改动现有 eval 集的前提下，扩充训练题库
==========================================

背景：select_tasks.py 每次运行会从头重新分层抽样并覆盖 train+eval，
这会打乱已经做过"训练前 baseline"测评的 eval 集合，破坏训练前后对比的
连续性。因此扩题必须：

1. 复用 select_tasks.py 里的候选池加载 / 分层抽样逻辑（保持同一套选题标准：
   <15min fix + 单文件 + F2P<=5 + P2P<=60）
2. 排除当前 tasks.jsonl 里已经存在的所有 task_id（无论 train 还是 eval）
3. 只往 split.json 的 train 列表追加新 task_id，eval 列表原样保留
4. 新任务的元数据追加进 tasks.jsonl（不覆盖已有行）

用法：
    python3 pipeline/extend_train_tasks.py --n 30 --per-repo 15
    python3 pipeline/extend_train_tasks.py --n 30 --dry-run
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.select_tasks import load_pool, stratified_pick  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="新增训练题数")
    ap.add_argument("--per-repo", type=int, default=15, help="单 repo 最多新增几题")
    ap.add_argument("--tasks-file", default=str(ROOT / "data" / "tasks.jsonl"))
    ap.add_argument("--split-file", default=str(ROOT / "data" / "split.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks_path = Path(args.tasks_file)
    split_path = Path(args.split_file)

    existing_tasks = [
        json.loads(l) for l in tasks_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    existing_ids = {t["task_id"] for t in existing_tasks}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    eval_ids = set(split["eval"])

    pool = load_pool()
    print(f"候选池：{len(pool)} 题；已占用（train+eval）：{len(existing_ids)} 题")

    new_picks = stratified_pick(pool, args.n, args.per_repo, taken=existing_ids)
    dist = defaultdict(int)
    for t in new_picks:
        dist[t["repo"]] += 1
    print(f"\n本次新增训练题：{len(new_picks)} 题")
    print("  repo 分布：" + ", ".join(f"{k.split('/')[-1]}={v}" for k, v in sorted(dist.items(), key=lambda x: -x[1])))
    for t in new_picks:
        print(f"  {t['task_id']:38s} {t['repo']:26s} patch={t['patch_len']:6d} F2P={t['n_f2p']} P2P={t['n_p2p']}")

    # 安全校验：新增题绝不能撞进已有 eval 集
    overlap = {t["task_id"] for t in new_picks} & eval_ids
    if overlap:
        sys.exit(f"[x] 新增题与现有 eval 集重叠（不应发生）：{overlap}")

    if args.dry_run:
        print("\n[dry-run] 未写入文件")
        return 0

    with open(tasks_path, "a", encoding="utf-8") as f:
        for t in new_picks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    split["train"] = split["train"] + [t["task_id"] for t in new_picks]
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ tasks.jsonl 追加 {len(new_picks)} 题（累计 {len(existing_ids) + len(new_picks)} 题）")
    print(f"✓ split.json train 更新为 {len(split['train'])} 题；eval 保持 {len(split['eval'])} 题不变")
    return 0


if __name__ == "__main__":
    sys.exit(main())
