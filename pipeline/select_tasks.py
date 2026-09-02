#!/usr/bin/env python3
"""
从 SWE-bench Verified 选题并生成任务元数据
==========================================

选题策略（每一条都是为了让 7B 模型「有机会做对」，同时保证评测有统计功效）：

1. **难度**：只取官方标注 `<15 min fix`（194/500 道）。
   理由：SWE-bench Verified 整体很难，顶尖闭源模型 + 完整 Agent 框架的 pass@1 也就
   50%~70%。7B 单轮生成 patch 若挑难题，pass@1 大概率恒为 0，训练信号全无
   （上一轮就栽在"reward 全 0 → advantage 全 0 → 梯度为 0"的死循环）。

2. **单文件改动**：golden patch 只碰 1 个文件。
   理由：多文件改动要求模型同时定位多处，对 7B 不现实；且 search/replace 格式下
   单文件的 prompt 组织最清晰。

3. **测试规模适中**：`F2P ≤ 5`、`P2P ≤ 60`。
   理由：F2P 太多则 reward 分辨率过粗；P2P 太多则 `pytest` 单次执行过慢
   （沙箱打分是 GRPO 每步的耗时瓶颈）。

4. **按 repo 分层抽样**：单个 repo 最多占 3 题。
   理由：Verified 里 django 占 231/500，不分层会选出一堆 django，
   模型可能学到"django 特有的代码风格"而非通用修复能力，且镜像层高度重复
   会让"跨项目泛化"的结论不成立。

5. **训练集 / 评测集严格不重叠**，且各自独立分层。

划分（与验收标准对应）：
    · 训练集 10 题 → 线 A 采集 10 条 tracing，直接对应验收第 1 条「≥10 题」
                     线 B 用它们跑 GRPO
    · 评测集 10 题 → 训练前后 pass@1 对比（验收第 6 条），k=8 采样 = 80 个样本

**为什么评测集也要 10 题**：课题只对采集侧规定了「≥10 题」，对评测侧只要求
「有对比数据」，未给题数。但评测规模直接决定结论能否成立 ——
上一轮用 4 题 × k=1，而 strict 成功率实测仅 1.8%，期望成功次数 = 4×0.018 = **0.07**，
结果必然全 0、Welch t 检验完全不显著，等于没有结论。
样本数 = 题数 × k，10 题 × 8 = 80 个样本，相对标准误比 4 题方案降约 37%。
镜像体积曾是压缩题量的理由，但节点只是搬运中转站（沙箱从 TCR 拉镜像），
搬完即 `ctr content prune references` 回收（实测一次回收 18GiB），磁盘不再是约束。

用法：
    python3 pipeline/select_tasks.py                     # 生成 data/tasks.jsonl + split.json
    python3 pipeline/select_tasks.py --train 10 --eval 8
    python3 pipeline/select_tasks.py --dry-run           # 只看选出哪些题，不写文件
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "swebench_verified.parquet"

# SWE-bench 官方镜像命名：instance_id 里的 `__` 编码为 `_1776_`
# （实测确认，例：astropy__astropy-12907 → sweb.eval.x86_64.astropy_1776_astropy-12907）
IMAGE_FMT = "swebench/sweb.eval.x86_64.{slug}:latest"


def image_slug(instance_id: str) -> str:
    return instance_id.replace("__", "_1776_")


def official_image(instance_id: str) -> str:
    return IMAGE_FMT.format(slug=image_slug(instance_id))


def as_list(v) -> list:
    """F2P/P2P 字段在 parquet 里是 JSON 字符串，统一成 list。"""
    if isinstance(v, str):
        return json.loads(v)
    return list(v) if v is not None else []


def patched_files(patch: str) -> list[str]:
    """从 unified diff 里取出被改动的文件路径。"""
    files = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
    return files


def load_pool():
    import pandas as pd

    if not DATASET.exists():
        sys.exit(
            f"[x] 找不到数据集 {DATASET}\n"
            "    下载：curl -sL -o data/swebench_verified.parquet \\\n"
            "      https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified"
            "/resolve/main/data/test-00000-of-00001.parquet"
        )
    df = pd.read_parquet(DATASET)
    rows = []
    for _, r in df.iterrows():
        if r["difficulty"] != "<15 min fix":
            continue
        f2p, p2p = as_list(r["FAIL_TO_PASS"]), as_list(r["PASS_TO_PASS"])
        files = patched_files(r["patch"])
        if len(set(files)) != 1:
            continue
        if not (1 <= len(f2p) <= 5) or len(p2p) > 60:
            continue
        rows.append(
            {
                "task_id": r["instance_id"],
                "repo": r["repo"],
                "base_commit": r["base_commit"],
                "environment_setup_commit": r["environment_setup_commit"],
                "version": str(r["version"]),
                "difficulty": r["difficulty"],
                "problem_statement": r["problem_statement"],
                "golden_patch": r["patch"],
                "test_patch": r["test_patch"],
                "fail_to_pass": f2p,
                "pass_to_pass": p2p,
                "modified_files": sorted(set(files)),
                "official_image": official_image(r["instance_id"]),
                "patch_len": len(r["patch"]),
                "n_f2p": len(f2p),
                "n_p2p": len(p2p),
            }
        )
    return rows


def stratified_pick(pool: list[dict], n: int, per_repo: int, taken: set[str]) -> list[dict]:
    """按 repo 分层挑 n 题：每轮从各 repo 各取 1 道最简单的，轮转直到取满。"""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in pool:
        if t["task_id"] in taken:
            continue
        by_repo[t["repo"]].append(t)
    for v in by_repo.values():
        v.sort(key=lambda t: (t["patch_len"], t["n_p2p"]))

    # repo 顺序：题目多的 repo 优先（其题目质量与镜像可得性更有保障）
    order = sorted(by_repo, key=lambda k: -len(by_repo[k]))
    picked: list[dict] = []
    used_per_repo: dict[str, int] = defaultdict(int)
    round_no = 0
    while len(picked) < n and round_no < per_repo:
        progressed = False
        for repo in order:
            if len(picked) >= n:
                break
            if used_per_repo[repo] > round_no:
                continue
            bucket = by_repo[repo]
            idx = used_per_repo[repo]
            if idx < len(bucket):
                picked.append(bucket[idx])
                used_per_repo[repo] += 1
                progressed = True
        if not progressed:
            break
        round_no += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10, help="训练集题数（验收第 1 条要求 ≥10）")
    ap.add_argument("--eval", type=int, default=10, help="评测集题数（决定 pass@1 对比的统计功效）")
    ap.add_argument(
        "--per-repo",
        type=int,
        default=4,
        help="单个 repo 最多贡献几题。候选池按 repo 极不均衡（django 54 / 多个 repo 仅 1），"
        "per-repo=3 时上限恰为 20 题（正好卡满 10+10，无余量），故默认放宽到 4",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pool = load_pool()
    print(f"候选池：{len(pool)} 题（<15min fix + 单文件 + F2P≤5 + P2P≤60）")
    dist: dict[str, int] = defaultdict(int)
    for t in pool:
        dist[t["repo"]] += 1
    print("  repo 分布：" + ", ".join(f"{k.split('/')[-1]}={v}" for k, v in sorted(dist.items(), key=lambda x: -x[1])))

    if args.train + args.eval > len(pool):
        sys.exit(f"[x] 候选池不足：需要 {args.train + args.eval}，仅有 {len(pool)}")

    taken: set[str] = set()
    train = stratified_pick(pool, args.train, args.per_repo, taken)
    taken |= {t["task_id"] for t in train}
    evalset = stratified_pick(pool, args.eval, args.per_repo, taken)
    taken |= {t["task_id"] for t in evalset}

    # 防泄漏：这是上一轮修过的 bug，此处硬校验
    overlap = {t["task_id"] for t in train} & {t["task_id"] for t in evalset}
    if overlap:
        sys.exit(f"[x] 训练/评测集重叠：{overlap}")
    if len(train) < args.train or len(evalset) < args.eval:
        sys.exit(
            f"[x] 分层抽样未取满（train={len(train)}/{args.train}, "
            f"eval={len(evalset)}/{args.eval}），可放宽 --per-repo"
        )

    for label, items in (("训练集", train), ("评测集", evalset)):
        print(f"\n{label}（{len(items)} 题）:")
        print(f"  {'task_id':38s} {'repo':26s} {'patch':>6s} {'F2P':>4s} {'P2P':>4s}")
        for t in items:
            print(
                f"  {t['task_id']:38s} {t['repo']:26s} "
                f"{t['patch_len']:6d} {t['n_f2p']:4d} {t['n_p2p']:4d}"
            )

    if args.dry_run:
        print("\n[dry-run] 未写入文件")
        return 0

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    all_tasks = train + evalset
    with open(out_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    split = {
        "train": [t["task_id"] for t in train],
        "eval": [t["task_id"] for t in evalset],
        "criteria": {
            "difficulty": "<15 min fix",
            "single_file_patch": True,
            "max_f2p": 5,
            "max_p2p": 60,
            "max_per_repo": args.per_repo,
        },
    }
    (out_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ 已写入 data/tasks.jsonl（{len(all_tasks)} 题）与 data/split.json")
    print("  镜像清单（供搬运脚本使用）：")
    for t in all_tasks[:3]:
        print(f"    {t['official_image']}")
    print(f"    …共 {len(all_tasks)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
