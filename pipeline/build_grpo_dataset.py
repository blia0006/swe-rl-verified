#!/usr/bin/env python3
"""
构建 VERL GRPO 训练数据集（parquet）
====================================

## prompt 设计的三条依据（都来自上一轮的实测教训）

1. **必须嵌入文件真实内容**
   上一轮实测：不嵌内容 → 模型盲写 → reward 恒 0。
   模型得看见代码才能"认出"要改的片段。

2. **用 search/replace 而非 unified diff**
   上一轮 440 采样中 227 次 `corrupt patch`，全因手算 hunk header 出错。
   search/replace 不需要行号 —— 把"算数"这项与修复能力无关的负担移除。

3. **文件路径以固定模板给出**
   上一轮实测模型会写错路径（丢 `src/` 前缀、凭空加前缀），
   这类样本连 `--recount -C0` 都救不回。prompt 里把确切路径列清楚。

## 数据规模与 step 数的关系

VERL 里 `step 数 = 样本行数 ÷ train_batch_size × total_epochs`。
验收要求 ≥50 step，因此：

    10 题 × repeat 次数 = 总行数，train_batch_size=1 时行数即 step 数

默认 repeat=6 → 60 行 → 60 step，留出余量。
`shuffle=False` 让题目轮转均匀（每题恰好训 6 次），
避免某题连续出现造成的局部过拟合。

用法：
    python3 pipeline/build_grpo_dataset.py
    python3 pipeline/build_grpo_dataset.py --repeat 6 --max-file-lines 400
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = """You are an expert software engineer fixing a bug in a real \
open-source repository.

You will be given an issue description and the current content of the relevant \
source file(s). Your task is to produce the minimal code edit that fixes the issue.

Respond ONLY with one or more edit blocks in exactly this format:

### <file path>
<<<<<<< SEARCH
<exact lines copied from the current file>
=======
<the replacement lines>
>>>>>>> REPLACE

Rules:
- The SEARCH section must match the current file content EXACTLY, character for \
character, including indentation.
- Keep SEARCH sections short but unique enough to match only one location.
- Do NOT write line numbers. Do NOT write a unified diff.
- Use the exact file path given in the prompt.
- Make the smallest change that fixes the issue. Do not reformat unrelated code.
- Output nothing except the edit block(s)."""


def build_user_prompt(task, contents, max_file_lines):
    """组装单题的 user prompt。"""
    issue = (task.get("problem_statement") or "").strip()
    # 过长的 issue 会挤掉文件内容；保留头部（问题描述通常在前面）
    if len(issue) > 6000:
        issue = issue[:6000] + "\n...(truncated)"

    parts = [
        "# Repository\n%s" % task["repo"],
        "\n# Issue\n%s" % issue,
        "\n# Files you may edit",
    ]
    # 路径以固定清单给出 —— 缓解上一轮实测到的路径幻觉
    for path in contents:
        parts.append("- `%s`" % path)

    parts.append("\n# Current file content")
    for path, text in contents.items():
        lines = text.split("\n")
        shown = lines[:max_file_lines]
        truncated = len(lines) > max_file_lines
        parts.append("\n### %s" % path)
        parts.append("```python")
        parts.append("\n".join(shown))
        if truncated:
            parts.append("... (file truncated at %d lines)" % max_file_lines)
        parts.append("```")

    parts.append(
        "\n# Task\nProduce the edit block(s) that fix the issue described above."
    )
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", default=str(ROOT / "data" / "tasks.jsonl"))
    ap.add_argument("--split-file", default=str(ROOT / "data" / "split.json"))
    ap.add_argument("--contents-file", default=str(ROOT / "data" / "file_contents.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "grpo_train.parquet"))
    ap.add_argument(
        "--repeat",
        type=int,
        default=6,
        help="每题重复次数。train_batch_size=1 时，总行数即训练 step 数（验收要求 ≥50）",
    )
    ap.add_argument("--max-file-lines", type=int, default=400,
                    help="单文件在 prompt 中最多展示多少行")
    ap.add_argument("--preview", action="store_true", help="打印一条完整 prompt 后退出")
    ap.add_argument(
        "--split-key",
        default="train",
        choices=["train", "eval"],
        help="从 split.json 取哪一部分题目。用 eval 可单独构建"
             "真正独立于训练集的held-out验证 parquet（供 data.val_files 用）",
    )
    args = ap.parse_args()

    tasks = {
        json.loads(l)["task_id"]: json.loads(l)
        for l in Path(args.tasks_file).read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    contents_all = json.loads(Path(args.contents_file).read_text(encoding="utf-8"))

    target_ids = split[args.split_key]
    missing = [t for t in target_ids if not contents_all.get(t)]
    if missing:
        print("[!] 以下题目缺文件内容，将被跳过（先跑 extract_file_contents.py）：")
        for t in missing:
            print("    " + t)
    usable = [t for t in target_ids if contents_all.get(t)]
    if not usable:
        sys.exit("[x] 没有可用题目")

    # 防泄漏：训练集绝不能碰评测集（上一轮修过的 bug，这里硬校验）
    other_key = "eval" if args.split_key == "train" else "train"
    overlap = set(usable) & set(split[other_key])
    if overlap:
        sys.exit("[x] %s 集与 %s 集重叠：%s" % (args.split_key, other_key, overlap))

    if args.preview:
        tid = usable[0]
        print(build_user_prompt(tasks[tid], contents_all[tid], args.max_file_lines))
        return 0

    rows = []
    # 轮转而非"同题连续"：让每个 step 见到不同题目，
    # 避免连续同题造成的局部过拟合与 reward 曲线锯齿
    for r in range(args.repeat):
        for tid in usable:
            task = tasks[tid]
            contents = contents_all[tid]
            user = build_user_prompt(task, contents, args.max_file_lines)
            rows.append(
                {
                    "data_source": "swebench_verified",
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "ability": "code_repair",
                    # VERL 会把 reward_model.ground_truth 传给 compute_score
                    "reward_model": {"style": "rule", "ground_truth": tid},
                    "extra_info": {
                        "task_id": tid,
                        "repo": task["repo"],
                        "index": len(rows),
                        "round": r,
                    },
                }
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    lens = [len(r["prompt"][1]["content"]) for r in rows]
    print("✓ 已写入 %s" % out)
    print("  行数      : %d（%d 题 × %d 轮）" % (len(rows), len(usable), args.repeat))
    print("  → step 数 : %d（train_batch_size=1，验收要求 ≥50）" % len(rows))
    print("  prompt 字符: 最小 %d / 中位 %d / 最大 %d"
          % (min(lens), sorted(lens)[len(lens) // 2], max(lens)))
    print("  预估 token : 最大约 %d（按 3.5 字符/token 估）" % (max(lens) // 3.5))
    if max(lens) / 3.5 > 7000:
        print("  [!] 最长 prompt 偏大，注意 max_prompt_length 要够，"
              "或调小 --max-file-lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
