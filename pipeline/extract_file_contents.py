#!/usr/bin/env python3
"""
从沙箱抽取题目相关文件的真实内容
================================

两处都要用，且必须用**同一份**内容，否则 search/replace 会对不上：

  · 构建训练 prompt 时嵌入文件内容 —— 上一轮实测：不嵌内容会让模型盲写，
    reward 恒 0
  · reward function 把 search/replace 块转 unified diff 时作为原文

⚠️ 必须从**沙箱内**读取，不能用数据集里的 patch 反推：
上一轮踩过本地缓存与镜像内容不一致的坑（一道题的 golden.patch 与镜像里的
实际文件对不上，导致误判为代码 bug，排查了很久）。镜像里的才是唯一事实。

抽取哪些文件：从 golden patch 的 `+++ b/<path>` 取，即"标准答案改过的文件"。
这不算泄题 —— prompt 只给文件内容（模型本就该有权读代码库），不给 patch 本身；
真实 Agent 场景里定位文件是另一个子任务，本课题聚焦于修复能力。

用法：
    python3 pipeline/extract_file_contents.py            # 全部题目
    python3 pipeline/extract_file_contents.py --task <id>
    python3 pipeline/extract_file_contents.py --jobs 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_FILE_CHARS = 60000  # 单文件上限，防止 prompt 爆掉


def load_env():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)


def patched_files(patch):
    return [ln[6:].strip() for ln in patch.splitlines() if ln.startswith("+++ b/")]


def sbx_run(sbx, cmd, timeout=120):
    """e2b 在非 0 退出码时抛异常，这里统一收敛成返回值。"""
    try:
        r = sbx.commands.run(cmd, user="root", timeout=timeout)
        return (r.exit_code or 0), (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        code = getattr(e, "exit_code", None)
        out = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
        return (code if code is not None else 1), (out or str(e))


def extract_one(task, registry, namespace, tool_id, image_tag):
    from clients.ags import AGSClient
    from clients.sandbox import start_instance_with_warmup
    from e2b_code_interpreter import Sandbox

    task_id = task["task_id"]
    slug = task_id.replace("__", "-").lower()
    image = "%s/%s/sweb-%s:%s" % (registry, namespace, slug, image_tag)
    wanted = patched_files(task["golden_patch"])
    print("\n=== %s（%d 个文件）" % (task_id, len(wanted)), flush=True)

    ags = AGSClient()
    inst = None
    try:
        inst, _ = start_instance_with_warmup(ags, tool_id, image, cpu="2", memory="4Gi")
        sbx = Sandbox.connect(inst)
        out = {}
        for rel in wanted:
            code, content = sbx_run(sbx, "cat /testbed/%s" % rel, 120)
            if code != 0:
                print("    ✗ %s 读取失败：%s" % (rel, content[:150]))
                continue
            if len(content) > MAX_FILE_CHARS:
                print("    ! %s 过大（%d 字符），截断" % (rel, len(content)))
                content = content[:MAX_FILE_CHARS]
            out[rel] = content
            print("    ✓ %s（%d 字符 / %d 行）" % (rel, len(content), content.count("\n") + 1))
        return task_id, out
    except Exception as e:
        print("    ✗ %s: %s" % (task_id, str(e)[:250]))
        return task_id, {}
    finally:
        if inst:
            try:
                ags.stop_instance(inst)
            except Exception as e:
                print("    ⚠️ 实例回收失败 %s: %s" % (inst[:16], e))


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", default=str(ROOT / "data" / "tasks.jsonl"))
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--image-tag", default=os.environ.get("SANDBOX_IMAGE_TAG", "sbx"))
    ap.add_argument("--out", default=str(ROOT / "data" / "file_contents.json"))
    args = ap.parse_args()

    registry = os.environ.get("TCR_REGISTRY", "")
    namespace = os.environ.get("TCR_NAMESPACE", "")
    tool_name = os.environ.get("AGS_TOOL_NAME") or os.environ.get("SWE_SYNTH_SHARED_TOOL", "")
    if not (registry and namespace and tool_name):
        sys.exit("[x] 需要 .env 的 TCR_REGISTRY / TCR_NAMESPACE / AGS_TOOL_NAME")

    from clients.ags import AGSClient

    tool = AGSClient().find_tool(tool_name)
    if not tool:
        sys.exit("[x] 找不到沙箱工具 %s" % tool_name)
    tool_id = tool["tool_id"]

    tasks = [
        json.loads(l)
        for l in Path(args.tasks_file).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if args.task:
        want = set(args.task)
        tasks = [t for t in tasks if t["task_id"] in want]

    out_path = Path(args.out)
    merged = {}
    if out_path.exists():
        try:
            merged = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}

    todo = [t for t in tasks if not merged.get(t["task_id"])]
    print("待抽取 %d 题（已有 %d 题）；并发 %d" % (len(todo), len(merged), args.jobs))
    if not todo:
        print("✓ 全部已就位")
        return 0

    t0 = time.time()
    results = []
    if args.jobs <= 1:
        for t in todo:
            results.append(extract_one(t, registry, namespace, tool_id, args.image_tag))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(extract_one, t, registry, namespace, tool_id, args.image_tag): t["task_id"]
                for t in todo
            }
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append((futs[f], {}))
                    print("    ✗ %s 异常：%s" % (futs[f], str(e)[:200]))

    for task_id, files in results:
        if files:
            merged[task_id] = files

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")

    ok = [tid for tid, f in results if f]
    print("\n%s\n成功 %d/%d，耗时 %.0fs" % ("=" * 60, len(ok), len(results), time.time() - t0))
    for tid, f in results:
        if not f:
            print("  ✗ %s 未抽到任何文件" % tid)
    print("已写入 %s（累计 %d 题）" % (args.out, len(merged)))
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
