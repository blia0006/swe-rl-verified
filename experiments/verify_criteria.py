#!/usr/bin/env python3
"""题目有效性验证（官方判据链路）
==================================

判据是 RL 的地基：reward 错了，训练必然空转。因此每道题都必须用
**真实镜像 + 官方 test_cmd** 验证三个场景，缺一不可：

| 场景 | 期望 | 验证了什么 |
|---|---|---|
| ① 空解（不打补丁） | F2P 未全过、P2P 全过 | 题目有区分度，且镜像与标注一致 |
| ② golden patch | F2P 全过、P2P 全过、reward=1.0 | 判据能识别正确答案（否则模型永远拿不到分） |
| ③ 垃圾 patch | apply 失败 | 防作弊链路正常 |

② 不成立 → 该题**不可用**（多为镜像与数据集版本不匹配），必须剔除。

## 与上一版的关键区别

上一版用 `pytest -rA <单个 test id>` 判所有题，对 django（runtests.py）、
sphinx（tox）、sympy（bin/test）三类共 9/20 题**完全不兼容**，导致这些题
恒判 0 分 —— 这是上一轮"近半数训练步 score 全 0"的地基性根因。
本版改走 `pipeline/sandbox_eval`，命令与解析器均取自官方 `swebench` 包，
逐题按 (repo, version) 分派。

## 沙箱复用与日志

同一沙箱实例内连续跑三场景，每次判分前 `git reset --hard` 还原，
省掉 2 次启动开销。并发时**每题日志先缓冲、结束后整块输出** ——
逐行 print 在多线程下会交错，实测出现"A 题标题下显示 B 题结果"的错觉，
足以导致误判。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GARBAGE_PATCH = """--- a/nonexistent_file_xyz.py
+++ b/nonexistent_file_xyz.py
@@ -1,3 +1,3 @@
-old line that does not exist
+new line
 context
"""


def load_env() -> None:
    import logging

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    # e2b SDK 对每个 HTTP 请求打两条 INFO 日志。单题三场景有数十次文件/命令调用，
    # 会把真正的判定结果冲走（实测有效输出被噪声淹没到需要 grep 才能看）。
    for name in ("e2b", "e2b.api.client_sync", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def tcr_image(task_id: str, registry: str, namespace: str, tag: str = "sbx") -> str:
    """`sbx` 是注入了 AGS envd agent 的融合镜像。

    原始 `v1`（官方镜像）缺 agent，沙箱起不来（init command path error）——
    详见 scripts/build_sandbox_images.sh 的说明。
    """
    return f"{registry}/{namespace}/sweb-{task_id.replace('__', '-').lower()}:{tag}"


def sbx_probe(sbx: Any) -> tuple[bool, str]:
    """环境自检：官方镜像应有 /testbed 与 conda testbed 环境。"""
    from pipeline.sandbox_eval import sbx_run

    _, out = sbx_run(sbx, "ls -d /testbed && ls /opt/miniconda3/envs/", 120)
    return ("/testbed" in out and "testbed" in out), out


def verify_task(task: dict, registry: str, namespace: str, tool_id: str) -> dict:
    from clients.ags import AGSClient
    from clients.sandbox import connect_with_retry, start_instance_with_warmup

    from pipeline.sandbox_eval import run_eval

    task_id = task["task_id"]
    image = tcr_image(task_id, registry, namespace)
    rec: dict[str, Any] = {"task_id": task_id, "image": image, "scenarios": {}}

    buf: list[str] = [f"{'=' * 70}", f"=== {task_id}", f"    镜像: {image}"]

    def flush_block() -> dict:
        print("\n" + "\n".join(buf), flush=True)
        return rec

    ags = AGSClient()
    t0 = time.time()
    try:
        instance_id, _ = start_instance_with_warmup(
            ags, tool_id, image, cpu="2", memory="4Gi", verbose=False
        )
    except Exception as e:  # noqa: BLE001
        rec.update(verdict="instance_failed", error=str(e)[:500])
        buf.append(f"    ✗ 沙箱启动失败：{str(e)[:300]}")
        return flush_block()
    buf.append(f"    实例 {instance_id} 启动 {time.time() - t0:.1f}s")

    try:
        sbx = connect_with_retry(instance_id)

        code, out = sbx_probe(sbx)
        rec["layout_ok"] = code
        buf.append(f"    布局检查: {'✓' if code else '✗ ' + out[:200]}")
        if not code:
            rec["verdict"] = "layout_mismatch"
            return flush_block()

        scenarios = (
            ("① 空解     ", None),
            ("② golden   ", task["golden_patch"]),
            ("③ 垃圾patch", GARBAGE_PATCH),
        )
        for label, patch in scenarios:
            t1 = time.time()
            r = run_eval(sbx, task, patch, timeout=1800)
            d = r.to_dict()
            d["elapsed_s"] = round(time.time() - t1, 1)
            rec["scenarios"][label.strip()] = d
            f2p = f"{r.outcome.f2p_passed}/{r.outcome.f2p_total}" if r.outcome else "-"
            p2p = f"{r.outcome.p2p_passed}/{r.outcome.p2p_total}" if r.outcome else "-"
            buf.append(
                f"    {label} stage={r.stage.value:14s} F2P={f2p:7s} P2P={p2p:8s} "
                f"reward={r.reward.reward:.3f} ({d['elapsed_s']}s)"
            )
            if r.error:
                buf.append(f"                 └ {r.error[:150]}")

        rec["verdict"] = judge(rec["scenarios"])
        # 空解基线 reward = 该题的"白给分"。数值越高说明区分度越差，
        # 训练时这部分 reward 与策略好坏无关，是纯噪声。供筛题排序用。
        rec["baseline_reward"] = (
            rec["scenarios"].get("① 空解", {}).get("reward") or {}
        ).get("reward")
        buf.append(
            f"    → 判定: {rec['verdict'].upper()}  空解基线={rec['baseline_reward']}"
        )
        return flush_block()
    except Exception as e:  # noqa: BLE001
        rec.update(verdict="error", error=f"{type(e).__name__}: {str(e)[:400]}")
        buf.append(f"    ✗ 异常：{type(e).__name__}: {str(e)[:300]}")
        return flush_block()
    finally:
        try:
            ags.stop_instance(instance_id)
        except Exception:  # noqa: BLE001,S110 —— 回收失败不应影响验证结论
            pass


def judge(sc: dict[str, dict]) -> str:
    """三场景合起来判定题目是否可用。

    ## 判定口径（含一次实测修正）

    最初要求「空解时 F2P **全部** fail」，实测 django-16429 出现 F2P=1/4 ——
    并非题目无效，而是官方 F2P 列表里含 `subtest`，同名条目被 runner 拆成多次
    上报，其中一部分与本 bug 无关、本来就通过。SWE-bench Verified 里这类
    标注噪声很常见。

    因此改为**有区分度**判定：

    | 场景 | 要求 | 理由 |
    |---|---|---|
    | ① 空解 | F2P 未全过（`pass < total`） | 只要不是"什么都不做就满分"，题目就有学习信号 |
    | ① 空解 | P2P 必须全过 | 若空解就有 P2P 挂，说明镜像环境与标注不一致，reward 无意义 |
    | ② golden | resolved 且 reward=1.0 | 判据必须能认出正确答案，否则模型永远拿不到分 |
    | ③ 垃圾 | apply 被拒 | 防作弊链路正常 |

    空解基线 reward 一并记录：它是该题的"白给分"，数值越高说明区分度越差，
    供后续筛题排序用。
    """
    empty = sc.get("① 空解", {})
    golden = sc.get("② golden", {})
    garbage = sc.get("③ 垃圾patch", {})

    eg = empty.get("grade_detail") or {}
    gg = golden.get("grade_detail") or {}

    empty_has_signal = (
        empty.get("stage") == "tested"
        and eg.get("f2p_total", 0) > 0
        and eg.get("f2p_pass", 0) < eg.get("f2p_total", 0)
    )
    empty_p2p_clean = eg.get("p2p_total") is not None and eg.get("p2p_pass") == eg.get(
        "p2p_total"
    )
    golden_ok = (
        golden.get("stage") == "tested"
        and gg.get("resolved") is True
        and abs((golden.get("reward") or {}).get("reward", 0) - 1.0) < 1e-6
    )
    garbage_ok = garbage.get("stage") in ("apply_failed", "collect_error")

    if empty_has_signal and empty_p2p_clean and golden_ok and garbage_ok:
        return "usable"
    reasons = []
    if not empty_has_signal:
        reasons.append("空解无区分度(F2P全过或未跑成)")
    if not empty_p2p_clean:
        reasons.append("空解P2P即有失败(镜像与标注不一致)")
    if not golden_ok:
        reasons.append("golden未判满分")
    if not garbage_ok:
        reasons.append("垃圾补丁未被拒")
    return "unusable:" + "+".join(reasons)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", help="指定 task_id，可重复")
    ap.add_argument("--all", action="store_true", help="验证 tasks.jsonl 全部题目")
    ap.add_argument("--jobs", type=int, default=1, help="并发题目数")
    ap.add_argument("--out", default="data/criteria_check.json")
    args = ap.parse_args()

    load_env()
    import os

    registry = os.environ["TCR_REGISTRY"]
    namespace = os.environ["TCR_NAMESPACE"]
    tool_name = os.environ.get("AGS_TOOL_NAME", "swe-rl-vpc-runner")

    from clients.ags import AGSClient

    tool = AGSClient().find_tool(tool_name)
    if not tool:
        print(f"✗ 找不到沙箱工具 {tool_name}", file=sys.stderr)
        return 2
    tool_id = tool["tool_id"]

    tasks = [json.loads(l) for l in open(ROOT / "data/tasks.jsonl") if l.strip()]
    if args.task:
        want = set(args.task)
        tasks = [t for t in tasks if t["task_id"] in want]
    elif not args.all:
        print("需指定 --task 或 --all", file=sys.stderr)
        return 2

    print(f"待验证 {len(tasks)} 题，并发 {args.jobs}", flush=True)
    t0 = time.time()
    results: list[dict] = []
    if args.jobs <= 1:
        for t in tasks:
            results.append(verify_task(t, registry, namespace, tool_id))
    else:
        with ThreadPoolExecutor(args.jobs) as ex:
            futs = {
                ex.submit(verify_task, t, registry, namespace, tool_id): t for t in tasks
            }
            for f in as_completed(futs):
                results.append(f.result())

    usable = [r for r in results if r.get("verdict") == "usable"]
    print(f"\n{'=' * 70}\n可用 {len(usable)}/{len(results)} 题，耗时 {time.time() - t0:.0f}s")
    for r in results:
        if r.get("verdict") != "usable":
            print(f"  ✗ {r['task_id']}: {r.get('verdict')} {r.get('error', '')[:120]}")

    out = ROOT / args.out
    out.write_text(
        json.dumps(
            {"results": results, "usable": [r["task_id"] for r in usable]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out}")
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
