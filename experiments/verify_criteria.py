#!/usr/bin/env python3
"""
判据适配层的真实沙箱验证（本轮最关键的一次验证）
================================================

判据层是全链路的地基：reward 错了，训练必然空转（上一轮就因判分链路的
异常处理 bug，把 78 次 apply 失败误报成"基础设施故障"，差点据此写错结论）。

因此必须用**真实镜像 + 真实 pytest** 验证三个场景，缺一不可：

| 场景 | 期望 | 验证了什么 |
|---|---|---|
| ① 空解（不打 patch） | F2P **全 fail**、P2P 全 pass | 题目本身有效：修复前确实失败，且判据能识别 |
| ② golden patch | F2P **全 pass**、P2P 全 pass、reward=1.0 | 判据能识别正确答案（否则模型永远拿不到分） |
| ③ 垃圾 patch | apply 失败 或 collect_error | 防作弊/防误判链路正常 |

只要 ① 或 ② 不成立，这道题就**不可用**（可能是镜像与数据集版本不匹配），
必须从题目集里剔除 —— 上一轮 21 题里有 6 题因此被剔除，这次提前查。

用法：
    python3 experiments/verify_criteria.py --task scikit-learn__scikit-learn-14141
    python3 experiments/verify_criteria.py --all          # 全部已搬运的题
    python3 experiments/verify_criteria.py --all --jobs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VERIFY_SCRIPT = ROOT / "sandbox_agent" / "swebench_verify.py"
GARBAGE_PATCH = """--- a/nonexistent_file_xyz.py
+++ b/nonexistent_file_xyz.py
@@ -1,3 +1,3 @@
-old line that does not exist
+new line
 context
"""


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)


def tcr_image(task_id: str, registry: str, namespace: str, tag: str = "sbx") -> str:
    """与 scripts/pod_build_sandbox_images.py::tcr_ref 保持一致。

    默认用 `sbx` 标签 —— 即注入了 AGS envd agent 的融合镜像。
    原始 `v1` 标签是官方镜像，缺 agent，沙箱起不来（init command path error）。
    """
    slug = task_id.replace("__", "-").lower()
    return f"{registry}/{namespace}/sweb-{slug}:{tag}"


def sbx_run(sbx, cmd: str, timeout: int = 900) -> tuple[int, str]:
    """执行沙箱命令并把异常收敛成返回值。

    ⚠️ e2b SDK 在**退出码非 0 时直接抛异常**，不返回 exit_code。
    因此 `if res.exit_code != 0` 这类判断永远走不到，所有失败都会被外层
    except 兜成"沙箱调用异常" —— 上一轮实测导致 78 次 apply 失败被误报成
    基础设施故障，失败归因完全失真。必须在这里收敛。
    """
    try:
        res = sbx.commands.run(cmd, user="root", timeout=timeout)
        return (res.exit_code or 0), (res.stdout or "") + (res.stderr or "")
    except Exception as e:
        code = getattr(e, "exit_code", None)
        out = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
        return (code if code is not None else 1), (out or str(e))


def verify_task(task: dict, registry: str, namespace: str, tool_id: str) -> dict:
    """对单题跑三场景验证。"""
    from clients.ags import AGSClient
    from clients.sandbox import start_instance_with_warmup
    from e2b_code_interpreter import Sandbox

    from pipeline.reward import Stage, compute_reward, outcome_from_json

    task_id = task["task_id"]
    image = tcr_image(task_id, registry, namespace)
    rec: dict = {"task_id": task_id, "image": image, "scenarios": {}}
    print(f"\n{'=' * 70}\n=== {task_id}\n    镜像: {image}", flush=True)

    ags = AGSClient()
    t0 = time.time()
    try:
        # 走带预热等待的封装：AGS 首次使用新镜像需约 4 分钟预热，
        # 期间报 ImagePrepare / still preparing，都不是真故障（详见 clients/sandbox.py）
        instance_id, eff = start_instance_with_warmup(
            ags, tool_id, image, cpu="2", memory="4Gi"
        )
    except Exception as e:
        rec["verdict"] = "instance_failed"
        rec["error"] = str(e)[:500]
        print(f"    ✗ 沙箱启动失败：{str(e)[:300]}")
        return rec
    print(f"    实例 {instance_id} 启动 {time.time() - t0:.1f}s")

    try:
        sbx = Sandbox.connect(instance_id)

        # 环境自检：官方镜像应有 /testbed 与 conda 环境
        code, out = sbx_run(sbx, "ls -d /testbed && ls /opt/miniconda3/envs/", 120)
        rec["layout_ok"] = "/testbed" in out and "testbed" in out
        print(f"    布局检查: {'✓' if rec['layout_ok'] else '✗ ' + out[:200]}")
        if not rec["layout_ok"]:
            rec["verdict"] = "layout_mismatch"
            return rec

        # 注入判据脚本与题目规格
        # ⚠️ 必须显式 user="root"：SWE-bench 官方镜像里没有名为 `user` 的账号，
        # 而 e2b SDK 默认以 `user` 身份写文件，会报
        #   AuthenticationException: error looking up user 'user'
        spec = {
            "task_id": task_id,
            "fail_to_pass": task["fail_to_pass"],
            "pass_to_pass": task["pass_to_pass"],
            "test_patch": task["test_patch"],
        }
        sbx_run(sbx, "mkdir -p /task", 60)
        for path, content in (
            ("/task/swebench_verify.py", VERIFY_SCRIPT.read_text(encoding="utf-8")),
            ("/task/spec.json", json.dumps(spec, ensure_ascii=False)),
            ("/task/golden.diff", task["golden_patch"]),
            ("/task/garbage.diff", GARBAGE_PATCH),
        ):
            sbx.files.write(path, content, user="root")

        # ⚠️ 必须用镜像自带的 conda python，不能用系统 python3：
        # 官方镜像的系统 python 可能老到不支持 `from __future__ import annotations`
        # （实测报 "future feature annotations is not defined"），
        # 而 /opt/miniconda3/envs/testbed 才是题目实际的运行环境
        # 判据脚本本身用**系统 python3**（3.10）执行 —— testbed 环境的解释器
        # 可能老至 3.6（随题目依赖而定），跑不了稍新的语法。
        # 脚本内部会自行调用 testbed 的 python 来跑 pytest（见 TESTBED_PY）。
        # ⚠️ 必须写**绝对路径** /usr/bin/python3：
        # 镜像的 PATH 把 conda 排在前面，裸写 `python3` 会解析到 testbed 环境的
        # python3.6（实测报 subprocess 无 capture_output 参数）。
        # 判据脚本要跑在系统 3.10 上，pytest 才由它内部调 testbed python 执行。
        base = "/usr/bin/python3 /task/swebench_verify.py --spec /task/spec.json"

        def scenario(name: str, extra: str) -> dict:
            t = time.time()
            code, out = sbx_run(sbx, f"{base} --out /task/r.json --restore {extra}", 900)
            try:
                payload = sbx.files.read("/task/r.json", user="root")
                res = json.loads(payload)
            except Exception as e:
                print(f"    ✗ {name}: 读不到 result.json（{str(e)[:150]}）")
                print(f"      命令输出: {out[-400:]}")
                return {"error": "no_result", "stdout_tail": out[-800:]}
            res["_elapsed_s"] = round(time.time() - t, 1)
            return res

        # ---------- ① 空解基线 ----------
        r = scenario("空解", "")
        f2p, p2p = r.get("fail_to_pass", {}), r.get("pass_to_pass", {})
        empty_ok = (
            r.get("stage") == "tested"
            and f2p.get("passed", -1) == 0            # F2P 必须全 fail
            and p2p.get("passed") == p2p.get("total")  # P2P 必须全 pass
        )
        rec["scenarios"]["empty"] = {
            "stage": r.get("stage"),
            "f2p": f"{f2p.get('passed')}/{f2p.get('total')}",
            "p2p": f"{p2p.get('passed')}/{p2p.get('total')}",
            "ok": empty_ok,
            "elapsed_s": r.get("_elapsed_s"),
            "tail": "" if empty_ok else str(r.get("raw_tail", ""))[-1200:],
        }
        print(
            f"    ① 空解      stage={str(r.get('stage')):14s} "
            f"F2P={f2p.get('passed')}/{f2p.get('total')} "
            f"P2P={p2p.get('passed')}/{p2p.get('total')} "
            f"{'✓' if empty_ok else '✗ 期望 F2P=0/N 且 P2P 全过'} "
            f"({r.get('_elapsed_s')}s)"
        )

        # ---------- ② golden patch ----------
        r = scenario("golden", "--patch /task/golden.diff")
        f2p, p2p = r.get("fail_to_pass", {}), r.get("pass_to_pass", {})
        golden_ok = r.get("stage") == "tested" and f2p.get("passed") == f2p.get("total")
        reward = 0.0
        if r.get("stage") == "tested":
            bd = compute_reward(Stage.TESTED, outcome_from_json(r))
            reward = bd.reward
        rec["scenarios"]["golden"] = {
            "stage": r.get("stage"),
            "f2p": f"{f2p.get('passed')}/{f2p.get('total')}",
            "p2p": f"{p2p.get('passed')}/{p2p.get('total')}",
            "apply_strategy": r.get("apply_strategy"),
            "reward": reward,
            "ok": golden_ok and abs(reward - 1.0) < 1e-6,
            "elapsed_s": r.get("_elapsed_s"),
            "tail": "" if golden_ok else str(r.get("raw_tail", ""))[-1200:],
        }
        print(
            f"    ② golden    stage={str(r.get('stage')):14s} "
            f"F2P={f2p.get('passed')}/{f2p.get('total')} "
            f"P2P={p2p.get('passed')}/{p2p.get('total')} "
            f"reward={reward:.3f} "
            f"{'✓' if rec['scenarios']['golden']['ok'] else '✗ 期望 F2P 全过且 reward=1.0'} "
            f"({r.get('_elapsed_s')}s)"
        )

        # ---------- ③ 垃圾 patch ----------
        r = scenario("垃圾", "--patch /task/garbage.diff")
        garbage_ok = r.get("stage") in ("apply_failed", "collect_error")
        rec["scenarios"]["garbage"] = {
            "stage": r.get("stage"),
            "ok": garbage_ok,
            "elapsed_s": r.get("_elapsed_s"),
        }
        print(
            f"    ③ 垃圾patch stage={str(r.get('stage')):14s} "
            f"{'✓' if garbage_ok else '✗ 期望 apply_failed'} ({r.get('_elapsed_s')}s)"
        )

        rec["verdict"] = (
            "usable"
            if (empty_ok and rec["scenarios"]["golden"]["ok"] and garbage_ok)
            else "unusable"
        )
        print(f"    → 判定: {rec['verdict'].upper()}")
    finally:
        try:
            ags.stop_instance(instance_id)
        except Exception as e:
            print(f"    ⚠️ 实例回收失败（请手动检查 {instance_id}）：{e}")
    return rec


def main() -> int:
    load_env()
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=[], help="指定 task_id，可重复")
    ap.add_argument("--all", action="store_true", help="验证 tasks.jsonl 全部题目")
    ap.add_argument("--jobs", type=int, default=1, help="并发题数")
    ap.add_argument("--tasks-file", default=str(ROOT / "data" / "tasks.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "criteria_check.json"))
    args = ap.parse_args()

    registry = os.environ.get("TCR_REGISTRY", "")
    namespace = os.environ.get("TCR_NAMESPACE", "")
    tool_name = os.environ.get("AGS_TOOL_NAME") or os.environ.get("SWE_SYNTH_SHARED_TOOL", "")
    if not (registry and namespace and tool_name):
        sys.exit("[x] 需要 .env 里的 TCR_REGISTRY / TCR_NAMESPACE / AGS_TOOL_NAME")

    from clients.ags import AGSClient

    tool = AGSClient().find_tool(tool_name)
    if not tool:
        sys.exit(f"[x] 找不到沙箱工具 {tool_name}")
    tool_id = tool["tool_id"]

    tasks = [
        json.loads(l)
        for l in Path(args.tasks_file).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if args.task:
        want = set(args.task)
        tasks = [t for t in tasks if t["task_id"] in want]
    elif not args.all:
        tasks = tasks[:1]
    if not tasks:
        sys.exit("[x] 没有匹配的题目")

    print(f"待验证 {len(tasks)} 题，并发 {args.jobs}")
    results: list[dict] = []
    if args.jobs <= 1:
        for t in tasks:
            results.append(verify_task(t, registry, namespace, tool_id))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(verify_task, t, registry, namespace, tool_id): t["task_id"]
                for t in tasks
            }
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append(
                        {"task_id": futs[f], "verdict": "exception", "error": str(e)[:400]}
                    )

    usable = [r for r in results if r.get("verdict") == "usable"]
    print(f"\n{'=' * 70}\n可用 {len(usable)}/{len(results)} 题")
    for r in results:
        if r.get("verdict") != "usable":
            sc = r.get("scenarios", {})
            detail = ", ".join(
                f"{k}={'ok' if v.get('ok') else v.get('stage', v.get('error'))}"
                for k, v in sc.items()
            )
            print(f"  ✗ {r['task_id']}: {r.get('verdict')} {detail} {r.get('error', '')[:150]}")

    Path(args.out).write_text(
        json.dumps({r["task_id"]: r for r in results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {args.out}")
    return 0 if len(usable) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
