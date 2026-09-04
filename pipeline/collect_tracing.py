#!/usr/bin/env python3
"""tracing 批量采集：SandBox 执行 → COS → TKE 训练侧
=====================================================

课题验收第 2、3 条的产出脚本。对每道题跑 N 次 ReAct rollout，产出：

```
data/tracing/<task_id>/rollout_<k>.jsonl      每步操作 + 观察 + 最终 reward
data/tracing/dataset.jsonl                    汇总（对齐 VERL rollout 格式）
cos://<bucket>/tracing/<run_id>/...           上传副本（SandBox → TKE 传递通道）
```

## 单沙箱多 rollout

同一题的 N 次 rollout **复用同一个沙箱实例**，每次开始前 `git reset --hard`
还原仓库。理由：

- 沙箱启动约 11s，N=4 时省下 33s/题
- 镜像预热只需一次（首次可能等数分钟）

风险是状态残留。`reset_repo` 用 `git reset --hard` + `git clean -fd`，
并**刻意保留 `*.so`/`*.pyc`**（不加 `-x`）—— 否则 C 扩展类 repo
（matplotlib/sklearn）每次都要重新编译，单题从几十秒涨到几分钟。

## 为什么 rollout 之间必须有随机性

GRPO 的 advantage 来自**组内相对比较**：同一题 N 次采样，好的比坏的得分高。
若 N 次结果完全一样，组内方差为 0 → advantage 恒为 0 → **梯度为 0，白跑**。
这正是上一轮"14/31 步 score 全 0"的另一半原因。

因此 `temperature` 必须 > 0（默认 0.8）。采集完会**显式报告每题的组内方差**，
方差为 0 的题会被标记 —— 这类题对训练无贡献，应从训练集剔除或提高温度重采。

## reward 由谁算

统一走 `pipeline/sandbox_eval.run_eval`（训练侧口径），**不采用** Agent 自己
`run_tests` 的结论。Agent 只跑 F2P（快），最终判分要跑完整 F2P+P2P 并检查回归。
两者若不一致，以判分为准。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AGENT_SRC = ROOT / "sandbox_agent" / "react_agent.py"
OUT_DIR = ROOT / "data" / "tracing"


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    for n in ("e2b", "e2b.api.client_sync", "httpx", "httpcore"):
        logging.getLogger(n).setLevel(logging.WARNING)


def collect_task(
    task: dict,
    *,
    n_rollouts: int,
    max_steps: int,
    temperature: float,
    base_url: str,
    model: str,
    tool_id: str,
    registry: str,
    namespace: str,
    strict_eval: bool = False,
) -> dict:
    """对一道题采集 n_rollouts 条 tracing。返回该题的汇总记录。"""
    from clients.ags import AGSClient
    from clients.sandbox import connect_with_retry, start_instance_with_warmup

    from experiments.verify_criteria import tcr_image
    from pipeline.official_spec import make_eval_script
    from pipeline.sandbox_eval import run_eval, sbx_run, sbx_write

    task_id = task["task_id"]
    image = tcr_image(task_id, registry, namespace)
    rec: dict[str, Any] = {"task_id": task_id, "rollouts": []}
    buf = [f"=== {task_id}"]

    ags = AGSClient()
    t0 = time.time()
    try:
        iid, _ = start_instance_with_warmup(
            ags, tool_id, image, cpu="2", memory="4Gi", verbose=False
        )
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"沙箱启动失败：{str(e)[:300]}"
        print("\n" + "\n".join(buf + [f"    ✗ {rec['error']}"]), flush=True)
        return rec
    buf.append(f"    沙箱 {iid} 启动 {time.time() - t0:.1f}s")

    task_dir = OUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
        sbx = connect_with_retry(iid)
        sbx_run(sbx, "mkdir -p /task", 60)

        # spec 刻意**不含 golden_patch** —— Agent 绝不能看到答案，否则采集到的
        # tracing 毫无训练价值（模型会直接抄，学不到定位与修复的过程）
        spec = {
            "task_id": task_id,
            "repo": task["repo"],
            "problem_statement": task.get("problem_statement") or task.get("issue", ""),
            "test_script_path": "/task/agent_test.sh",
        }
        sbx_write(sbx, "/task/react_agent.py", AGENT_SRC.read_text(encoding="utf-8"))
        sbx_write(sbx, "/task/spec.json", json.dumps(spec, ensure_ascii=False))
        sbx_write(sbx, "/task/agent_test.sh", make_eval_script(task))

        for k in range(n_rollouts):
            t1 = time.time()
            # 每条 rollout 前把仓库还原，避免上一条的改动残留
            from pipeline.sandbox_eval import reset_repo

            reset_repo(sbx, task["base_commit"])

            code, out = sbx_run(
                sbx,
                f"/usr/bin/python3 /task/react_agent.py --spec /task/spec.json "
                f"--base-url {base_url} --model {model} --max-steps {max_steps} "
                f"--temperature {temperature} --rollout-id {k} "
                f"--tracing /task/tracing_{k}.jsonl 2>&1 | tail -5",
                timeout=2400,
            )
            _, tr = sbx_run(sbx, f"cat /task/tracing_{k}.jsonl 2>/dev/null", 120)
            lines = [json.loads(l) for l in tr.splitlines() if l.strip()]
            summary = next((s for s in lines if s.get("type") == "summary"), {})
            patch = summary.get("patch") or ""

            # 判分：走训练侧统一口径，不信 Agent 自己的 run_tests 结论
            ev = run_eval(sbx, task, patch, timeout=1800, strict=strict_eval)

            steps = [s for s in lines if s.get("type") == "step"]
            r = {
                "rollout_id": k,
                "steps_used": summary.get("steps_used"),
                "tool_seq": [s.get("tool") for s in steps],
                "edits_applied": summary.get("edits_applied"),
                "parse_errors": summary.get("parse_errors"),
                "finished_by_agent": summary.get("finished_by_agent"),
                "patch_chars": len(patch),
                "reward": ev.reward.reward,
                "stage": ev.stage.value,
                "resolved": ev.reward.strict_pass,
                "f2p": f"{ev.outcome.f2p_passed}/{ev.outcome.f2p_total}" if ev.outcome else "-",
                "p2p": f"{ev.outcome.p2p_passed}/{ev.outcome.p2p_total}" if ev.outcome else "-",
                "wall_s": round(time.time() - t1, 1),
            }
            rec["rollouts"].append(r)

            # 落盘：完整 tracing（含每步 observation）+ 判分结果
            (task_dir / f"rollout_{k}.jsonl").write_text(
                tr + json.dumps({"type": "eval", **ev.to_dict()}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            buf.append(
                f"    rollout {k}: steps={r['steps_used']} edits={r['edits_applied']} "
                f"F2P={r['f2p']} reward={r['reward']:.3f} "
                f"resolved={r['resolved']} ({r['wall_s']}s)"
            )

        rewards = [r["reward"] for r in rec["rollouts"]]
        rec["reward_mean"] = round(statistics.mean(rewards), 4) if rewards else 0.0
        # 组内标准差是 GRPO 能否学到东西的**先决条件**：为 0 则 advantage 全 0
        rec["reward_std"] = round(statistics.pstdev(rewards), 4) if len(rewards) > 1 else 0.0
        rec["pass_count"] = sum(1 for r in rec["rollouts"] if r["resolved"])
        rec["has_variance"] = rec["reward_std"] > 1e-6
        buf.append(
            f"    → mean={rec['reward_mean']} std={rec['reward_std']} "
            f"resolved={rec['pass_count']}/{len(rewards)} "
            f"{'✓有梯度' if rec['has_variance'] else '✗组内无差异(对GRPO无贡献)'}"
        )
        print("\n" + "\n".join(buf), flush=True)
        return rec
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print("\n" + "\n".join(buf + [f"    ✗ {rec['error']}"]), flush=True)
        return rec
    finally:
        try:
            ags.stop_instance(iid)
        except Exception:  # noqa: BLE001,S110
            pass


def resolve_base_url(cli_value: str) -> str:
    """确定**沙箱侧**访问推理服务的地址。

    ⚠️ 这里有个必须区分的陷阱：`127.0.0.1:8000` 对**编排进程**有效
    （vLLM 就在同机），但对**沙箱容器**无效 —— 沙箱的 localhost 是它自己。
    实测在节点上把 base_url 传成 127.0.0.1 时，沙箱侧探测直接失败。

    因此优先级：显式参数 > 环境变量 > 节点内网 IP（自动探测）。
    并且**显式拒绝** localhost 形式：宁可报错也不要跑一整轮才发现全失败。
    """
    v = (cli_value or os.environ.get("VLLM_BASE_URL") or "").strip()
    if v and not any(h in v for h in ("127.0.0.1", "localhost", "0.0.0.0")):
        return v

    # 自动探测本机内网 IP —— 沙箱与节点同 VPC，可直连
    import socket

    ip = ""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.0.0.1", 53))  # 不实际发包，只为让内核选出出口网卡地址
        ip = s.getsockname()[0]
        s.close()
    except Exception:  # noqa: BLE001
        pass
    if not ip:
        raise SystemExit(
            "无法确定沙箱可访问的推理地址。请显式传 --base-url http://<节点内网IP>:8000"
        )
    port = "8000"
    if v:
        # 从被拒的 localhost 形式里保留端口
        m = re.search(r":(\d+)", v)
        if m:
            port = m.group(1)
        print(f"  ⚠️ --base-url 含 localhost，沙箱无法访问，已改用内网地址 {ip}:{port}")
    return f"http://{ip}:{port}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "eval", "all"])
    ap.add_argument("--task", action="append", help="只采集指定题目，可重复")
    ap.add_argument("-n", "--rollouts", type=int, default=4, help="每题 rollout 数")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--jobs", type=int, default=4, help="并发题目数")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="swe-rl-policy")
    ap.add_argument("--strict-eval", action="store_true", help="评测口径（不给格式分）")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--no-cos", action="store_true", help="跳过 COS 上传")
    args = ap.parse_args()

    load_env()
    base_url = resolve_base_url(args.base_url)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")

    from clients.ags import AGSClient

    tool = AGSClient().find_tool(os.environ.get("AGS_TOOL_NAME", "swe-rl-vpc-runner"))
    if not tool:
        print("✗ 找不到沙箱工具", file=sys.stderr)
        return 2

    tasks = {
        json.loads(l)["task_id"]: json.loads(l)
        for l in open(ROOT / "data/tasks.jsonl")
        if l.strip()
    }
    # 只采集判据验证通过的题 —— 未通过的题 reward 无意义，采了也是噪声
    usable: set[str] = set()
    for p in ("criteria_check.json", "criteria_retry.json", "criteria_sphinx.json"):
        f = ROOT / "data" / p
        if f.is_file():
            usable |= set(json.loads(f.read_text()).get("usable", []))

    if args.task:
        want = [t for t in args.task if t in tasks]
    else:
        split = json.loads((ROOT / "data/split.json").read_text())
        want = (
            list(tasks)
            if args.split == "all"
            else list(split[args.split])
        )
    want = [t for t in want if t in usable]
    if not want:
        print("✗ 没有可采集的题目（检查判据验证结果）", file=sys.stderr)
        return 2

    print(
        f"采集 {len(want)} 题 × {args.rollouts} rollout（并发 {args.jobs}，"
        f"temp={args.temperature}，最多 {args.max_steps} 步）\n"
        f"推理服务: {base_url}\nrun_id: {run_id}",
        flush=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results: list[dict] = []
    kw = dict(
        n_rollouts=args.rollouts, max_steps=args.max_steps,
        temperature=args.temperature, base_url=base_url, model=args.model,
        tool_id=tool["tool_id"], registry=os.environ["TCR_REGISTRY"],
        namespace=os.environ["TCR_NAMESPACE"], strict_eval=args.strict_eval,
    )
    with ThreadPoolExecutor(max(1, args.jobs)) as ex:
        futs = [ex.submit(collect_task, tasks[t], **kw) for t in want]
        for f in as_completed(futs):
            results.append(f.result())

    # ---- 汇总 ----
    ok = [r for r in results if r.get("rollouts")]
    all_rw = [x["reward"] for r in ok for x in r["rollouts"]]
    n_resolved = sum(r.get("pass_count", 0) for r in ok)
    with_var = [r for r in ok if r.get("has_variance")]

    print(f"\n{'=' * 70}")
    print(f"采集完成 {len(ok)}/{len(results)} 题，耗时 {time.time() - t0:.0f}s")
    print(f"  rollout 总数     : {len(all_rw)}")
    print(f"  平均 reward      : {statistics.mean(all_rw):.4f}" if all_rw else "  无数据")
    print(f"  resolved 次数    : {n_resolved}/{len(all_rw)}"
          f"（pass@1 ≈ {n_resolved / len(all_rw):.1%}）" if all_rw else "")
    print(f"  组内有方差的题   : {len(with_var)}/{len(ok)}  ← GRPO 只能从这些题学到东西")
    for r in ok:
        if not r.get("has_variance"):
            print(f"    ✗ {r['task_id']}: std=0，mean={r.get('reward_mean')}（建议提温或剔除）")

    summary_path = OUT_DIR / f"summary_{run_id}.json"
    summary_path.write_text(
        json.dumps(
            {"run_id": run_id, "base_url": base_url, "model": args.model,
             "temperature": args.temperature, "n_rollouts": args.rollouts,
             "max_steps": args.max_steps, "results": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n汇总写入 {summary_path}")

    # ---- 上传 COS（课题指定的 SandBox → TKE 传递通道）----
    if not args.no_cos and os.environ.get("COS_BUCKET"):
        from clients import cos

        bucket = os.environ["COS_BUCKET"]
        n_up = 0
        for f in sorted(OUT_DIR.rglob("*.jsonl")):
            key = f"tracing/{run_id}/{f.relative_to(OUT_DIR).as_posix()}"
            try:
                cos.upload_file(bucket, key, str(f))
                n_up += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ 上传失败 {f.name}: {str(e)[:150]}")
        try:
            cos.upload_file(bucket, f"tracing/{run_id}/summary.json", str(summary_path))
            n_up += 1
        except Exception:  # noqa: BLE001,S110
            pass
        print(f"已上传 {n_up} 个文件到 cos://{bucket}/tracing/{run_id}/")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
