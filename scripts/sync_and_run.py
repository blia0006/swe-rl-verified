#!/usr/bin/env python3
"""把项目代码同步到 GPU 节点，并在节点上托管长任务
==================================================

## 为什么必须有这个脚本

采集与训练动辄跑几小时。若编排进程留在本机，**合上笔记本或断网就全断**。
本脚本把代码同步到节点，再用 `setsid nohup` 起进程，让任务的生命周期
与本地终端彻底解耦 —— 本机断网只是看不到日志，任务照跑。

节点具备独立跑完全流程的条件（已实测）：
  · AGS API 可达（HTTP 200）→ 能创建/回收沙箱
  · 沙箱实例子域名可解析（`49999-<id>.ap-beijing.tencentags.com`）→ 能驱动沙箱
    ⚠️ 裸域名 `ap-beijing.tencentags.com` 解析为 0.0.0.1，测它会误判为"不通"
  · COS 可达 → tracing 能上传
  · vLLM 就在本机（`127.0.0.1:8000`），推理零网络开销

## 依赖环境

节点用独立 venv `/data/swe-rl/venv-orch`（py3.10）：
  · 不用 `pip install --target`：e2b 的 `pyqwest` 含原生扩展，
    --target 方式装完 import 会报 circular import
  · 系统缺 `ensurepip`，venv 需用 `get-pip.py` 引导（已完成）
  · py3.10 能直接 `import swebench`（本机 3.9 不行，只能 exec 单文件）

用法：
    python3 scripts/sync_and_run.py sync                    # 只同步代码
    python3 scripts/sync_and_run.py collect --split train   # 同步并启动采集
    python3 scripts/sync_and_run.py train                   # 同步并启动训练
    python3 scripts/sync_and_run.py status                  # 查看远程任务状态
    python3 scripts/sync_and_run.py tail <job>              # 跟踪日志
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE = str(ROOT / "scripts" / "node.py")
REMOTE = "/data/swe-rl"
PY = f"{REMOTE}/venv-orch/bin/python"

# 需要同步的代码（数据与日志不同步：tracing 由节点生成后上传 COS）
SYNC_FILES = [
    "pipeline/official_spec.py",
    "pipeline/sandbox_eval.py",
    "pipeline/reward.py",
    "pipeline/collect_tracing.py",
    "pipeline/verl_reward_fn.py",
    "pipeline/edit_format.py",
    "pipeline/build_grpo_dataset.py",
    "pipeline/extract_file_contents.py",
    "pipeline/select_tasks.py",
    "sandbox_agent/react_agent.py",
    "clients/ags.py",
    "clients/sandbox.py",
    "clients/cos.py",
    "experiments/verify_criteria.py",
    "experiments/verify_linea_agent.py",
    "experiments/inspect_tracing.py",
    "experiments/test_agent_tools.py",
    "scripts/orchestrate.py",
    "scripts/run_grpo_training.sh",
    "scripts/start_train_container.sh",
    "data/tasks.jsonl",
    "data/split.json",
    "data/criteria_check.json",
    "data/criteria_retry.json",
    "data/criteria_sphinx.json",
    ".env",
]


def node(args: list[str], timeout: int = 600) -> str:
    p = subprocess.run(
        [sys.executable, NODE, *args],
        capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
    )
    out = "\n".join(
        ln for ln in (p.stdout or "").splitlines()
        if "NotOpenSSL" not in ln and "warnings.warn" not in ln
    )
    return out + (("\n" + p.stderr) if p.returncode and p.stderr else "")


def do_sync() -> None:
    print("同步代码到节点…")
    n_ok = 0
    for rel in SYNC_FILES:
        src = ROOT / rel
        if not src.is_file():
            print(f"  – 跳过（不存在）：{rel}")
            continue
        out = node(["put", rel, f"{REMOTE}/{rel}"], timeout=300)
        if "md5 一致" in out or "已投放" in out:
            n_ok += 1
        else:
            print(f"  ✗ {rel}: {out.strip()[:200]}")
    print(f"  ✓ 已同步 {n_ok}/{len(SYNC_FILES)} 个文件")


def launch(job: str, cmd: str) -> None:
    """在节点上以完全脱离会话的方式启动长任务。

    `setsid` + `nohup` + 重定向三者缺一不可：
      · setsid   脱离控制终端，云助手任务结束时不会被连带杀掉
      · nohup    忽略 SIGHUP
      · 重定向   否则进程会因为 stdout 关闭而在写日志时崩掉
    额外写 pid 文件，便于 status/stop 查询。
    """
    log = f"{REMOTE}/logs/{job}.log"
    pidf = f"{REMOTE}/logs/{job}.pid"
    wrapped = (
        f"mkdir -p {REMOTE}/logs && cd {REMOTE} && "
        f"setsid nohup {cmd} > {log} 2>&1 < /dev/null & "
        f"echo $! > {pidf}; sleep 2; "
        f"echo \"pid=$(cat {pidf}) log={log}\"; "
        f"ps -p $(cat {pidf}) >/dev/null 2>&1 && echo '进程存活 ✓' || echo '进程已退出 ✗'"
    )
    print(node(["run", wrapped], timeout=180))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["sync", "collect", "train", "status", "tail", "stop"])
    ap.add_argument("--split", default="train")
    ap.add_argument("-n", "--rollouts", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--job", default="collect", help="status/tail/stop 的目标任务名")
    ap.add_argument("--lines", type=int, default=40)
    args = ap.parse_args()

    if args.action == "sync":
        do_sync()
        return 0

    if args.action == "status":
        print(node(["run",
            f"for f in {REMOTE}/logs/*.pid; do "
            f"  [ -f \"$f\" ] || continue; j=$(basename $f .pid); p=$(cat $f); "
            f"  if ps -p $p >/dev/null 2>&1; then st='运行中'; else st='已结束'; fi; "
            f"  echo \"$j  pid=$p  $st  日志 $(wc -l < {REMOTE}/logs/$j.log 2>/dev/null) 行\"; "
            f"done; echo; echo '--- vLLM ---'; "
            f"curl -s -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo '推理服务在线 ✓' || echo '推理服务离线 ✗'"
        ]))
        return 0

    if args.action == "tail":
        print(node(["run", f"tail -n {args.lines} {REMOTE}/logs/{args.job}.log"]))
        return 0

    if args.action == "stop":
        print(node(["run",
            f"p=$(cat {REMOTE}/logs/{args.job}.pid 2>/dev/null); "
            f"[ -n \"$p\" ] && kill -TERM -$p 2>/dev/null; kill -TERM $p 2>/dev/null; "
            f"sleep 2; ps -p $p >/dev/null 2>&1 && echo '仍在运行' || echo '已停止'"
        ]))
        return 0

    do_sync()

    if args.action == "collect":
        run_id = args.run_id or f"node-{args.split}"
        cmd = (
            f"{PY} pipeline/collect_tracing.py --split {args.split} "
            f"-n {args.rollouts} --max-steps {args.max_steps} "
            f"--temperature {args.temperature} --jobs {args.jobs} "
            f"--run-id {run_id} --base-url http://127.0.0.1:8000"
        )
        print(f"\n在节点启动采集：{run_id}")
        launch("collect", cmd)
    else:
        print("\n在节点启动训练")
        launch("train", f"bash {REMOTE}/scripts/run_grpo_training.sh")

    print("\n本机现在可以断网/关机，任务在节点独立运行。")
    print(f"恢复后查看：python3 scripts/sync_and_run.py status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
