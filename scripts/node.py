#!/usr/bin/env python3
"""
GPU 节点远程监督通道（本地 CodeBuddy 终端 → 云上 GPU 宿主机）
==============================================================

用腾讯云 **云助手 TAT**（节点上已装 Agent，Online）在 GPU 宿主机上执行命令并取回输出。
相比 `kubectl exec` 进 pod：
  · 不需要集群公网端点、不需要 kubeconfig、不需要给节点开任何入站端口
  · 命令下发走腾讯云 API（本机只需出网 443），节点侧零暴露
  · 监督对象是**宿主机**（nvidia-smi / 磁盘 / 容器 / 训练日志），不是 pod 内部视角

用法（在项目根目录执行）：
    python3 scripts/node.py info                       # 节点概况：GPU/显存/磁盘/内存/容器
    python3 scripts/node.py run 'nvidia-smi'           # 在宿主机跑任意命令
    python3 scripts/node.py run -f local_script.sh     # 把本地脚本整段送上去执行
    python3 scripts/node.py tail /root/train.log -n 80 # 看日志尾部
    python3 scripts/node.py watch /root/train.log      # 持续跟踪（轮询增量，Ctrl-C 退出）
    python3 scripts/node.py cexec swe-rl-gpu 'ls /workspace'   # 在节点上的容器里执行
    python3 scripts/node.py inject-key                 # 注入本机 SSH 公钥（免重启，为后续 ssh 铺路）

安全约定：
  · 凭证只从环境变量 / .env 读取，绝不写入代码或日志
  · 下发内容整段 base64 传输，不在本地拼 shell 字符串，避免注入与转义事故
  · 全部命令以 root 在宿主机执行 —— 这是运维通道，请只用于本项目节点
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TERMINAL_STATES = {"SUCCESS", "FAILED", "TIMEOUT", "START_FAILED", "CANCELLED"}


# ------------------------------------------------------------------ 基础设施

def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
    except ImportError:  # 没装 python-dotenv 时的极简回退
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def need(key: str, default: str = "") -> str:
    val = (os.environ.get(key) or default).strip()
    if not val:
        sys.exit(f"[x] 缺少环境变量 {key}（请写入 .env）")
    return val


def tat_client():
    from tencentcloud.common import credential
    from tencentcloud.tat.v20201028 import tat_client as tc

    cred = credential.Credential(
        need("TENCENTCLOUD_SECRET_ID"), need("TENCENTCLOUD_SECRET_KEY")
    )
    return tc.TatClient(cred, need("GPU_NODE_REGION", "ap-beijing"))


def instance_id() -> str:
    return need("GPU_NODE_INSTANCE_ID")


# ------------------------------------------------------------------ 核心：下发命令

def remote_exec(script: str, timeout: int = 600, quiet: bool = False) -> tuple[int, str]:
    """在 GPU 宿主机上执行一段 shell，返回 (exit_code, output)。"""
    from tencentcloud.tat.v20201028 import models

    client = tat_client()
    req = models.RunCommandRequest()
    req.from_json_string(
        json.dumps(
            {
                "Content": base64.b64encode(script.encode()).decode(),
                "InstanceIds": [instance_id()],
                "CommandType": "SHELL",
                "Username": "root",
                "Timeout": timeout,
                "WorkingDirectory": os.environ.get("GPU_NODE_WORKDIR", "/root"),
                "CommandName": "swe-rl-supervise",
                "SaveCommandContent": False,
            }
        )
    )
    inv = client.RunCommand(req).InvocationId

    deadline = time.time() + timeout + 60
    while time.time() < deadline:
        time.sleep(2)
        q = models.DescribeInvocationTasksRequest()
        q.from_json_string(json.dumps({"InvocationIds": [inv], "HideOutput": False}))
        task = client.DescribeInvocationTasks(q).InvocationTaskSet[0]
        if task.TaskStatus not in TERMINAL_STATES:
            continue
        result = task.TaskResult
        out = ""
        if getattr(result, "Output", None):
            out = base64.b64decode(result.Output).decode("utf-8", errors="replace")
        if getattr(result, "OutputUploadCOSErrorInfo", None) and not out:
            out = f"[输出过大，已上传 COS] {result.OutputUrl}"
        code = result.ExitCode if result.ExitCode is not None else -1
        if not quiet and task.TaskStatus != "SUCCESS":
            print(f"[!] TaskStatus={task.TaskStatus} exit={code}", file=sys.stderr)
        return code, out
    sys.exit(f"[x] 等待超时，invocation={inv}（可去控制台云助手查看）")


# ------------------------------------------------------------------ 子命令

def cmd_info(_args) -> int:
    script = r"""
echo "=== host ==="; hostname; uptime | tr -s ' '
echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
echo "=== gpu procs ==="; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "=== mem/disk ==="; free -g | head -2; df -h / | tail -1
echo "=== containers (非 kube-system) ==="; crictl ps -o table 2>/dev/null | grep -v -E 'csi-|coredns|kube-proxy|ip-masq|nvidia-|tke-cni|snapshot-' | cut -c1-160
"""
    code, out = remote_exec(script, timeout=120)
    print(out)
    return code


def cmd_run(args) -> int:
    script = Path(args.script).read_text(encoding="utf-8") if args.script else args.command
    if not script:
        sys.exit("[x] 需要 <command> 或 -f <script>")
    code, out = remote_exec(script, timeout=args.timeout)
    print(out, end="" if out.endswith("\n") else "\n")
    return code


def cmd_put(args) -> int:
    """把本地文件投放到节点（内容走 base64，避免 SCP/公网入站需求）。

    ⚠️ 云助手 `RunCommand` 的 `Content` 字段上限为 **65535 字节**（实测报
    `InvalidParameterValue.TooLong`），且内容本身还要 base64 膨胀 4/3 倍。
    因此大文件按块切分、逐块追加，最后用 md5 校验整体一致性。

    仍不适合投放模型权重/镜像这类大件（那些走 ModelScope / TCR / COS）；
    本命令定位是代码与数据文件（几十 KB ~ 数 MB）。
    """
    import hashlib

    src = Path(args.local).expanduser()
    if not src.is_file():
        sys.exit(f"[x] 本地文件不存在：{src}")
    raw = src.read_bytes()
    remote = args.remote or f"/data/swe-rl/{src.name}"
    local_md5 = hashlib.md5(raw).hexdigest()

    # 每块原始字节数：base64 后膨胀 4/3，外层还要包 heredoc；云助手 Content 上限
    # 65535 字节，取 24KB 原始数据 → 约 32KB base64，留足余量
    chunk = 24 * 1024
    blocks = [raw[i : i + chunk] for i in range(0, len(raw), chunk)] or [b""]
    rq = json.dumps(remote)

    for idx, blk in enumerate(blocks):
        # base64 内容经 heredoc 送入，避免作为命令行参数触发 shell 参数长度上限
        # （曾用 printf '%s' <base64> 导致 Content 超限报 TooLong）
        payload = base64.b64encode(blk).decode()
        redirect = ">" if idx == 0 else ">>"
        script = (
            f"mkdir -p $(dirname {rq}) && "
            f"base64 -d {redirect} {rq} <<'__B64_EOF__'\n{payload}\n__B64_EOF__"
        )
        code, out = remote_exec(script, timeout=180, quiet=True)
        if code != 0:
            print(f"[x] 第 {idx + 1}/{len(blocks)} 块写入失败：{out[:300]}", file=sys.stderr)
            return 1
        if len(blocks) > 1:
            print(f"  投放中 {idx + 1}/{len(blocks)} 块", end="\r", file=sys.stderr)

    code, out = remote_exec(f"wc -c < {rq}; md5sum {rq}", timeout=120, quiet=True)
    if local_md5 in out:
        print(f"[ok] 已投放 {remote}（{len(raw)} 字节，{len(blocks)} 块，md5 一致）")
        return 0
    print(f"[!] md5 校验失败：本地={local_md5}\n远端输出：{out[:300]}", file=sys.stderr)
    return 1


def cmd_nohup(args) -> int:
    """在节点后台启动长任务，日志落宿主机文件，立即返回。

    训练/下载这类跑几十分钟的任务必须这样起 —— 云助手单次调用有超时上限，
    前台跑会被截断。之后用 `node.py watch <log>` 跟踪。

    ⚠️ 关键：Python 输出重定向到文件时默认是**全缓冲**（4~8KB 才落盘），
    会导致 `watch` 长时间看不到任何输出、误判任务卡死。
    因此统一注入 `PYTHONUNBUFFERED=1`，并优先用 `stdbuf` 关掉 libc 层缓冲。
    """
    log = args.log
    prefix = "PYTHONUNBUFFERED=1 "
    cmd = args.command
    if not args.no_unbuffer:
        # stdbuf 不一定存在（busybox 等），存在才用，避免 command not found
        cmd = f"if command -v stdbuf >/dev/null 2>&1; then stdbuf -oL -eL {cmd}; else {cmd}; fi"
    script = (
        f"mkdir -p $(dirname {json.dumps(log)}) && cd {json.dumps(args.workdir)} && "
        f"{prefix}nohup sh -c {json.dumps(cmd)} > {json.dumps(log)} 2>&1 & "
        f"echo started pid=$!; sleep 3; echo '--- 日志前几行 ---'; "
        f"head -5 {json.dumps(log)} 2>/dev/null"
    )
    code, out = remote_exec(script, timeout=120)
    print(out.strip())
    print(f"\n[提示] 跟踪日志：python3 scripts/node.py watch {log}")
    return code


def cmd_tail(args) -> int:
    script = f"tail -n {int(args.lines)} -- {json.dumps(args.path)}"
    code, out = remote_exec(script, timeout=120)
    print(out, end="")
    return code


def cmd_watch(args) -> int:
    """轮询式跟踪：每轮取从上次字节偏移开始的增量，近似 tail -f。"""
    offset = 0
    print(f"[watch] {args.path} 每 {args.interval}s 拉一次增量，Ctrl-C 退出", file=sys.stderr)
    try:
        while True:
            script = (
                f"f={json.dumps(args.path)}; "
                f"[ -f \"$f\" ] || {{ echo '__MISSING__'; exit 0; }}; "
                f"sz=$(stat -c %s \"$f\"); echo \"__SIZE__ $sz\"; "
                f"if [ \"$sz\" -gt {offset} ]; then tail -c +{offset + 1} \"$f\"; fi"
            )
            _, out = remote_exec(script, timeout=120, quiet=True)
            lines = out.split("\n")
            body: list[str] = []
            for ln in lines:
                if ln.startswith("__SIZE__ "):
                    offset = int(ln.split()[1])
                elif ln == "__MISSING__":
                    print("[watch] 文件还不存在，继续等…", file=sys.stderr)
                else:
                    body.append(ln)
            text = "\n".join(body).strip("\n")
            if text:
                print(text, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[watch] 已停止", file=sys.stderr)
        return 0


def cmd_cexec(args) -> int:
    """在节点上的容器内执行命令；container 参数可传容器名，也可传 pod 名。

    内层命令再做一次 base64，避免「本地 shell → TAT → sh -lc → crictl exec」
    多层转义把换行/引号吃掉（曾导致 python -c 的 \\n 变成字面量而报 SyntaxError）。
    """
    inner = " ".join(args.command) if isinstance(args.command, list) else args.command
    b64 = base64.b64encode(inner.encode()).decode()
    name = json.dumps(args.container)
    script = (
        f"cid=$(crictl ps -q --name {name} | head -1); "
        f'if [ -z "$cid" ]; then pid=$(crictl pods -q --name {name} | head -1); '
        f'[ -n "$pid" ] && cid=$(crictl ps -q --pod $pid | head -1); fi; '
        f'[ -n "$cid" ] || {{ echo "no running container/pod matched: {args.container}"; exit 1; }}; '
        f"crictl exec $cid sh -c \"echo {b64} | base64 -d | sh\""
    )
    code, out = remote_exec(script, timeout=args.timeout)
    print(out, end="")
    return code


def cmd_inject_key(args) -> int:
    pub = Path(args.pubkey).expanduser()
    if not pub.exists():
        sys.exit(f"[x] 找不到公钥 {pub}（可先 ssh-keygen -t ed25519 -f ~/.ssh/swe_rl_gpu）")
    key = pub.read_text(encoding="utf-8").strip().splitlines()[0]
    if not key.startswith(("ssh-", "ecdsa-")):
        sys.exit("[x] 这看起来不是公钥文件")
    script = (
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && "
        f"grep -qxF {json.dumps(key)} /root/.ssh/authorized_keys || echo {json.dumps(key)} >> /root/.ssh/authorized_keys; "
        "chmod 600 /root/.ssh/authorized_keys; wc -l < /root/.ssh/authorized_keys"
    )
    code, out = remote_exec(script, timeout=60)
    print(f"[ok] authorized_keys 现有 {out.strip()} 行（免重启生效）")
    return code


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description="GPU 节点远程监督（云助手 TAT 通道）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="节点概况：GPU/显存/进程/磁盘/容器").set_defaults(func=cmd_info)

    p = sub.add_parser("run", help="在宿主机执行命令")
    p.add_argument("command", nargs="?", default="")
    p.add_argument("-f", "--script", help="本地脚本文件，整段上传执行")
    p.add_argument("--timeout", type=int, default=600)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("tail", help="看远端文件尾部")
    p.add_argument("path")
    p.add_argument("-n", "--lines", default=50)
    p.set_defaults(func=cmd_tail)

    p = sub.add_parser("put", help="投放本地文件到节点（≤512KB，走 base64）")
    p.add_argument("local")
    p.add_argument("remote", nargs="?", default="")
    p.set_defaults(func=cmd_put)

    p = sub.add_parser("nohup", help="在节点后台起长任务，日志落文件后立即返回")
    p.add_argument("command")
    p.add_argument("--log", required=True, help="宿主机日志路径")
    p.add_argument("--workdir", default="/data/swe-rl")
    p.add_argument(
        "--no-unbuffer",
        action="store_true",
        help="不注入行缓冲（默认注入，否则日志要攒够几 KB 才落盘，看不到进度）",
    )
    p.set_defaults(func=cmd_nohup)

    p = sub.add_parser("watch", help="轮询跟踪远端日志（近似 tail -f）")
    p.add_argument("path")
    p.add_argument("--interval", type=int, default=10)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("cexec", help="在节点上的容器内执行命令")
    p.add_argument("container")
    p.add_argument("command", nargs="+")
    p.add_argument("--timeout", type=int, default=600)
    p.set_defaults(func=cmd_cexec)

    p = sub.add_parser("inject-key", help="注入本机 SSH 公钥到节点 root（免重启）")
    p.add_argument("--pubkey", default="~/.ssh/swe_rl_gpu.pub")
    p.set_defaults(func=cmd_inject_key)

    args = ap.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
