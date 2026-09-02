#!/usr/bin/env python3
"""
验证线 A 通路：VPC 沙箱 → NodePort → TKE Pod 内的服务
====================================================

这是课题「Agent 在沙箱内解题，推理由 TKE GPU 提供」的通路验证。

链路：
    沙箱容器（VPC 内）
      → 节点内网 IP:30800（NodePort）
      → Service swe-rl-llm
      → Pod swe-rl-train:8000

全程不出 VPC —— 对比第一轮用 `Service type=LoadBalancer` 走公网 CLB。

做法：在 Pod 内起一个极简 HTTP 服务（只依赖标准库），
然后从沙箱访问 `http://<节点内网IP>:30800/`，看能否拿到响应。

用法：
    python3 experiments/verify_linea_path.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
NODE = str(ROOT / "scripts" / "node.py")

PROBE_TOKEN = "SWE_RL_LINEA_OK"
NODE_PORT = 30800


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)


def node_run(cmd: str, timeout: int = 300) -> str:
    p = subprocess.run(
        [sys.executable, NODE, "run", cmd],
        capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
    )
    return "\n".join(
        ln for ln in (p.stdout or "").splitlines()
        if "NotOpenSSL" not in ln and "warnings.warn" not in ln
    )


def sbx_run(sbx, cmd: str, timeout: int = 60) -> tuple[int, str]:
    try:
        r = sbx.commands.run(cmd, user="root", timeout=timeout)
        return (r.exit_code or 0), (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        code = getattr(e, "exit_code", None)
        out = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
        return (code if code is not None else 1), (out or str(e))


def main() -> int:
    load_env()
    print("=" * 72)
    print(" 线 A 通路验证：VPC 沙箱 → NodePort → TKE Pod")
    print("=" * 72)

    # ---- 1) 在 Pod 内起探针服务（只用标准库，不装任何东西）----
    print("\n[1/3] 在 TKE Pod 内启动探针服务（:8000）…")
    # 探针脚本走挂载盘（/data/swe-rl 已 hostPath 挂进 Pod），
    # 不用 heredoc —— 「本地 shell → 云助手 → kubectl exec → bash -lc」四层转义
    # 极易出错，且 pkill 的模式会匹配到命令自身导致 exit 143（SIGTERM 自杀）。
    start = (
        "export KUBECONFIG=/data/swe-rl/.kube/config; "
        "kubectl exec swe-rl-train -- "
        "sh -c 'nohup python3 /data/swe-rl/experiments/linea_probe_server.py "
        ">/tmp/probe.log 2>&1 & sleep 3; "
        "curl -s -m 5 http://127.0.0.1:8000/ || cat /tmp/probe.log'"
    )
    out = node_run(start)
    pod_ok = PROBE_TOKEN in out
    print(f"      Pod 内自测: {'✓' if pod_ok else '✗ ' + out[-200:]}")
    if not pod_ok:
        return 1

    # ---- 2) 从节点访问 NodePort（验证 Service 转发）----
    print(f"\n[2/3] 从节点访问 NodePort :{NODE_PORT}…")
    node_ip = node_run(
        "hostname -I | tr ' ' '\\n' | grep -E '^10\\.' | head -1"
    ).strip().splitlines()[-1].strip()
    out = node_run(f"curl -s -m 8 http://{node_ip}:{NODE_PORT}/ || echo NODEPORT_FAIL")
    np_ok = PROBE_TOKEN in out
    print(f"      NodePort 转发: {'✓' if np_ok else '✗ ' + out[-200:]}")

    # ---- 3) 从沙箱访问（决定性验证）----
    print("\n[3/3] 从 VPC 沙箱访问节点 NodePort（决定性验证）…")
    from clients.ags import AGSClient
    from clients.sandbox import start_instance_with_warmup
    from e2b_code_interpreter import Sandbox

    ags = AGSClient()
    tool = ags.find_tool(os.environ["AGS_TOOL_NAME"])
    if not tool:
        print("      ✗ 找不到沙箱工具")
        return 1

    ns = f"{os.environ['TCR_REGISTRY']}/{os.environ['TCR_NAMESPACE']}"
    first = json.loads((ROOT / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    image = f"{ns}/sweb-{first['task_id'].replace('__', '-').lower()}:sbx"

    inst = None
    sbx_ok = False
    try:
        inst, _ = start_instance_with_warmup(ags, tool["tool_id"], image, cpu="2", memory="4Gi")
        sbx = Sandbox.connect(inst)
        _, out = sbx_run(
            sbx, f"curl -s -m 10 http://{node_ip}:{NODE_PORT}/ || echo SBX_FAIL"
        )
        sbx_ok = PROBE_TOKEN in out
        print(f"      沙箱 → NodePort: {'✓' if sbx_ok else '✗'}  响应: {out.strip()[:90]}")
    finally:
        # 收尾：停探针 + 回收实例
        node_run(
            "export KUBECONFIG=/data/swe-rl/.kube/config; "
            "kubectl exec swe-rl-train -- "
            "sh -c \"kill \$(pgrep -f linea_probe_server) 2>/dev/null; exit 0\" || true"
        )
        if inst:
            try:
                ags.stop_instance(inst)
            except Exception as e:
                print(f"      ⚠️ 实例回收失败 {inst}: {e}")

    print("\n" + "-" * 72)
    print(f"  Pod 内服务      {'✓' if pod_ok else '✗'}")
    print(f"  NodePort 转发   {'✓' if np_ok else '✗'}")
    print(f"  沙箱内网直达    {'✓' if sbx_ok else '✗'}")
    print("-" * 72)
    if pod_ok and np_ok and sbx_ok:
        print("\n✓ 线 A 通路打通：沙箱经 VPC 内网访问 TKE Pod 服务，全程不出 VPC")
        return 0
    print("\n✗ 通路未打通。若沙箱一步失败，检查安全组是否放通节点 30800 端口（同 VPC 内）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
