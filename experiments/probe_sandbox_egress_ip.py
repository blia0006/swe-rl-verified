#!/usr/bin/env python3
"""
实测 AGS 沙箱的**出网出口 IP**（决定 NAT DNAT 方案的安全组白名单粒度）
=====================================================================

背景：线 A 要让沙箱内的 Agent 通过 HTTP 调用 GPU 节点上的推理服务。
GPU 节点无公网 IP，方案是复用其 VPC 内已有的 NAT 网关做 **DNAT 端口转发**
（`<NAT_EIP>:<PORT> → <GPU 内网 IP>:<PORT>`），而不给 GPU 节点绑 EIP。

要做最小化放通，必须先知道沙箱的出口 IP：
  · 若出口 IP 固定/少量 → 安全组只放通这几个 IP，暴露面最小
  · 若每次都变（大段随机） → 需退到"放通 0.0.0.0/0 + 强制 API Key + 用完即撤"，
    或改用反向隧道方案

同时验证沙箱能否访问该 NAT 的公网 EIP（连通性前置）。

用法：
    python3 experiments/probe_sandbox_egress_ip.py            # 起 1 个实例
    python3 experiments/probe_sandbox_egress_ip.py -n 3       # 起 3 个，看 IP 是否一致
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")


def probe_once(idx: int, nat_eip: str) -> dict:
    """起一个沙箱实例，查它的出口 IP，并测到 NAT EIP 的连通性。"""
    from clients.ags import AGSClient
    from e2b_code_interpreter import Sandbox

    ags = AGSClient()
    tool_name = os.environ.get("AGS_TOOL_NAME") or os.environ.get("SWE_SYNTH_SHARED_TOOL", "")
    if not tool_name:
        sys.exit("[x] 请在 .env 设置 AGS_TOOL_NAME（可复用的沙箱工具名）")
    tool = ags.find_tool(tool_name)
    if not tool:
        sys.exit(f"[x] 找不到沙箱工具 {tool_name}")

    t0 = time.time()
    instance_id, image = ags.start_instance(tool["tool_id"])
    boot = time.time() - t0
    print(f"[{idx}] 实例 {instance_id} 启动 {boot:.1f}s image={image or '<默认>'}")

    result: dict = {"instance_id": instance_id, "boot_s": round(boot, 2)}
    try:
        sbx = Sandbox.connect(instance_id)
        # 出口 IP：用两个独立来源交叉验证，避免单一站点缓存/代理造成误判
        script = (
            'echo "--- ip-a ---"; curl -s -m 15 https://api.ipify.org; echo; '
            'echo "--- ip-b ---"; curl -s -m 15 https://ifconfig.me/ip; echo'
        )
        if nat_eip:
            port = os.environ.get("LLM_PORT", "8000")
            script += (
                '; echo "--- reach-nat-eip ---"; '
                f'curl -s -m 8 -o /dev/null -w "%{{http_code}}" '
                f"http://{nat_eip}:{port}/health || echo conn_refused_or_timeout; echo"
            )
        try:
            res = sbx.commands.run(script, user="root", timeout=90)
            out = res.stdout or ""
        except Exception as e:  # e2b 在非 0 退出码时抛异常，需收敛（旧项目踩过的坑）
            out = getattr(e, "stdout", "") or str(e)

        ips: list[str] = []
        for line in out.splitlines():
            line = line.strip()
            if line and line.count(".") == 3 and all(
                p.isdigit() for p in line.split(".")
            ):
                ips.append(line)
        result["egress_ips"] = ips
        result["raw"] = out
        print(f"[{idx}] 出口 IP: {ips or '未解析到'}")
        print(f"[{idx}] 原始输出:\n{out}")
    finally:
        try:
            ags.stop_instance(instance_id)
            print(f"[{idx}] 实例已回收")
        except Exception as e:
            print(f"[{idx}] ⚠️ 回收失败（请手动检查）：{e}")
    return result


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=1, help="起几个实例交叉比对")
    ap.add_argument(
        "--nat-eip",
        default=os.environ.get("NAT_EIP", ""),
        help="NAT 网关 EIP（默认读 .env 的 NAT_EIP）；留空则跳过连通性探针",
    )
    args = ap.parse_args()

    if args.nat_eip:
        print(f"NAT EIP 探针目标已设置（服务未起时预期 refused/timeout）\n")
    else:
        print("未提供 NAT_EIP，跳过连通性探针，只测出口 IP\n")
    results = [probe_once(i + 1, args.nat_eip) for i in range(args.count)]

    all_ips = sorted({ip for r in results for ip in r.get("egress_ips", [])})
    print("\n" + "=" * 60)
    print(f"汇总：{len(results)} 个实例，出口 IP 集合 = {all_ips}")
    if len(all_ips) == 1:
        print("→ 出口 IP 固定，安全组可精确白名单该 IP（暴露面最小）✅")
    elif all_ips:
        print("→ 出口 IP 有多个，需放通对应网段；建议再多采样确认范围")
    else:
        print("→ 未取到出口 IP，需检查沙箱出网或换探测站点")
    return 0


if __name__ == "__main__":
    sys.exit(main())
