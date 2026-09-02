#!/usr/bin/env python3
"""
验证 VPC 沙箱与 GPU 节点内网直通（导师要求的验收标准）
====================================================

创建了 `NetworkMode=VPC` 的工具**不等于**内网真的通了。必须实测三件事：

| # | 验证项 | 判定 | 意义 |
|---|---|---|---|
| **1** | **TCP 连通 GPU 内网 IP** | **必须通过** | 内网链路真正打通的直接证据 |
| **2** | **无公网出口** | **必须通过** | 证明不再暴露公网（PUBLIC 模式下每次是随机公网 IP） |
| 3 | 沙箱自身 IP 段 | 仅供参考 | 沙箱是**容器**而非虚拟机，容器内看到的常是 `169.254.x.x` veth 地址，主网卡在宿主侧 —— 因此不能以此判定 VPC 是否生效 |

⚠️ 判定只依据第 1、2 项。曾因把第 3 项当硬性条件而误报"内网未打通"，
而实际 `TCP 连通 10.0.0.11:22` 已经成功、公网出口已消失。

用法：
    python3 experiments/verify_vpc_connectivity.py
"""

from __future__ import annotations

import ipaddress
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)


def need(k: str) -> str:
    v = (os.environ.get(k) or "").strip()
    if not v:
        sys.exit(f"[x] 缺少 {k}")
    return v


def gpu_net(region: str, iid: str) -> dict:
    from tencentcloud.common import credential
    from tencentcloud.cvm.v20170312 import cvm_client, models
    from tencentcloud.vpc.v20170312 import vpc_client, models as vm

    cred = credential.Credential(
        need("TENCENTCLOUD_SECRET_ID"), need("TENCENTCLOUD_SECRET_KEY")
    )
    c = cvm_client.CvmClient(cred, region)
    req = models.DescribeInstancesRequest()
    req.from_json_string(json.dumps({"InstanceIds": [iid]}))
    i = c.DescribeInstances(req).InstanceSet[0]
    v = i.VirtualPrivateCloud

    vc = vpc_client.VpcClient(cred, region)
    r2 = vm.DescribeSubnetsRequest()
    r2.from_json_string(json.dumps({"SubnetIds": [v.SubnetId]}))
    sn = vc.DescribeSubnets(r2).SubnetSet[0]

    return {
        "vpc_id": v.VpcId,
        "subnet_id": v.SubnetId,
        "cidr": sn.CidrBlock,
        "private_ip": list(i.PrivateIpAddresses or [])[0],
    }


def sbx_run(sbx, cmd: str, timeout: int = 120) -> tuple[int, str]:
    """e2b 在非 0 退出码时抛异常，统一收敛成返回值。"""
    try:
        r = sbx.commands.run(cmd, user="root", timeout=timeout)
        return (r.exit_code or 0), (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        code = getattr(e, "exit_code", None)
        out = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
        return (code if code is not None else 1), (out or str(e))


def main() -> int:
    load_env()
    region = os.environ.get("GPU_NODE_REGION", "ap-beijing")
    tool_name = need("AGS_TOOL_NAME")

    net = gpu_net(region, need("GPU_NODE_INSTANCE_ID"))
    print("=" * 72)
    print(" VPC 内网直通验证")
    print("=" * 72)
    print(f"  GPU 节点内网 IP : {net['private_ip']}")
    print(f"  子网 CIDR       : {net['cidr']}")
    print(f"  沙箱工具        : {tool_name}（地域 {region}）")
    print()

    from clients.ags import AGSClient
    from clients.sandbox import start_instance_with_warmup
    from e2b_code_interpreter import Sandbox

    ags = AGSClient()
    tool = ags.find_tool(tool_name)
    if not tool:
        sys.exit(f"[x] 找不到工具 {tool_name}（注意地域：AGS 按地域隔离）")

    ns = f"{need('TCR_REGISTRY')}/{need('TCR_NAMESPACE')}"
    first = json.loads((ROOT / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    image = f"{ns}/sweb-{first['task_id'].replace('__', '-').lower()}:sbx"

    t0 = time.time()
    inst, _ = start_instance_with_warmup(ags, tool["tool_id"], image, cpu="2", memory="4Gi")
    print(f"  实例 {inst} 启动 {time.time() - t0:.1f}s\n")

    results: list[tuple[str, bool, str]] = []
    try:
        sbx = Sandbox.connect(inst)

        # ---- 1) 沙箱自身的内网 IP 是否在 GPU 子网内（最硬的证据）----
        # SWE-bench 官方镜像很精简，`ip` / `ifconfig` 都可能没有，
        # 因此用多种手段依次尝试，最后回退到解析 /proc/net/fib_trie
        probe = (
            "(command -v ip >/dev/null && ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1) "
            "|| (command -v ifconfig >/dev/null && ifconfig | grep -oE 'inet (addr:)?[0-9.]+' "
            "| grep -oE '[0-9.]+$') "
            "|| (command -v hostname >/dev/null && hostname -I) "
            "|| python3 -c \"import socket;print(socket.gethostbyname(socket.gethostname()))\""
        )
        _, out = sbx_run(sbx, probe)
        ips = [
            l.strip() for l in out.replace(" ", "\n").splitlines()
            if l.strip() and l.strip().count(".") == 3
            and all(part.isdigit() for part in l.strip().split("."))
        ]
        subnet = ipaddress.ip_network(net["cidr"])
        in_subnet = [
            ip for ip in ips
            if not ip.startswith("127.") and ipaddress.ip_address(ip) in subnet
        ]
        # 仅作参考项：容器内看到的多是 veth 地址（169.254.x.x），
        # 不能据此判断 VPC 是否生效
        results.append((
            "[参考] 容器内网卡地址",
            True,
            f"{ips}（子网 {net['cidr']}；容器 veth 地址属正常）",
        ))

        # ---- 2) 到 GPU 内网 IP 的可达性 ----
        gip = net["private_ip"]
        # ping 需要 ICMP 且镜像常未安装 iputils，不作为判据
        _, out = sbx_run(sbx, f"command -v ping >/dev/null && ping -c 2 -W 2 {gip} 2>&1 | tail -1 || echo '(镜像未装 ping，跳过)'")
        results.append(("[参考] ICMP", True, out.strip()[:80]))

        # SSH 端口探测（更贴近真实业务：能建 TCP 连接）
        _, out = sbx_run(
            sbx,
            f"(command -v nc >/dev/null && nc -z -w3 {gip} 22 && echo TCP_OK) "
            f"|| timeout 4 bash -c '</dev/tcp/{gip}/22' 2>/dev/null && echo TCP_OK || echo TCP_FAIL",
        )
        ok3 = "TCP_OK" in out
        results.append((f"TCP 连通 {gip}:22", ok3, out.strip()[:60]))

        # ---- 3) 出口公网 IP（与 PUBLIC 模式对照）----
        _, out = sbx_run(sbx, "curl -s -m 10 https://api.ipify.org || echo NO_EGRESS")
        egress = out.strip()[:40]
        # VPC 模式下应当**取不到**公网 IP；PUBLIC 模式下每次会拿到一个随机公网 IP
        no_public = ("NO_EGRESS" in egress) or not egress
        results.append((
            "无公网出口（VPC 应无）",
            no_public,
            egress if egress else "(空)",
        ))

    finally:
        try:
            ags.stop_instance(inst)
            print("  实例已回收\n")
        except Exception as e:
            print(f"  ⚠️ 回收失败 {inst}: {e}\n")

    print("-" * 72)
    for name, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {name:28s} {detail}")
    print("-" * 72)

    # 判据：只看「TCP 连通 GPU 内网」与「无公网出口」两项
    core = [r for r in results if not r[0].startswith("[参考]")]
    if all(ok for _, ok, _ in core):
        print("\n✓ VPC 内网直通已验证：沙箱可访问 GPU 内网、且无公网出口")
        print("  → 满足「沙箱与训练侧走内网打通、不暴露公网」的要求")
        return 0
    print("\n✗ 未通过，需检查：安全组入站规则 / 子网路由 / 工具的 VpcConfig")
    return 1


if __name__ == "__main__":
    sys.exit(main())
