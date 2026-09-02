#!/usr/bin/env python3
"""
创建 VPC 网络类型的沙箱工具（与 GPU 节点内网互通）
================================================

## 为什么要做这件事

原先的沙箱工具用 `NetworkMode=PUBLIC`（`clients/ags.py` 的默认值），导致：

  · 沙箱容器直接挂在公网上，出口 IP 每次随机
    （实测 6 次采样得 6 个互不相同的公网 IP，跨 118/81/121/150/49/122 多个 /8 段）
  · 无法做 IP 白名单，训练侧若要暴露推理服务只能开公网端口 + NAT 转发
  · 不符合「沙箱与训练侧走内网打通」的要求

改用 `NetworkMode=VPC` 后，沙箱容器直接接入指定子网，与同 VPC 内的 GPU 节点
（内网 IP `10.0.0.x`）直接互通 —— **零公网暴露，不需要 NAT、不需要 API Key 兜底**。

## 硬约束：必须同地域

跨地域 VPC 不通（需 CCN 云联网）。因此沙箱地域必须与 GPU 节点一致。
已实测 AGS 支持 ap-beijing / ap-shanghai / ap-guangzhou / ap-nanjing，
本脚本默认取 GPU 节点所在地域。

## 安全设计

  · **默认 dry-run**：只打印将要提交的参数，不实际创建。加 `--yes` 才执行
  · **只做创建，绝不删除**：配额不足时报错退出，由人工决定删哪个
  · 凭证只从环境变量读取，输出中不含任何密钥

用法：
    python3 scripts/create_vpc_sandbox_tool.py              # 预览参数（安全）
    python3 scripts/create_vpc_sandbox_tool.py --yes        # 确认后执行
    python3 scripts/create_vpc_sandbox_tool.py --list       # 只看现有工具与配额
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUOTA_PER_REGION = 10  # 每地域沙箱工具配额（实测上海 10/10 已满）


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)


def need(key: str) -> str:
    v = (os.environ.get(key) or "").strip()
    if not v:
        sys.exit(f"[x] 缺少环境变量 {key}（见 .env）")
    return v


def gpu_node_network(region: str, instance_id: str) -> dict:
    """读取 GPU 节点的 VPC / 子网 / 安全组 —— 沙箱必须与它对齐才能内网互通。"""
    from tencentcloud.common import credential
    from tencentcloud.cvm.v20170312 import cvm_client, models

    cred = credential.Credential(
        need("TENCENTCLOUD_SECRET_ID"), need("TENCENTCLOUD_SECRET_KEY")
    )
    c = cvm_client.CvmClient(cred, region)
    req = models.DescribeInstancesRequest()
    req.from_json_string(json.dumps({"InstanceIds": [instance_id]}))
    rs = c.DescribeInstances(req).InstanceSet
    if not rs:
        sys.exit(f"[x] 在 {region} 找不到实例 {instance_id}")
    i = rs[0]
    v = i.VirtualPrivateCloud
    return {
        "zone": i.Placement.Zone,
        "vpc_id": v.VpcId,
        "subnet_id": v.SubnetId,
        "security_group_ids": list(i.SecurityGroupIds or []),
        "private_ip": list(i.PrivateIpAddresses or []),
    }


def ags_client_for(region: str):
    from tencentcloud.ags.v20250920 import ags_client
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile

    cred = credential.Credential(
        need("TENCENTCLOUD_SECRET_ID"), need("TENCENTCLOUD_SECRET_KEY")
    )
    hp = HttpProfile()
    hp.endpoint = "ags.tencentcloudapi.com"
    hp.reqTimeout = 60
    cp = ClientProfile()
    cp.httpProfile = hp
    return ags_client.AgsClient(cred, region, cp)


def list_tools(region: str) -> list[dict]:
    from tencentcloud.ags.v20250920 import models

    c = ags_client_for(region)
    req = models.DescribeSandboxToolListRequest()
    req.Offset, req.Limit = 0, 100
    rsp = c.DescribeSandboxToolList(req)
    out = []
    for t in rsp.SandboxToolSet or []:
        out.append({
            "name": getattr(t, "ToolName", "?"),
            "tool_id": getattr(t, "ToolId", "?"),
            "status": getattr(t, "Status", "?"),
        })
    return out


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="确认执行创建（默认只预览）")
    ap.add_argument("--list", action="store_true", help="只列出现有工具与配额")
    ap.add_argument("--name", default="swe-rl-vpc-runner", help="工具名")
    ap.add_argument("--cpu", default="2")
    ap.add_argument("--memory", default="4Gi")
    ap.add_argument(
        "--image",
        default="",
        help="占位镜像（运行时用 image_override 逐题切换）；默认取训练集第一题",
    )
    args = ap.parse_args()

    region = os.environ.get("GPU_NODE_REGION", "ap-beijing")
    instance_id = need("GPU_NODE_INSTANCE_ID")

    tools = list_tools(region)
    print(f"地域 {region} 现有沙箱工具 {len(tools)}/{QUOTA_PER_REGION}：")
    for t in tools:
        print(f"    {t['name']:52s} {t['tool_id']}  {t['status']}")
    if args.list:
        return 0

    if len(tools) >= QUOTA_PER_REGION:
        print(
            f"\n[x] 配额已满（{len(tools)}/{QUOTA_PER_REGION}）。\n"
            "    本脚本**不会自动删除**任何工具 —— 请人工确认可删除哪一个后再重试。"
        )
        return 1
    if any(t["name"] == args.name for t in tools):
        print(f"\n[!] 已存在同名工具 {args.name}，无需重复创建")
        return 0

    net = gpu_node_network(region, instance_id)

    # 占位镜像：工具必须绑定一个镜像才能创建；实际运行时按题目 override
    image = args.image
    if not image:
        tasks_file = ROOT / "data" / "tasks.jsonl"
        first = json.loads(tasks_file.read_text(encoding="utf-8").splitlines()[0])
        slug = first["task_id"].replace("__", "-").lower()
        image = f"{need('TCR_REGISTRY')}/{need('TCR_NAMESPACE')}/sweb-{slug}:sbx"

    print("\n" + "=" * 72)
    print(" 将要创建的沙箱工具（参数已与 GPU 节点对齐）")
    print("=" * 72)
    rows = [
        ("地域", region),
        ("工具名", args.name),
        ("网络模式", "VPC  ← 关键改动（原先是 PUBLIC）"),
        ("可用区（GPU 节点）", net["zone"]),
        ("VPC", net["vpc_id"]),
        ("子网", net["subnet_id"]),
        ("安全组", ", ".join(net["security_group_ids"]) or "(未绑定)"),
        ("GPU 节点内网 IP", ", ".join(net["private_ip"])),
        ("占位镜像", image),
        ("规格", f"{args.cpu} 核 / {args.memory}"),
    ]
    for k, v in rows:
        print(f"  {k:20s} {v}")
    print("=" * 72)
    print(" 效果：沙箱容器接入上述子网，与 GPU 节点内网直通，无需公网暴露")
    print("=" * 72)

    if not args.yes:
        print("\n[dry-run] 未执行任何创建操作。")
        print("确认无误后追加 --yes 执行：")
        print("    python3 scripts/create_vpc_sandbox_tool.py --yes")
        return 0

    from clients.ags import AGSClient

    print("\n正在创建…")
    ags = AGSClient(region=region) if "region" in AGSClient.__init__.__code__.co_varnames else AGSClient()
    tool_id = ags.create_tool(
        args.name,
        image,
        network_mode="VPC",
        subnet_ids=[net["subnet_id"]],
        security_group_ids=net["security_group_ids"],
        cpu=args.cpu,
        memory=args.memory,
        description="SWE-RL: VPC 网络沙箱，与 GPU 节点内网互通",
    )
    print(f"✓ 已创建，ToolId = {tool_id}")
    print("\n请把工具名写入 .env：")
    print(f"    AGS_TOOL_NAME={args.name}")
    print(f"    AGS_REGION={region}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
