#!/usr/bin/env python3
"""创建本项目专用 COS bucket 并端到端验证（幂等）
==================================================

课题要求 tracing 经 **COS/CFS** 从 SandBox 侧传到 TKE 训练侧。本脚本负责
准备该通道，并**实测一遍 上传→列举→下载→比对**，避免"配置看着对、真跑才发现
权限不足"。

## 为什么新建而非复用

账号下有 349 个 bucket（76 个在北京），但全部属于其他项目/同事。往别人的
bucket 写数据既不礼貌也不安全（可能触发对方的生命周期规则导致数据丢失）。
新建独立 bucket 的成本可忽略（tracing 总量约几 MB）。

## 命名

`swe-rl-trace-bj-<APPID>` —— COS 要求 bucket 名带 APPID 后缀。APPID 从
`COS_APPID` 读取，或从任一现有 bucket 名的末段自动推断（同账号下 APPID 唯一）。

`-bj` 后缀不是冗余：**COS bucket 名在同一 APPID 下跨地域全局唯一**。账号里已有
一个上一轮在 ap-shanghai 建的 `swe-rl-tracing-<APPID>`，直接同名建北京桶会报
`BucketAlreadyExists ... in other region`。而沙箱与 GPU 节点都在北京，
用北京桶可走内网、免流量费、延迟更低，所以宁可换名也不复用上海那个。

## 权限设置

**私有读写**（默认），不开公网访问。GPU 节点与沙箱都用密钥访问，无需公有权限。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REGION = "ap-beijing"  # 与 GPU 节点、沙箱同地域，走内网免流量费
BUCKET_PREFIX = "swe-rl-trace-bj"


def infer_appid(client) -> str:
    """从现有 bucket 名推断 APPID。

    COS bucket 名固定形如 `<name>-<appid>`，同账号下 APPID 唯一，
    因此任取一个现有 bucket 的末段即可，无需额外调 CAM 接口。
    """
    if os.environ.get("COS_APPID"):
        return os.environ["COS_APPID"]
    bs = client.list_buckets().get("Buckets", {}).get("Bucket", [])
    if isinstance(bs, dict):
        bs = [bs]
    for b in bs:
        parts = str(b.get("Name", "")).rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[1]
    raise RuntimeError("无法推断 COS APPID，请在 .env 里设置 COS_APPID")


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)

    from qcloud_cos import CosConfig, CosS3Client

    sid = os.environ.get("TENCENTCLOUD_SECRET_ID", "")
    skey = os.environ.get("TENCENTCLOUD_SECRET_KEY", "")
    if not sid or not skey:
        print("✗ 缺少 TENCENTCLOUD_SECRET_ID / SECRET_KEY", file=sys.stderr)
        return 2

    cli = CosS3Client(CosConfig(Region=REGION, SecretId=sid, SecretKey=skey))
    appid = infer_appid(cli)
    bucket = f"{BUCKET_PREFIX}-{appid}"
    print(f"目标 bucket: {bucket}（地域 {REGION}，私有读写）")

    exists = False
    try:
        cli.head_bucket(Bucket=bucket)
        exists = True
        print("  已存在，跳过创建")
    except Exception:  # noqa: BLE001 —— head 失败即视为不存在，交给下面创建
        pass

    if not exists:
        try:
            cli.create_bucket(Bucket=bucket, ACL="private")
            print("  ✓ 创建成功")
            time.sleep(3)  # 新建 bucket 需要几秒才可写
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ 创建失败：{str(e)[:300]}", file=sys.stderr)
            return 1

    # ---- 端到端实测：上传 → 列举 → 下载 → 比对 ----
    from clients import cos

    os.environ["COS_REGION"] = REGION
    key = "_selftest/roundtrip.json"
    payload = json.dumps(
        {"probe": "swe-rl", "ts": time.time()}, ensure_ascii=False
    ).encode()
    try:
        cos.upload_bytes(bucket, key, payload)
        keys = cos.list_objects(bucket, "_selftest/")
        assert key in keys, f"上传后未在列表中找到 {key}：{keys}"
        tmp = ROOT / "data" / "_cos_roundtrip.json"
        cos.download_file(bucket, key, str(tmp))
        assert tmp.read_bytes() == payload, "下载内容与上传不一致"
        tmp.unlink(missing_ok=True)
        print("  ✓ 通路自检：上传/列举/下载/比对 全部通过")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ 通路自检失败：{type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
        return 1

    print("\n请把以下两行加入 .env（本脚本不自动改 .env，避免覆盖你的手工修改）：")
    print(f"  COS_REGION={REGION}")
    print(f"  COS_BUCKET={bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
