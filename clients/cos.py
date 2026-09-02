"""腾讯云 COS 客户端：上传 tracing.jsonl（SandBox → TKE 的传递通道）。

只封装本课题实际要用到的两个操作：上传单个文件、列出某前缀下的对象。
凭证同样只从环境变量读取，不落盘、不出现在日志。
"""
from __future__ import annotations

import os


class COSError(RuntimeError):
    """COS 操作失败。"""


def _client():
    try:
        from qcloud_cos import CosConfig, CosS3Client
    except ImportError as e:
        raise COSError("缺少 cos-python-sdk-v5，请先 pip install cos-python-sdk-v5") from e

    sid = os.environ.get("TENCENTCLOUD_SECRET_ID", "")
    skey = os.environ.get("TENCENTCLOUD_SECRET_KEY", "")
    region = os.environ.get("COS_REGION") or os.environ.get("TENCENTCLOUD_REGION", "ap-shanghai")
    if not sid or not skey:
        raise COSError("未配置 TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY（见 .env）")
    cfg = CosConfig(Region=region, SecretId=sid, SecretKey=skey)
    return CosS3Client(cfg)


def upload_file(bucket: str, key: str, local_path: str) -> str:
    """上传单个文件到 COS，返回对象的 cos:// 风格路径（仅供日志展示）。"""
    cli = _client()
    with open(local_path, "rb") as f:
        cli.put_object(Bucket=bucket, Body=f, Key=key)
    return f"cos://{bucket}/{key}"


def upload_bytes(bucket: str, key: str, data: bytes) -> str:
    cli = _client()
    cli.put_object(Bucket=bucket, Body=data, Key=key)
    return f"cos://{bucket}/{key}"


def download_file(bucket: str, key: str, local_path: str) -> None:
    cli = _client()
    rsp = cli.get_object(Bucket=bucket, Key=key)
    body = rsp["Body"].get_raw_stream().read()
    os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(body)


def list_objects(bucket: str, prefix: str = "") -> list[str]:
    cli = _client()
    rsp = cli.list_objects(Bucket=bucket, Prefix=prefix)
    return [c["Key"] for c in (rsp.get("Contents") or [])]
