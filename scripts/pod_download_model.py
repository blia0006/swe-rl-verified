#!/usr/bin/env python3
"""
在 GPU 宿主机上下载模型权重（经 ModelScope 国内直连）
====================================================

为什么不用 HuggingFace：GPU 节点实测**访问不了 huggingface.co**（返回 000），
但 ModelScope 可达（302/200）。为什么不在本机下再传：本机是 ARM64 且要多一次
上传下载，节点直连 ModelScope 带宽更好。

为什么不用 `modelscope` SDK：节点上没装，而 pip 装 SDK 会连带拉一堆依赖
（datasets/pandas/pyarrow 等），没必要 —— ModelScope 的文件下载就是普通 HTTPS GET，
用标准库直接拉即可，零新增依赖。

特性：
  · 断点续传（HTTP Range），中断后重跑不会从头下
  · 逐文件校验大小，不完整则重下
  · 落盘在宿主机目录（默认 /data/swe-rl/model/<name>），容器通过 bind mount 使用，
    **容器重建不丢权重** —— 这是上一轮 pod 方案的教训

用法（本脚本设计为在 GPU 宿主机上执行，由 scripts/node.py 投放）：
    python3 pod_download_model.py --model Qwen/Qwen2.5-Coder-7B-Instruct
    python3 pod_download_model.py --model ... --dest /data/swe-rl/model  --check-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://modelscope.cn/api/v1/models/{model}/repo/files?Revision={rev}"
DOWNLOAD = "https://modelscope.cn/api/v1/models/{model}/repo?Revision={rev}&FilePath={path}"

# 只下推理/训练必需的文件，跳过 GGUF、ONNX、示例图等无关体积
SKIP_SUFFIX = {".gguf", ".onnx", ".png", ".jpg", ".jpeg", ".gif", ".md", ".pdf"}
SKIP_NAMES = {"README.md", "LICENSE", "configuration.json", ".gitattributes"}


def http_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "swe-rl/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def list_files(model: str, rev: str) -> list[dict]:
    data = http_json(API.format(model=urllib.parse.quote(model, safe="/"), rev=rev))
    files = data.get("Data", {}).get("Files", [])
    out = []
    for f in files:
        path = f.get("Path", "")
        if f.get("Type") != "blob" or not path:
            continue
        if path in SKIP_NAMES or Path(path).suffix.lower() in SKIP_SUFFIX:
            continue
        out.append({"path": path, "size": int(f.get("Size") or 0)})
    return out


def download_one(model: str, rev: str, item: dict, dest: Path, retries: int = 4) -> None:
    """带断点续传地下载单个文件。"""
    path, size = item["path"], item["size"]
    target = dest / path
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and size and target.stat().st_size == size:
        print(f"  [skip] {path}  已完整 ({size / 2**20:.1f} MiB)")
        return

    url = DOWNLOAD.format(
        model=urllib.parse.quote(model, safe="/"),
        rev=rev,
        path=urllib.parse.quote(path, safe=""),
    )

    for attempt in range(1, retries + 1):
        have = target.stat().st_size if target.exists() else 0
        if size and have > size:  # 本地比远端还大，说明是坏文件，删掉重下
            target.unlink()
            have = 0
        headers = {"User-Agent": "swe-rl/1.0"}
        mode = "wb"
        if have:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
        try:
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r, open(target, mode) as f:
                # 服务端忽略 Range 时会返回 200 + 全量，此时必须从头写
                if have and r.status == 200:
                    f.close()
                    target.unlink()
                    have = 0
                    with open(target, "wb") as f2:
                        _pump(r, f2)
                else:
                    _pump(r, f)
            got = target.stat().st_size
            dt = time.time() - t0
            speed = (got - have) / max(dt, 1e-6) / 2**20
            if size and got != size:
                raise OSError(f"大小不符 got={got} want={size}")
            print(f"  [ok]   {path}  {got / 2**20:.1f} MiB  {speed:.1f} MiB/s")
            return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt == retries:
                raise
            wait = 2**attempt
            print(f"  [retry {attempt}/{retries}] {path}: {e} → {wait}s 后续传")
            time.sleep(wait)


def _pump(resp, fh, chunk: int = 1 << 20) -> None:
    while True:
        buf = resp.read(chunk)
        if not buf:
            break
        fh.write(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--revision", default="master")
    ap.add_argument(
        "--dest",
        default=os.environ.get("MODEL_DIR", "/data/swe-rl/model"),
        help="宿主机落盘根目录；实际路径为 <dest>/<模型名>",
    )
    ap.add_argument("--check-only", action="store_true", help="只核对完整性，不下载")
    args = ap.parse_args()

    name = args.model.split("/")[-1]
    dest = Path(args.dest) / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"模型   : {args.model}")
    print(f"落盘   : {dest}")
    files = list_files(args.model, args.revision)
    total = sum(f["size"] for f in files)
    print(f"待下载 : {len(files)} 个文件，共 {total / 2**30:.2f} GiB\n")

    missing = []
    for f in files:
        t = dest / f["path"]
        if not t.exists() or (f["size"] and t.stat().st_size != f["size"]):
            missing.append(f)

    if args.check_only:
        if missing:
            print(f"✗ 缺失/不完整 {len(missing)} 个：")
            for f in missing:
                print(f"    {f['path']}")
            return 1
        print("✓ 全部文件完整")
        return 0

    if not missing:
        print("✓ 已全部就位，无需下载")
        return 0

    t0 = time.time()
    for f in files:
        download_one(args.model, args.revision, f, dest)
    print(f"\n耗时 {time.time() - t0:.0f}s")

    # 收尾自检：训练/推理必需文件是否齐全
    need = ["config.json", "tokenizer_config.json"]
    lack = [n for n in need if not (dest / n).exists()]
    weights = list(dest.glob("*.safetensors")) + list(dest.glob("*.bin"))
    if lack or not weights:
        print(f"✗ 自检未通过：缺 {lack}，权重文件数={len(weights)}")
        return 1
    got = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"✓ 自检通过：{len(weights)} 个权重分片，落盘合计 {got / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
