#!/usr/bin/env python3
"""诊断 OCI 归档结构：找出 push 时报 `content digest ...: not found` 的悬空引用。

用法（节点上）： python3 probe_oci.py <解包后的归档目录>
"""
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def bp(digest: str) -> pathlib.Path:
    return root / "blobs" / "sha256" / digest.split(":", 1)[1]


def walk(digest: str, depth: int = 0, label: str = "") -> None:
    pad = "  " * depth
    p = bp(digest)
    exists = p.exists()
    mark = "OK " if exists else "MISSING"
    print(f"{pad}[{mark}] {label} {digest[:24]}")
    if not exists:
        return
    try:
        obj = json.loads(p.read_bytes())
    except Exception:
        return  # 是 layer 二进制，不是 JSON
    for child in obj.get("manifests", []):
        plat = child.get("platform", {})
        arch = f"{plat.get('architecture')}/{plat.get('os')}"
        walk(child["digest"], depth + 1, f"child arch={arch}")
    cfg = obj.get("config")
    if isinstance(cfg, dict) and "digest" in cfg:
        walk(cfg["digest"], depth + 1, "config")
    for i, layer in enumerate(obj.get("layers", [])):
        p2 = bp(layer["digest"])
        if not p2.exists():
            print(f"{pad}  [MISSING] layer[{i}] {layer['digest'][:24]}")


idx = json.loads((root / "index.json").read_text())
print("=== index.json ===")
for m in idx["manifests"]:
    walk(m["digest"], 0, "top")

print("\n=== 归档内 blob 总数 ===")
blobs = list((root / "blobs" / "sha256").glob("*"))
print(len(blobs))
