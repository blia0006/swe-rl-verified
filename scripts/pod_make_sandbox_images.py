#!/usr/bin/env python3
"""
沙箱镜像制作（反向融合）：以 AGS 可用镜像为底，叠入 SWE-bench 题目环境
=====================================================================

## 为什么是"反向"

先说清两种融合方向的区别：

| 方向 | 做法 | 结果 |
|---|---|---|
| 正向（已放弃） | 以**官方镜像**为底，叠入 AGS 的 envd agent | ✗ 需手工改写 OCI 清单，AGS 报 `ImagePrepare: Internal server error` |
| **反向（本脚本）** | 以**已验证可用的 AGS 镜像**为底，叠入官方镜像的题目环境 | ✓ 清单由 `ctr` 正常流程生成，AGS 必然接受 |

正向的致命弱点：新镜像的 manifest/config 是我手工拼的 JSON。虽然本地
`ctr run` 能起（实测 `/init` + `envd 0.2.11` + `/testbed` 齐备），
但 AGS 平台侧对清单有额外校验，手工构造过不了，且报 `Internal server error`
无法进一步定位 —— 这类"对方黑盒校验"的问题不该硬碰。

反向则完全不碰清单：所有镜像层与元数据都由 containerd 自己产生。

## 具体做法

```
底座：swe-synth-base:ubuntu22.04-v1   （已验证可作 AGS 沙箱工具，含 envd + s6）
叠加：官方镜像的 /testbed + /opt/miniconda3   （题目代码 + 依赖环境）
```

用 `ctr images mount` 把两个镜像的 rootfs 挂到宿主机目录，
把题目环境**文件级拷贝**进底座的可写层，再 commit —— 但 ctr 无 commit，
因此改为：挂载底座为可写 → 拷入题目环境 → 用 `ctr images export` 导出
**由 ctr 自己生成**的镜像。

⚠️ 这条路仍需 commit 能力。containerd v1.7 的 ctr 确实没有。
因此最终实现采用 **buildkit-less 的 tar 叠加 + import**，但**只新增一层**、
**不改动原有清单结构**（仅追加），并保留原镜像的 config 主体 ——
与正向方案的区别在于：底座的 config 本就被 AGS 接受过，我们只追加 layer
与 diff_id，不触碰 Entrypoint/Cmd 等 AGS 关心的字段。

用法（GPU 宿主机执行）：
    python3 pod_make_sandbox_images.py --tasks /data/swe-rl/data/tasks.jsonl --only <task_id>
    python3 pod_make_sandbox_images.py --tasks ... --jobs 2 --resume
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

NS = "k8s.io"
BASE_IMAGE_TAG = "swe-synth-base:ubuntu22.04-v1"
WORK_ROOT = Path("/data/swe-rl/build2")

# 从官方镜像搬到底座的目录：题目代码 + 其 conda 环境
PAYLOAD_PATHS = ["testbed", "opt/miniconda3"]

_LOCK = threading.Lock()
MEDIA_LAYER_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
MEDIA_LAYER_D2 = "application/vnd.docker.image.rootfs.diff.tar.gzip"


def load_env(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def need(key: str) -> str:
    v = (os.environ.get(key) or "").strip()
    if not v:
        sys.exit(f"[x] 缺少环境变量 {key}")
    return v


def _scrub(t: str, secrets: tuple[str, ...]) -> str:
    for s in secrets:
        if s and len(s) >= 6:
            t = t.replace(s, "***")
    return t


def _denoise(t: str) -> str:
    return "\n".join(
        ln for ln in t.splitlines()
        if "DEPRECATION" not in ln and "is deprecated since containerd" not in ln
    ).strip()


def sh(cmd: str, timeout: int = 1800, quiet: bool = False,
       secrets: tuple[str, ...] = ()) -> tuple[int, str]:
    if not quiet:
        print(f"    $ {_scrub(cmd, secrets)}", flush=True)
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, _denoise(_scrub((p.stdout or "") + (p.stderr or ""), secrets))
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"


def tcr_ref(task_id: str, registry: str, namespace: str, tag: str) -> str:
    slug = task_id.replace("__", "-").lower()
    return f"{registry}/{namespace}/sweb-{slug}:{tag}"


def image_present(ref: str) -> bool:
    code, out = sh(f"ctr -n {NS} images ls -q", timeout=120, quiet=True)
    return code == 0 and ref in out.splitlines()


def sha256_file(path: Path) -> tuple[str, int]:
    h, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def blob_path(root: Path, digest: str) -> Path:
    return root / "blobs" / "sha256" / digest.split(":", 1)[1]


def write_blob(root: Path, data: bytes) -> tuple[str, int]:
    d = hashlib.sha256(data).hexdigest()
    p = blob_path(root, f"sha256:{d}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return f"sha256:{d}", len(data)


def read_json_blob(root: Path, digest: str) -> dict:
    return json.loads(blob_path(root, digest).read_bytes())


# ------------------------------------------------------------------ 提取题目环境层

def build_payload_layer(task_id: str, src_ref: str, creds: dict, work: Path
                        ) -> tuple[str, int, str]:
    """把官方镜像的题目环境打成一个 gzip 层。返回 (digest, size, diff_id)。"""
    secrets = (creds["password"], f"{creds['username']}:{creds['password']}")
    if not image_present(src_ref):
        code, out = sh(
            f"ctr -n {NS} images pull --platform linux/amd64 "
            f"-u '{creds['username']}:{creds['password']}' {src_ref}",
            secrets=secrets,
        )
        if code != 0:
            raise RuntimeError(f"pull 官方镜像失败: {out[-400:]}")

    mnt = work / "src_mnt"
    mnt.mkdir(parents=True, exist_ok=True)
    sh(f"ctr -n {NS} images unmount {mnt} 2>/dev/null", quiet=True)
    code, out = sh(f"ctr -n {NS} images mount --rw {src_ref} {mnt}", timeout=600)
    if code != 0:
        raise RuntimeError(f"挂载官方镜像失败: {out[-400:]}")

    raw = work / "payload.tar"
    try:
        for rel in PAYLOAD_PATHS:
            if not (mnt / rel).exists():
                raise RuntimeError(f"官方镜像缺少 {rel}（布局与预期不符）")
        # 用系统 tar 而非 python tarfile：conda 环境有大量符号链接与硬链接，
        # 系统 tar 处理更可靠且快得多（几 GB 规模）
        paths = " ".join(PAYLOAD_PATHS)
        code, out = sh(
            f"tar cf {raw} -C {mnt} --numeric-owner {paths}", timeout=1800
        )
        if code != 0:
            raise RuntimeError(f"打包题目环境失败: {out[-400:]}")
    finally:
        sh(f"ctr -n {NS} images unmount {mnt}", timeout=120, quiet=True)

    diff_id, raw_size = sha256_file(raw)
    gz = work / "payload.tar.gz"
    # compresslevel=1：这层有几 GB，压缩比在此场景收益远小于耗时代价
    with open(raw, "rb") as fi, gzip.open(gz, "wb", compresslevel=1) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    raw.unlink(missing_ok=True)
    digest, size = sha256_file(gz)
    print(f"    题目环境层：压缩 {size / 2**30:.2f}GiB / 原始 {raw_size / 2**30:.2f}GiB")
    return f"sha256:{digest}", size, f"sha256:{diff_id}"


# ------------------------------------------------------------------ 叠加到底座

def append_layer(root: Path, layer_digest: str, layer_size: int, diff_id: str,
                 env_extra: list[str]) -> None:
    """在底座的 OCI 归档里追加一层。

    与正向方案的关键差别：**不修改 Entrypoint / Cmd**。
    底座的这些字段本就被 AGS 接受过（它是现役可用的沙箱工具镜像），
    保持原样即可 —— 我们只追加文件层，外加必要的 PATH 环境变量。
    """
    index = json.loads((root / "index.json").read_text())

    def patch(man_digest: str) -> tuple[str, int]:
        man = read_json_blob(root, man_digest)

        if "manifests" in man:
            children = []
            for child in man["manifests"]:
                plat = child.get("platform", {})
                arch = plat.get("architecture")
                # 剔除悬空的 attestation/SBOM 子项（arch=unknown），
                # 否则 push 报 content digest not found
                if (arch in (None, "unknown") or plat.get("os") == "unknown") and \
                        not blob_path(root, child["digest"]).exists():
                    print(f"    · 剔除悬空子项 {child['digest'][:19]}…")
                    continue
                if arch not in (None, "amd64"):
                    children.append(child)
                    continue
                nd, ns = patch(child["digest"])
                c = dict(child)
                c["digest"], c["size"] = nd, ns
                children.append(c)
            man["manifests"] = children
            return write_blob(root, json.dumps(man, separators=(",", ":")).encode())

        cfg_desc = man["config"]
        cfg = read_json_blob(root, cfg_desc["digest"])
        cfg.setdefault("rootfs", {}).setdefault("diff_ids", []).append(diff_id)

        # 只补 PATH，让题目环境的 conda python 可用；不动 Entrypoint/Cmd
        for key in ("config", "container_config"):
            if key in cfg and isinstance(cfg[key], dict):
                env = list(cfg[key].get("Env") or [])
                merged = []
                seen_path = False
                for e in env:
                    if e.startswith("PATH=") and not seen_path:
                        seen_path = True
                        merged.append(f"PATH={':'.join(env_extra)}:{e[5:]}")
                    else:
                        merged.append(e)
                if not seen_path:
                    merged.append(
                        "PATH=" + ":".join(env_extra)
                        + ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                    )
                cfg[key]["Env"] = merged

        cfg.setdefault("history", []).append({
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "created_by": "swe-rl: add SWE-bench testbed + conda env",
        })
        ncd, ncs = write_blob(root, json.dumps(cfg, separators=(",", ":")).encode())
        man["config"] = {
            "mediaType": cfg_desc.get("mediaType", "application/vnd.oci.image.config.v1+json"),
            "digest": ncd,
            "size": ncs,
        }
        existing = man["layers"][-1]["mediaType"] if man.get("layers") else MEDIA_LAYER_GZIP
        man["layers"].append({
            "mediaType": MEDIA_LAYER_D2 if "docker" in existing else MEDIA_LAYER_GZIP,
            "digest": layer_digest,
            "size": layer_size,
        })
        return write_blob(root, json.dumps(man, separators=(",", ":")).encode())

    new_ms = []
    for m in index["manifests"]:
        nd, ns = patch(m["digest"])
        mm = dict(m)
        mm["digest"], mm["size"] = nd, ns
        new_ms.append(mm)
    index["manifests"] = new_ms
    (root / "index.json").write_text(json.dumps(index, separators=(",", ":")))


def make_one(task: dict, args, creds: dict, base_ref: str) -> dict:
    task_id = task["task_id"]
    src = tcr_ref(task_id, creds["registry"], creds["namespace"], args.src_tag)
    dst = tcr_ref(task_id, creds["registry"], creds["namespace"], args.tag)
    secrets = (creds["password"], f"{creds['username']}:{creds['password']}")
    rec = {"task_id": task_id, "src": src, "sandbox_image": dst}
    print(f"\n=== {task_id}\n    底座: {base_ref}\n    题目: {src}\n    产出: {dst}", flush=True)

    t0 = time.time()
    work = WORK_ROOT / f"w_{task_id.replace('__', '-').lower()}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    try:
        layer_digest, layer_size, diff_id = build_payload_layer(task_id, src, creds, work)

        oci = work / "oci"
        oci.mkdir()
        code, out = sh(
            f"ctr -n {NS} images export --platform linux/amd64 {work}/base.tar {base_ref}",
            timeout=args.timeout,
        )
        if code != 0:
            rec.update(status="export_failed", error=out[-400:])
            return rec
        with tarfile.open(work / "base.tar") as tf:
            tf.extractall(oci)  # 来源是本地 ctr 导出，非不可信输入
        (work / "base.tar").unlink(missing_ok=True)

        shutil.move(str(work / "payload.tar.gz"), blob_path(oci, layer_digest))
        append_layer(
            oci, layer_digest, layer_size, diff_id,
            env_extra=["/opt/miniconda3/envs/testbed/bin", "/opt/miniconda3/bin"],
        )

        fused = work / "fused.tar"
        with tarfile.open(fused, "w") as tf:
            for item in sorted(oci.iterdir()):
                tf.add(item, arcname=item.name, recursive=True)

        code, out = sh(
            f"ctr -n {NS} images import --platform linux/amd64 --index-name {dst} {fused}",
            timeout=args.timeout,
        )
        if code != 0:
            rec.update(status="import_failed", error=out[-500:])
            return rec

        code, out = sh(
            f"ctr -n {NS} images push --platform linux/amd64 "
            f"-u '{creds['username']}:{creds['password']}' {dst}",
            timeout=args.timeout, secrets=secrets,
        )
        if code != 0:
            rec.update(status="push_failed", error=out[-500:])
            return rec
    except Exception as e:
        rec.update(status="exception", error=str(e)[:500])
        print(f"    ✗ {e}")
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)

    rec.update(status="ok", elapsed_s=round(time.time() - t0, 1))
    print(f"    ✓ 完成 {rec['elapsed_s']:.0f}s")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/data/swe-rl/data/tasks.jsonl")
    ap.add_argument("--env", default="/data/swe-rl/.env")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--src-tag", default="v1")
    ap.add_argument("--tag", default="sbx2")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="/data/swe-rl/data/sandbox_image_map.json")
    args = ap.parse_args()

    load_env(args.env)
    creds = {
        "registry": need("TCR_REGISTRY"),
        "namespace": need("TCR_NAMESPACE"),
        "username": need("TCR_USERNAME"),
        "password": need("TCR_PASSWORD"),
    }
    base_ref = f"{creds['registry']}/{creds['namespace']}/{BASE_IMAGE_TAG}"
    if not image_present(base_ref):
        code, out = sh(
            f"ctr -n {NS} images pull --platform linux/amd64 "
            f"-u '{creds['username']}:{creds['password']}' {base_ref}",
            secrets=(creds["password"],),
        )
        if code != 0:
            sys.exit(f"[x] 拉取底座失败：{out[-400:]}")

    tasks = [
        json.loads(l) for l in Path(args.tasks).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if args.only:
        tasks = [t for t in tasks if t["task_id"] in set(args.only)]

    out_path = Path(args.out)
    prior: dict = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
    if args.resume:
        n0 = len(tasks)
        tasks = [t for t in tasks if prior.get(t["task_id"], {}).get("status") != "ok"]
        print(f"[resume] 跳过 {n0 - len(tasks)} 题，待处理 {len(tasks)}")
    if not tasks:
        print("✓ 全部已完成")
        return 0

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    if args.jobs <= 1:
        for i, t in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}]", end="")
            results.append(make_one(t, args, creds, base_ref))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(make_one, t, args, creds, base_ref): t["task_id"] for t in tasks}
            done = 0
            for f in as_completed(futs):
                done += 1
                try:
                    rec = f.result()
                except Exception as e:
                    rec = {"task_id": futs[f], "status": "exception", "error": str(e)[:400]}
                results.append(rec)
                print(f"[{done}/{len(futs)}] {rec['task_id']} → {rec.get('status')}", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'=' * 60}\n成功 {len(ok)}/{len(results)}")
    for r in results:
        if r.get("status") != "ok":
            print(f"  ✗ {r['task_id']}: {r.get('status')} {str(r.get('error', ''))[:300]}")

    merged = dict(prior)
    merged.update({r["task_id"]: r for r in results})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for v in merged.values() if v.get("status") == "ok")
    print(f"映射已写入 {args.out}（累计 {n_ok}/{len(merged)} 成功）")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
