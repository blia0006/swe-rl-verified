#!/usr/bin/env python3
"""
镜像融合：给 SWE-bench 官方镜像装上 AGS 沙箱 agent（envd）
=========================================================

## 为什么需要这一步（实测踩坑）

把官方镜像直接注册成 AGS 沙箱工具会失败：

    code=FailedOperation.ContainerStart   message: init command path error

根因：AGS 沙箱要求容器内跑 **envd agent**（由 s6-overlay 托管为 PID 1），
它提供 `sbx.files.*` / `sbx.commands.*` 接口与 `/health:49983` 健康探针。
SWE-bench 官方镜像只有题目环境（`/testbed` + conda），没有这套 agent。

## 为什么可以融合（前提已实测确认）

从账号里既有可用的 base 镜像（`swe-synth-base:ubuntu22.04-v1`）探得：

| 组件 | 事实 | 意义 |
|---|---|---|
| `/usr/bin/envd` | **静态二进制**（`ldd` → not a dynamic executable），16.7MB | 无动态库依赖，可跨镜像直接拷贝 |
| `/init`、`/command`、`/package`、`/etc/s6-overlay` | s6-overlay 运行时 | 整目录拷贝即可 |
| 双方基础镜像 | 均 Ubuntu 22.04 系 | ABI 兼容，无需重编译 |

融合是**纯文件叠加**：把 agent 目录作为**新增的一层**追加到官方镜像上，
并改 ENTRYPOINT 为 `/init`。**完全不动 `/testbed` 与 conda 环境**。

## 实现路径的选择（前两种都试过，不可行）

| 方案 | 结论 |
|---|---|
| `docker build` | ✗ 节点无 docker/buildkit，且装它要下 GitHub release（节点不通） |
| `ctr images commit` | ✗ containerd v1.7.28 的 ctr **没有 commit 子命令**（实测 "No help topic"） |
| **`ctr images export` → 手工加层 → `import`** | ✓ 采用。export 1.4GB 仅 7.7s，纯标准库操作 OCI 归档 |

OCI 镜像本质是「一组 tar.gz 层 + JSON 清单」，加一层就是：
    追加 layer blob → 在 manifest.layers 里加一项 → 在 config.rootfs.diff_ids 加一项
    → 改 config.Entrypoint → 重算受影响的 digest → 更新 index.json

用法（在 GPU 宿主机执行）：
    python3 pod_build_sandbox_images.py --tasks /data/swe-rl/data/tasks.jsonl
    python3 pod_build_sandbox_images.py --tasks ... --only <task_id> --resume
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
BASE_IMAGE_TAG = "swe-synth-base:ubuntu22.04-v1"  # envd 的来源
WORK_ROOT = Path("/data/swe-rl/build")

# 必须从 base 镜像搬运的 agent 组件（逐项实测确认，缺一不可）
AGENT_PATHS = ["usr/bin/envd", "init", "command", "package", "etc/s6-overlay"]

AGENT_DIR = WORK_ROOT / "agent_rootfs"   # 解包后的 agent 文件树
AGENT_LAYER = WORK_ROOT / "agent_layer.tar.gz"  # 可直接追加的镜像层

_LOCK = threading.Lock()

MEDIA_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MEDIA_MANIFEST_D2 = "application/vnd.docker.distribution.manifest.v2+json"
MEDIA_INDEX = "application/vnd.oci.image.index.v1+json"
MEDIA_INDEX_D2 = "application/vnd.docker.distribution.manifest.list.v2+json"
MEDIA_LAYER_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
MEDIA_LAYER_D2 = "application/vnd.docker.image.rootfs.diff.tar.gzip"


# ------------------------------------------------------------------ 基础设施

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


def _scrub(text: str, secrets: tuple[str, ...]) -> str:
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "***")
    return text


def _denoise(text: str) -> str:
    """滤掉 containerd 的 DEPRECATION 噪音，否则真实报错会被挤出截断窗口。"""
    return "\n".join(
        ln
        for ln in text.splitlines()
        if "DEPRECATION" not in ln and "is deprecated since containerd" not in ln
    ).strip()


def sh(
    cmd: str, timeout: int = 1800, quiet: bool = False, secrets: tuple[str, ...] = ()
) -> tuple[int, str]:
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
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# ------------------------------------------------------------------ 步骤 1：制作 agent 层

def build_agent_layer(base_ref: str, creds: dict) -> tuple[str, int, str]:
    """从 base 镜像提取 agent 并制成一个可追加的镜像层。

    返回 (压缩后 digest, 压缩后大小, 未压缩 diff_id)。
    只需做一次，所有题目共用同一层 —— 这也让各题镜像在 TCR 侧共享该层。
    """
    secrets = (creds["password"], f"{creds['username']}:{creds['password']}")
    meta_path = WORK_ROOT / "agent_layer.json"
    if AGENT_LAYER.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[agent] 复用已有层 {meta['digest'][:19]}…（{meta['size'] / 2**20:.1f}MB）")
        return meta["digest"], meta["size"], meta["diff_id"]

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[agent] 从 base 镜像提取 AGS agent：{base_ref}")
    if not image_present(base_ref):
        code, out = sh(
            f"ctr -n {NS} images pull --platform linux/amd64 "
            f"-u '{creds['username']}:{creds['password']}' {base_ref}",
            secrets=secrets,
        )
        if code != 0:
            sys.exit(f"[x] 拉取 base 镜像失败：{out[-500:]}")

    # 用 images mount 直接挂载 rootfs 读取文件 —— 比起容器再 exec 更快更稳
    mnt = WORK_ROOT / "base_mnt"
    if AGENT_DIR.exists():
        shutil.rmtree(AGENT_DIR)
    AGENT_DIR.mkdir(parents=True)
    mnt.mkdir(parents=True, exist_ok=True)

    sh(f"ctr -n {NS} images unmount {mnt} 2>/dev/null", quiet=True)
    code, out = sh(f"ctr -n {NS} images mount --rw {base_ref} {mnt}", timeout=300)
    if code != 0:
        sys.exit(f"[x] 挂载 base 镜像失败：{out[-500:]}")
    try:
        for rel in AGENT_PATHS:
            srcp = mnt / rel
            if not srcp.exists():
                sys.exit(f"[x] base 镜像内缺少 {rel}（agent 组件不完整，无法融合）")
            dstp = AGENT_DIR / rel
            dstp.parent.mkdir(parents=True, exist_ok=True)
            if srcp.is_dir():
                shutil.copytree(srcp, dstp, symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy2(srcp, dstp, follow_symlinks=False)
            print(f"    + {rel}")
    finally:
        sh(f"ctr -n {NS} images unmount {mnt}", timeout=120, quiet=True)

    # 打成 gzip 层。需要同时算未压缩内容的 sha256（diff_id）与压缩后的 digest
    raw = WORK_ROOT / "agent_layer.tar"
    with tarfile.open(raw, "w") as tf:
        for rel in AGENT_PATHS:
            tf.add(AGENT_DIR / rel, arcname=rel, recursive=True)
    diff_id, raw_size = sha256_file(raw)
    with open(raw, "rb") as fi, gzip.open(AGENT_LAYER, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    digest, size = sha256_file(AGENT_LAYER)
    raw.unlink(missing_ok=True)

    meta = {
        "digest": f"sha256:{digest}",
        "size": size,
        "diff_id": f"sha256:{diff_id}",
        "raw_size": raw_size,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[agent] ✓ 层已就绪：{meta['digest'][:19]}… "
        f"压缩 {size / 2**20:.1f}MB / 原始 {raw_size / 2**20:.1f}MB"
    )
    return meta["digest"], size, meta["diff_id"]


# ------------------------------------------------------------------ 步骤 2：往 OCI 归档加层

def blob_path(root: Path, digest: str) -> Path:
    return root / "blobs" / "sha256" / digest.split(":", 1)[1]


def write_blob(root: Path, data: bytes) -> tuple[str, int]:
    """把 JSON/二进制写为内容寻址 blob，返回 (digest, size)。"""
    d = hashlib.sha256(data).hexdigest()
    p = blob_path(root, f"sha256:{d}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return f"sha256:{d}", len(data)


def read_json_blob(root: Path, digest: str) -> dict:
    return json.loads(blob_path(root, digest).read_bytes())


def inject_layer_into_archive(
    root: Path,
    layer_digest: str,
    layer_size: int,
    diff_id: str,
    entrypoint: list[str],
) -> None:
    """在已解包的 OCI 归档目录里追加一层并改 ENTRYPOINT。

    OCI 的 digest 是内容寻址：改了 config 就要改 manifest 里指向它的 digest，
    改了 manifest 又要改 index。因此必须**自底向上**重算：
        config → manifest → index
    """
    index = json.loads((root / "index.json").read_text())

    def patch_manifest(man_digest: str) -> tuple[str, int]:
        man = read_json_blob(root, man_digest)

        # 多架构 index：递归进去只改 amd64；同时剔除悬空子项
        if man.get("mediaType") in (MEDIA_INDEX, MEDIA_INDEX_D2) or "manifests" in man:
            new_children = []
            for child in man["manifests"]:
                plat = child.get("platform", {})
                arch = plat.get("architecture")

                # ⚠️ 必须剔除 arch=unknown/unknown 的子项。
                # 官方 SWE-bench 镜像带有 SBOM / attestation 描述符，其 platform 标为
                # unknown/unknown；用 `--platform linux/amd64` 导出时它们的 blob
                # **不会被包含**，于是清单里留下悬空引用，push 时报
                #   `ctr: content digest sha256:...: not found`
                # 这些元数据对运行镜像毫无作用，直接丢掉。
                if arch in (None, "unknown") or plat.get("os") == "unknown":
                    if not blob_path(root, child["digest"]).exists():
                        print(f"    · 剔除悬空子项 arch={arch} {child['digest'][:19]}…")
                        continue

                if arch not in (None, "amd64"):
                    new_children.append(child)
                    continue
                nd, ns = patch_manifest(child["digest"])
                c = dict(child)
                c["digest"], c["size"] = nd, ns
                new_children.append(c)
            man["manifests"] = new_children
            return write_blob(root, json.dumps(man, separators=(",", ":")).encode())

        # 单架构 manifest：改 config 与 layers
        cfg_desc = man["config"]
        cfg = read_json_blob(root, cfg_desc["digest"])

        cfg.setdefault("rootfs", {}).setdefault("diff_ids", []).append(diff_id)
        for key in ("config", "container_config"):
            if key in cfg and isinstance(cfg[key], dict):
                cfg[key]["Entrypoint"] = entrypoint
                # 清掉原 CMD：官方镜像的 CMD 与 s6 入口不兼容
                cfg[key]["Cmd"] = None
        cfg.setdefault("history", []).append(
            {
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "created_by": "swe-rl: inject AGS envd agent layer",
            }
        )
        new_cfg_digest, new_cfg_size = write_blob(
            root, json.dumps(cfg, separators=(",", ":")).encode()
        )

        man["config"] = {
            "mediaType": cfg_desc.get("mediaType", "application/vnd.oci.image.config.v1+json"),
            "digest": new_cfg_digest,
            "size": new_cfg_size,
        }
        # 层的 mediaType 要与既有层保持同一体系（docker v2 vs oci）
        existing_type = man["layers"][-1]["mediaType"] if man.get("layers") else MEDIA_LAYER_GZIP
        layer_type = MEDIA_LAYER_D2 if "docker" in existing_type else MEDIA_LAYER_GZIP
        man["layers"].append(
            {"mediaType": layer_type, "digest": layer_digest, "size": layer_size}
        )
        return write_blob(root, json.dumps(man, separators=(",", ":")).encode())

    new_manifests = []
    for m in index["manifests"]:
        nd, ns = patch_manifest(m["digest"])
        mm = dict(m)
        mm["digest"], mm["size"] = nd, ns
        new_manifests.append(mm)
    index["manifests"] = new_manifests
    (root / "index.json").write_text(json.dumps(index, separators=(",", ":")))


# ------------------------------------------------------------------ 步骤 3：单题融合

def build_one(task: dict, args, creds: dict, layer: tuple[str, int, str]) -> dict:
    task_id = task["task_id"]
    src = tcr_ref(task_id, creds["registry"], creds["namespace"], args.src_tag)
    dst = tcr_ref(task_id, creds["registry"], creds["namespace"], args.tag)
    secrets = (creds["password"], f"{creds['username']}:{creds['password']}")
    layer_digest, layer_size, diff_id = layer
    rec = {"task_id": task_id, "src": src, "sandbox_image": dst}
    print(f"\n=== {task_id}\n    源 : {src}\n    出 : {dst}", flush=True)

    t0 = time.time()
    if not image_present(src):
        code, out = sh(
            f"ctr -n {NS} images pull --platform linux/amd64 "
            f"-u '{creds['username']}:{creds['password']}' {src}",
            timeout=args.timeout,
            secrets=secrets,
        )
        if code != 0:
            rec.update(status="pull_failed", error=out[-500:])
            print(f"    ✗ pull 失败：{out[-300:]}")
            return rec

    work = WORK_ROOT / f"w_{task_id.replace('__', '-').lower()}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    tar_path = work / "img.tar"

    try:
        code, out = sh(
            f"ctr -n {NS} images export --platform linux/amd64 {tar_path} {src}",
            timeout=args.timeout,
        )
        if code != 0:
            rec.update(status="export_failed", error=out[-500:])
            print(f"    ✗ export 失败：{out[-300:]}")
            return rec

        extract = work / "oci"
        extract.mkdir()
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract)  # 内容来自本地 ctr 导出，非不可信来源
        tar_path.unlink(missing_ok=True)

        # 放入 agent 层的 blob，并改写清单
        shutil.copy2(AGENT_LAYER, blob_path(extract, layer_digest))
        inject_layer_into_archive(
            extract, layer_digest, layer_size, diff_id, entrypoint=["/init"]
        )

        new_tar = work / "fused.tar"
        with tarfile.open(new_tar, "w") as tf:
            for item in sorted(extract.iterdir()):
                tf.add(item, arcname=item.name, recursive=True)

        # 不能加 --no-unpack：实测会导致新 manifest/config blob 未进入 content store，
        # 随后 push 报 `content digest sha256:...: not found`。
        # 加 --platform 让 import 只处理 amd64，避免多架构索引残留悬空引用。
        code, out = sh(
            f"ctr -n {NS} images import --platform linux/amd64 "
            f"--index-name {dst} {new_tar}",
            timeout=args.timeout,
        )
        if code != 0:
            rec.update(status="import_failed", error=out[-600:])
            print(f"    ✗ import 失败：{out[-400:]}")
            return rec
        new_tar.unlink(missing_ok=True)

        code, out = sh(
            f"ctr -n {NS} images push --platform linux/amd64 "
            f"-u '{creds['username']}:{creds['password']}' {dst}",
            timeout=args.timeout,
            secrets=secrets,
        )
        if code != 0:
            rec.update(status="push_failed", error=out[-600:])
            print(f"    ✗ push 失败：{out[-400:]}")
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
    ap.add_argument("--src-tag", default="v1", help="已搬运的官方镜像 tag")
    ap.add_argument("--tag", default="sbx", help="融合后镜像 tag")
    ap.add_argument("--jobs", type=int, default=2)
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

    tasks = [
        json.loads(l)
        for l in Path(args.tasks).read_text(encoding="utf-8").splitlines()
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
        print(f"[resume] 跳过已完成 {n0 - len(tasks)} 题，待处理 {len(tasks)}")
    if not tasks:
        print("✓ 全部已完成")
        return 0

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    layer = build_agent_layer(base_ref, creds)

    results: list[dict] = []
    if args.jobs <= 1:
        for i, t in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}]", end="")
            results.append(build_one(t, args, creds, layer))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(build_one, t, args, creds, layer): t["task_id"] for t in tasks}
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
