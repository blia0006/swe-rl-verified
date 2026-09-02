#!/usr/bin/env python3
"""
把 SWE-bench 官方镜像搬运到自有 TCR（在 GPU 宿主机上执行）
=========================================================

为什么必须搬：AGS 沙箱只能拉 TCR 里的镜像，而官方镜像在 Docker Hub。

为什么在 GPU 节点上做（而不是本机）：
  · 节点是 x86_64 原生架构，官方镜像只发 amd64；本机 ARM64 要走模拟层
  · 节点经腾讯云镜像加速器拉 Docker Hub 实测 **45 MiB/s**
  · 节点到 TCR 是内网，推送快

⚠️ 关键坑（实测）：`ctr` **不读** containerd CRI plugin 的 registry mirror 配置
（那份配置是给 kubelet 用的）。所以 `ctr pull docker.io/...` 会直连
registry-1.docker.io 而超时。**必须显式把镜像地址写成 mirror 域名。**

流程（每题一遍，逐个做完立即释放本地层，控制磁盘水位）：
    pull(经 mirror) → tag(TCR) → push(TCR) → 可选 rm 本地镜像

用法：
    python3 pod_sync_images.py --tasks /data/swe-rl/data/tasks.jsonl        # 全部
    python3 pod_sync_images.py --tasks ... --only django__django-16429      # 单题
    python3 pod_sync_images.py --tasks ... --keep-local                     # 不删本地层
    python3 pod_sync_images.py --tasks ... --check                          # 只查 TCR 是否已有

凭证从环境变量读（`/data/swe-rl/.env`），不写入代码，也不落进日志。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

NS = "k8s.io"  # 与 kubelet 共用 namespace，便于 crictl 侧也能看到
DEFAULT_MIRROR = "mirror.ccs.tencentyun.com"

# 并发搬运时保护 `images rm` + `images tag` 这对非原子操作
_TAG_LOCK = threading.Lock()


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


def sh(
    cmd: list[str],
    timeout: int = 3600,
    quiet: bool = False,
    secrets: tuple[str, ...] = (),
) -> tuple[int, str]:
    """执行命令。返回 (exit_code, 合并输出)。

    ⚠️ 脱敏必须基于「凭证值本身」做字符串替换，不能靠"上一个参数是不是 -p"来猜 ——
    `ctr push -u user:pass` 把凭证塞在 `-u` 的**值**里，按参数位置判断会完全漏掉
    （实测已发生：密码被原样打进日志）。
    """
    printable = [_scrub_all(c, secrets) for c in cmd]
    if not quiet:
        print(f"    $ {' '.join(printable)}", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, _scrub_all(out, secrets)
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"


def _scrub_all(text: str, secrets: tuple[str, ...]) -> str:
    """把所有已知凭证值替换成 ***（在打印与返回值两处都做）。"""
    for s in secrets:
        if s and len(s) >= 6:  # 太短的值替换会误伤正常文本
            text = text.replace(s, "***")
    return text


def to_mirror(official: str, mirror: str) -> str:
    """swebench/sweb.eval.x86_64.xxx:latest → <mirror>/swebench/sweb.eval.x86_64.xxx:latest"""
    ref = official
    for prefix in ("docker.io/", "index.docker.io/"):
        if ref.startswith(prefix):
            ref = ref[len(prefix) :]
    return f"{mirror}/{ref}"


def tcr_ref(task_id: str, registry: str, namespace: str) -> str:
    """TCR 目标地址。tag 用 task_id 的规范化形式，保证可读且合法。"""
    slug = task_id.replace("__", "-").lower()
    return f"{registry}/{namespace}/sweb-{slug}:v1"


def image_exists_local(ref: str) -> bool:
    code, out = sh(["ctr", "-n", NS, "images", "ls", "-q"], timeout=120, quiet=True)
    return code == 0 and ref in out.splitlines()


def free_gb() -> float:
    """根文件系统剩余空间（GiB）。搬运十几个镜像可能占上百 GB，必须实时盯。"""
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 2**30


def disk_line() -> str:
    code, out = sh(["df", "-h", "/"], timeout=30, quiet=True)
    return out.splitlines()[-1].strip() if code == 0 and out.splitlines() else "n/a"


def prune_unreferenced() -> float:
    """回收未被任何 tag 引用的镜像层，返回回收的 GiB。

    ⚠️ 这一步不可省略，也不能指望自动 GC：实测 `ctr images rm` 之后磁盘
    **纹丝不动（回收 0GiB）**，因为 containerd 的 GC 是异步且有周期的；
    显式 `content prune references` 才立即回收（实测一次回收 18GiB）。

    本项目里节点只是"搬运中转站" —— 沙箱是从 TCR 拉镜像的，节点本地无需留存。
    因此每题搬完即删 tag + prune，磁盘峰值 ≈ 并发数 × 单题体积，与总题数无关。
    """
    before = free_gb()
    sh(["ctr", "-n", NS, "content", "prune", "references"], timeout=600, quiet=True)
    return free_gb() - before


def sync_one(task: dict, args, creds: dict) -> dict:
    task_id = task["task_id"]
    official = task["official_image"]
    src = to_mirror(official, args.mirror)
    dst = tcr_ref(task_id, creds["registry"], creds["namespace"])
    # 所有需要脱敏的值集中在此，传给每次 sh() 调用
    secrets = (creds["password"], f"{creds['username']}:{creds['password']}")

    print(f"\n=== {task_id}\n    源 : {src}\n    目标: {dst}", flush=True)
    rec = {"task_id": task_id, "tcr_image": dst, "official_image": official}

    if args.check:
        rec["status"] = "local_present" if image_exists_local(dst) else "absent"
        print(f"    → {rec['status']}")
        return rec

    t0 = time.time()
    # 1) pull（经 mirror；ctr 不读 CRI 的 mirror 配置，必须显式写 mirror 域名）
    if image_exists_local(src):
        print("    [1/3] pull 跳过（本地已有）")
    else:
        code, out = sh(
            ["ctr", "-n", NS, "images", "pull", "--platform", "linux/amd64", src],
            timeout=args.timeout,
            secrets=secrets,
        )
        if code != 0:
            rec.update(status="pull_failed", error=out[-500:])
            print(f"    ✗ pull 失败：{out[-300:]}")
            return rec
    t_pull = time.time()

    # 2) tag（先删同名 tag 保证幂等，重跑不报 already exists）
    #    并发下 rm+tag 必须串起来：否则 A 的 rm 可能删掉 B 刚建好的同名 tag
    with _TAG_LOCK:
        sh(["ctr", "-n", NS, "images", "rm", dst], timeout=60, quiet=True, secrets=secrets)
        code, out = sh(
            ["ctr", "-n", NS, "images", "tag", src, dst], timeout=120, secrets=secrets
        )
    if code != 0:
        rec.update(status="tag_failed", error=out[-500:])
        print(f"    ✗ tag 失败：{out[-300:]}")
        return rec

    # 3) push —— 凭证在 `-u` 的值里，sh() 会按值脱敏（打印与返回值两处都做）
    code, out = sh(
        [
            "ctr", "-n", NS, "images", "push",
            "--platform", "linux/amd64",
            "-u", f"{creds['username']}:{creds['password']}",
            dst,
        ],
        timeout=args.timeout,
        secrets=secrets,
    )
    if code != 0:
        rec.update(status="push_failed", error=out[-500:])
        print(f"    ✗ push 失败：{out[-300:]}")
        return rec

    dt = time.time() - t0
    rec.update(
        status="ok",
        elapsed_s=round(dt, 1),
        pull_s=round(t_pull - t0, 1),
        push_s=round(time.time() - t_pull, 1),
    )
    print(f"    ✓ 完成 {dt:.0f}s（pull {rec['pull_s']:.0f}s / push {rec['push_s']:.0f}s）")

    if not args.keep_local:
        # 节点只是中转站：沙箱从 TCR 拉镜像，本地无需留存。
        # 删两个 tag 后必须显式 prune，否则层数据不回收（实测 rm 后回收 0GiB）。
        with _TAG_LOCK:
            sh(["ctr", "-n", NS, "images", "rm", src], timeout=120, quiet=True)
            sh(["ctr", "-n", NS, "images", "rm", dst], timeout=120, quiet=True)
            freed = prune_unreferenced()
        rec["freed_gb"] = round(freed, 1)
        print(f"    ↺ 已清理本地中转层，回收 {freed:.1f}GiB，剩余 {free_gb():.0f}GiB")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/data/swe-rl/data/tasks.jsonl")
    ap.add_argument("--env", default="/data/swe-rl/.env")
    ap.add_argument("--mirror", default=os.environ.get("DOCKERHUB_MIRROR", DEFAULT_MIRROR))
    ap.add_argument("--only", action="append", default=[], help="只处理指定 task_id，可重复")
    ap.add_argument("--keep-local", action="store_true", help="保留本地 mirror tag")
    ap.add_argument("--check", action="store_true", help="只检查不搬运")
    ap.add_argument("--timeout", type=int, default=3600, help="单次 pull/push 超时(秒)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="并发题数。实测单题串行约 480s（pull 与 push 各占一半），"
        "18 题串行需 2.4h；并发主要吃网络带宽，节点 32 核不是瓶颈。"
        "不建议超过 4：磁盘写入与 TCR 侧限流会成为新瓶颈",
    )
    ap.add_argument("--min-free-gb", type=int, default=25, help="磁盘剩余低于此值即停止，防塞满")
    ap.add_argument("--out", default="/data/swe-rl/data/image_map.json")
    args = ap.parse_args()

    load_env(args.env)
    creds = {
        "registry": need("TCR_REGISTRY"),
        "namespace": need("TCR_NAMESPACE"),
        "username": need("TCR_USERNAME"),
        "password": need("TCR_PASSWORD"),
    }

    tasks = [
        json.loads(l) for l in Path(args.tasks).read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    if args.only:
        tasks = [t for t in tasks if t["task_id"] in set(args.only)]
    if not tasks:
        sys.exit("[x] 没有匹配的题目")

    print(f"共 {len(tasks)} 题；mirror={args.mirror}；并发={args.jobs}")
    print(f"磁盘：{disk_line()}（低于 {args.min_free_gb}GB 剩余时自动停止）")

    results: list[dict] = []
    if args.jobs <= 1 or args.check:
        for i, t in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}]", end="")
            if not args.check and free_gb() < args.min_free_gb:
                print(f"\n✗ 磁盘剩余不足 {args.min_free_gb}GB，停止后续搬运")
                break
            results.append(sync_one(t, args, creds))
            print(f"    磁盘: {disk_line()}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 用线程池即可：每个任务的主体是 subprocess 等待（IO 阻塞），不受 GIL 限制
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {}
            for t in tasks:
                if free_gb() < args.min_free_gb:
                    print(f"✗ 磁盘剩余不足 {args.min_free_gb}GB，未提交剩余任务")
                    break
                futures[pool.submit(sync_one, t, args, creds)] = t["task_id"]
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    rec = fut.result()
                except Exception as e:  # 单题异常不拖垮整批
                    rec = {"task_id": futures[fut], "status": "exception", "error": str(e)[:400]}
                results.append(rec)
                print(
                    f"[{done}/{len(futures)}] {rec['task_id']} → {rec.get('status')}"
                    f"    磁盘: {disk_line()}",
                    flush=True,
                )

    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'=' * 60}\n成功 {len(ok)}/{len(results)}")
    for r in results:
        if r.get("status") != "ok":
            print(f"  ✗ {r['task_id']}: {r.get('status')} {str(r.get('error', ''))[:200]}")
    if ok:
        avg = sum(r.get("elapsed_s", 0) for r in ok) / len(ok)
        print(f"平均单题 {avg:.0f}s")

    # 与已有映射合并，便于分批搬运时逐步累积
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if out_path.exists():
        try:
            merged = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update({r["task_id"]: r for r in results})
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"映射已写入 {args.out}（累计 {len(merged)} 题）")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
