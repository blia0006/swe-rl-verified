#!/usr/bin/env python3
"""
定位 AGS `FailedOperation.ImagePrepare` 的真实原因（对照实验）
============================================================

背景：给官方 SWE-bench 镜像装上 envd agent 后，AGS 仍报
`ImagePrepare: Internal server error`。已排除清单格式问题
（docker build 产出的清单是标准的，仍然失败）。

因此改用**控制变量法**，逐个排除剩余可能：

| # | 假设 | 对照镜像 | 若成功说明 |
|---|---|---|---|
| A | 平台只是偶发故障 | 现役可用的 base 原样 | 基线，必须成功 |
| B | 镜像**体积**超限 | base + 一层 ~2GB 填充数据 | 体积是门槛 |
| C | 镜像**层数**超限 | base + 10 个空层 | 层数是门槛 |
| D | 融合镜像本身有问题 | 我们做的 sbx 镜像 | —— |

只有把变量隔离开，才能知道该改哪里；否则只能盲试。

用法：
    python3 experiments/diagnose_image_prepare.py --build   # 在构建机造对照镜像
    python3 experiments/diagnose_image_prepare.py --test    # 逐个起沙箱测试
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")


def ns() -> str:
    return f"{os.environ['TCR_REGISTRY']}/{os.environ['TCR_NAMESPACE']}"


def sh_builder(script: str, timeout: int = 3600) -> tuple[int, str]:
    """在构建机上执行脚本（经 SSH，密码不入参数）。"""
    p = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "docker-builder", "bash -s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def build_probes() -> int:
    """在构建机造三个对照镜像。"""
    base = f"{ns()}/swe-synth-base:ubuntu22.04-v1"
    user = os.environ["TCR_USERNAME"]
    pwd = os.environ["TCR_PASSWORD"]
    reg = os.environ["TCR_REGISTRY"]

    script = f"""
set -e
printf '%s' '{pwd}' | docker login {reg} -u '{user}' --password-stdin >/dev/null
docker pull {base} >/dev/null
d=$(mktemp -d); cd "$d"

echo "=== A: base 原样重打 tag（基线）==="
docker tag {base} {ns()}/probe-a-plain:v1
docker push {ns()}/probe-a-plain:v1 2>&1 | tail -1

echo "=== B: base + 2GB 填充层（测体积门槛）==="
cat > Dockerfile.b <<'EOF'
FROM {base}
RUN dd if=/dev/urandom of=/big.bin bs=1M count=2048 2>/dev/null && ls -lh /big.bin
EOF
docker build --platform linux/amd64 -t {ns()}/probe-b-big:v1 -f Dockerfile.b . 2>&1 | tail -2
docker push {ns()}/probe-b-big:v1 2>&1 | tail -1

echo "=== C: base + 12 个空层（测层数门槛）==="
{{
echo "FROM {base}"
for i in $(seq 1 12); do echo "RUN touch /layer_$i"; done
}} > Dockerfile.c
docker build --platform linux/amd64 -t {ns()}/probe-c-layers:v1 -f Dockerfile.c . 2>&1 | tail -2
docker push {ns()}/probe-c-layers:v1 2>&1 | tail -1

echo "=== 层数与体积对照 ==="
for img in {base} {ns()}/probe-b-big:v1 {ns()}/probe-c-layers:v1 {ns()}/sweb-scikit-learn-scikit-learn-14141:sbx; do
  docker pull "$img" >/dev/null 2>&1 || true
  n=$(docker inspect "$img" --format '{{{{len .RootFS.Layers}}}}' 2>/dev/null || echo '?')
  s=$(docker inspect "$img" --format '{{{{.Size}}}}' 2>/dev/null || echo 0)
  printf '%-70s layers=%s size=%.2fGB\\n' "$img" "$n" "$(echo "$s/1073741824" | bc -l)"
done
docker rmi {ns()}/probe-b-big:v1 {ns()}/probe-c-layers:v1 >/dev/null 2>&1 || true
docker builder prune -f >/dev/null 2>&1 || true
rm -rf "$d"
"""
    code, out = sh_builder(script)
    print(out)
    return code


def test_probes(names: list[str]) -> int:
    from clients.ags import AGSClient

    tool_name = os.environ.get("AGS_TOOL_NAME") or os.environ.get("SWE_SYNTH_SHARED_TOOL", "")
    tool = AGSClient().find_tool(tool_name)
    if not tool:
        sys.exit(f"[x] 找不到沙箱工具 {tool_name}")
    tool_id = tool["tool_id"]

    print(f"{'镜像':52s} {'结果':10s} 说明")
    print("-" * 100)
    verdicts: dict[str, str] = {}
    for name in names:
        image = f"{ns()}/{name}"
        ags = AGSClient()
        t0 = time.time()
        try:
            # max_wait 调小：这里只关心"能否起来"，失败要快速返回而非重试到超时
            inst, _ = ags.start_instance(tool_id, image_override=image, max_wait=90)
            dt = time.time() - t0
            verdicts[name] = "OK"
            print(f"{name:52s} {'✓ 成功':10s} {dt:.1f}s  instance={inst[:16]}")
            try:
                ags.stop_instance(inst)
            except Exception:
                pass
        except Exception as e:
            msg = str(e)
            kind = (
                "ImagePrepare" if "ImagePrepare" in msg
                else "ContainerStart" if "ContainerStart" in msg
                else "Other"
            )
            verdicts[name] = kind
            print(f"{name:52s} {'✗ ' + kind:10s} {str(e)[-120:]}")

    print("\n=== 结论推断 ===")
    a = verdicts.get("probe-a-plain:v1")
    b = verdicts.get("probe-b-big:v1")
    c = verdicts.get("probe-c-layers:v1")
    if a and a != "OK":
        print("基线 A 都失败 → AGS 平台侧当前异常，或该工具已不可用；先排除环境问题再谈其他")
    else:
        if b and b != "OK":
            print("→ B(2GB 填充) 失败：**镜像体积**是门槛。官方 SWE-bench 镜像 1~2GB 可能超限")
        if c and c != "OK":
            print("→ C(12 层) 失败：**层数**是门槛")
        if b == "OK" and c == "OK":
            print("→ 体积与层数均非门槛；问题出在官方镜像自身的某个特性（如 OS 基础层差异）")
    return 0


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="在构建机造对照镜像")
    ap.add_argument("--test", action="store_true", help="起沙箱逐个测试")
    args = ap.parse_args()
    if args.build:
        return build_probes()
    if args.test:
        return test_probes([
            "probe-a-plain:v1",
            "probe-b-big:v1",
            "probe-c-layers:v1",
            "sweb-scikit-learn-scikit-learn-14141:sbx",
        ])
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
