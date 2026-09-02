#!/usr/bin/env bash
# 在构建机上制作 AGS 沙箱可用的题目镜像（正规 docker build 路径）
# ==============================================================
#
# ## 为什么用这条路
#
# AGS 沙箱要求容器内运行 envd agent（s6-overlay 托管，提供 files/commands
# 接口与 /health 探针）。SWE-bench 官方镜像没有，直接注册会报
#   FailedOperation.ContainerStart: init command path error
#
# 先前在 GPU 节点上尝试过两种手工方案，均不可行：
#   · ctr images commit          → containerd v1.7 的 ctr 无此子命令
#   · export → 手改 OCI 清单 → import → AGS 报 ImagePrepare: Internal server error
#     （本地 ctr run 能起，但 AGS 平台侧对手工拼的清单有额外校验，黑盒不可定位）
#
# 本脚本改用**标准 docker build**：清单由 docker 生成，与现役可用的沙箱镜像
# 同源同格式，不存在平台拒收问题。构建机（有 docker + 可访问 TCR）是唯一
# 具备该能力的节点。
#
# ## 镜像结构
#
#   FROM <官方题目镜像>              # 保留 /testbed 与 conda 环境，零改动
#   COPY --from=<base> /init /command /package /etc/s6-overlay /usr/bin/envd
#   ENTRYPOINT ["/init"]            # 交给 s6，由它拉起 envd
#
# 多阶段 COPY 让 envd 从已验证可用的 base 镜像原样搬运（静态二进制，无依赖）。
#
# 用法（本机执行，经 DOCKER_HOST=ssh://docker-builder 操作远端 docker）：
#   bash scripts/build_sandbox_images.sh <task_id> [<task_id> ...]
#   bash scripts/build_sandbox_images.sh --all
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
set -a; [ -f .env ] && . ./.env; set +a
: "${TCR_REGISTRY:?缺少 TCR_REGISTRY}"
: "${TCR_NAMESPACE:?缺少 TCR_NAMESPACE}"
: "${TCR_USERNAME:?缺少 TCR_USERNAME}"
: "${TCR_PASSWORD:?缺少 TCR_PASSWORD}"

NS="$TCR_REGISTRY/$TCR_NAMESPACE"
BASE="$NS/swe-synth-base:ubuntu22.04-v1"   # 提供 envd 的现役可用镜像
SRC_TAG="${SRC_TAG:-v1}"                    # 已搬运的官方镜像 tag
OUT_TAG="${OUT_TAG:-sbx}"                   # 融合产物 tag
MIN_FREE_GB="${MIN_FREE_GB:-6}"

# 让 docker 指向构建机（本机无 docker daemon）
export DOCKER_HOST="${DOCKER_HOST:-ssh://docker-builder}"

log() { printf '%s\n' "$*" >&2; }

free_gb() {
  docker run --rm --entrypoint sh "$BASE" -c "df -P / | awk 'NR==2{print int(\$4/1048576)}'" 2>/dev/null \
    || ssh docker-builder "df -P / | awk 'NR==2{print int(\$4/1048576)}'"
}

# 登录 TCR（密码经 stdin，不出现在进程列表/历史里）
printf '%s' "$TCR_PASSWORD" | docker login "$TCR_REGISTRY" -u "$TCR_USERNAME" --password-stdin >/dev/null 2>&1 \
  || { log "[x] TCR 登录失败"; exit 1; }
log "[ok] 已登录 $TCR_REGISTRY"

docker pull "$BASE" >/dev/null 2>&1 || { log "[x] 拉取底座失败：$BASE"; exit 1; }
log "[ok] 底座就绪：swe-synth-base"

# 收集要处理的 task_id
TASKS=()
if [ "${1:-}" = "--all" ]; then
  # 用 while read 而非 mapfile：mapfile 在部分 bash 版本/管道语境下会落到子 shell，
  # 导致数组在主 shell 里为空（实测报 TASKS[@]: unbound variable）
  while IFS= read -r line; do
    [ -n "$line" ] && TASKS+=("$line")
  done < <(python3 -c "
import json
for line in open('data/tasks.jsonl'):
    line = line.strip()
    if line:
        print(json.loads(line)['task_id'])
")
else
  TASKS=("$@")
fi
[ "${#TASKS[@]}" -gt 0 ] || { log "用法：$0 <task_id>... | --all"; exit 1; }

log "待构建 ${#TASKS[@]} 题；输出 tag=$OUT_TAG"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

ok=0; fail=0; skip=0; failed_ids=()
for tid in "${TASKS[@]}"; do
  slug="$(printf '%s' "$tid" | tr 'A-Z' 'a-z' | sed 's/__/-/g')"
  src="$NS/sweb-$slug:$SRC_TAG"
  dst="$NS/sweb-$slug:$OUT_TAG"

  # 幂等：TCR 上已有该 tag 就跳过（构建+推送单题约 5 分钟，重做代价高）
  if [ "${SKIP_EXISTING:-1}" = "1" ] && docker manifest inspect "$dst" >/dev/null 2>&1; then
    log "=== $tid  → 已存在，跳过"
    skip=$((skip+1)); continue
  fi

  log ""
  log "=== $tid"
  log "    源 : $src"
  log "    出 : $dst"

  cat > "$BUILD_DIR/Dockerfile" <<EOF
# syntax=docker/dockerfile:1
FROM $BASE AS agent

FROM $src
# 从现役可用镜像原样搬运 AGS agent（envd 是静态二进制，无动态库依赖）
COPY --from=agent /usr/bin/envd  /usr/bin/envd
COPY --from=agent /init          /init
COPY --from=agent /command       /command
COPY --from=agent /package       /package
COPY --from=agent /etc/s6-overlay /etc/s6-overlay
# 让题目环境的 conda python 默认可用（判据脚本依赖它）
ENV PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:\$PATH
# s6 作为 PID 1，由它拉起 envd —— 这是 AGS 沙箱能接管容器的前提
ENTRYPOINT ["/init"]
CMD []
EOF

  start=$(date +%s)
  if ! docker build --platform linux/amd64 -t "$dst" -f "$BUILD_DIR/Dockerfile" "$BUILD_DIR" 2>&1 | tail -4; then
    log "    ✗ build 失败"
    fail=$((fail+1)); failed_ids+=("$tid"); continue
  fi
  if ! docker push "$dst" 2>&1 | tail -2; then
    log "    ✗ push 失败"
    fail=$((fail+1)); failed_ids+=("$tid"); continue
  fi
  log "    ✓ 完成 $(( $(date +%s) - start ))s"
  ok=$((ok+1))

  # 构建机盘只有 50G，逐题清理本地层，避免塞满
  docker rmi "$dst" "$src" >/dev/null 2>&1 || true
  docker builder prune -f >/dev/null 2>&1 || true
done

log ""
log "============================================================"
log "成功 $ok / 失败 $fail / 跳过 $skip"
if [ "$fail" -gt 0 ]; then
  log "失败：${failed_ids[*]}"
  exit 1
fi
