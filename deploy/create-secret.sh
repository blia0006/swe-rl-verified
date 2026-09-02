#!/usr/bin/env bash
# 创建 Pod 所需的云凭证 Secret
# ============================
#
# 为什么用 Secret 而不是直接写进 Pod YAML：
#   · YAML 会进 git（本仓库是 public），凭证绝不能出现在里面
#   · Secret 由集群管理，Pod 通过 envFrom 注入，YAML 里只留引用名
#
# 本脚本从 /data/swe-rl/.env 读取凭证并创建 Secret，
# **不打印任何凭证值**（仅显示字段名）。
#
# 用法（在 GPU 节点上执行，节点位于 VPC 内可直连集群内网 APIServer）：
#     bash /data/swe-rl/deploy/create-secret.sh
set -uo pipefail

ENV_FILE="${ENV_FILE:-/data/swe-rl/.env}"
SECRET_NAME="${SECRET_NAME:-swe-rl-creds}"
NAMESPACE="${NAMESPACE:-default}"

[ -f "$ENV_FILE" ] || { echo "[x] 找不到 $ENV_FILE"; exit 1; }
command -v kubectl >/dev/null || { echo "[x] 未安装 kubectl"; exit 1; }

# 只注入 reward function 与沙箱调用真正需要的字段（最小权限原则）
KEYS="
TENCENTCLOUD_SECRET_ID
TENCENTCLOUD_SECRET_KEY
TENCENTCLOUD_REGION
AGS_ROLE_ARN
AGS_TOOL_NAME
AGS_REGION
E2B_API_KEY
E2B_DOMAIN
TCR_REGISTRY
TCR_NAMESPACE
COS_REGION
COS_BUCKET
"

args=()
missing=()
for k in $KEYS; do
  # 从 .env 取值；grep -m1 保证只取第一次出现
  line=$(grep -m1 "^${k}=" "$ENV_FILE" || true)
  if [ -z "$line" ]; then
    missing+=("$k")
    continue
  fi
  val="${line#*=}"
  if [ -z "$val" ]; then
    missing+=("$k")
    continue
  fi
  args+=("--from-literal=${k}=${val}")
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "[!] 以下字段缺失或为空，将不注入：${missing[*]}"
fi
if [ "${#args[@]}" -eq 0 ]; then
  echo "[x] 没有任何可注入的字段"
  exit 1
fi

# 幂等：先删旧的再建（Secret 不支持原地改字段集合）
kubectl -n "$NAMESPACE" delete secret "$SECRET_NAME" >/dev/null 2>&1 || true
kubectl -n "$NAMESPACE" create secret generic "$SECRET_NAME" "${args[@]}" >/dev/null

echo "[ok] 已创建 Secret ${NAMESPACE}/${SECRET_NAME}，含字段："
kubectl -n "$NAMESPACE" get secret "$SECRET_NAME" -o jsonpath='{.data}' \
  | tr ',' '\n' | grep -oE '"[A-Z_]+"' | tr -d '"' | sed 's/^/    /'
