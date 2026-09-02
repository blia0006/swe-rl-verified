#!/usr/bin/env bash
# 部署训练 Pod 到 TKE
# ==================
#
# gpu-pod.yaml 中的节点 IP 用占位符 `__GPU_NODE_IP__` 表示（本仓库 public，
# 不写入真实资源标识）。本脚本在部署时从环境变量注入真实值。
#
# 用法（在 GPU 节点上执行；节点位于 VPC 内，可直连集群内网 APIServer）：
#     bash /data/swe-rl/deploy/apply.sh            # 部署
#     bash /data/swe-rl/deploy/apply.sh --dry-run  # 只渲染不提交
#     bash /data/swe-rl/deploy/apply.sh --delete   # 清理
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-/data/swe-rl/.env}"
YAML="$HERE/gpu-pod.yaml"
RENDERED="/tmp/swe-rl-pod-rendered.yaml"

[ -f "$YAML" ] || { echo "[x] 找不到 $YAML"; exit 1; }
command -v kubectl >/dev/null || { echo "[x] 未安装 kubectl"; exit 1; }

# 节点内网 IP：优先取环境变量，否则从本机网卡自动探测
if [ -z "${GPU_NODE_IP:-}" ]; then
  GPU_NODE_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^10\.|^172\.|^192\.168\.' | head -1)
fi
[ -n "${GPU_NODE_IP:-}" ] || { echo "[x] 无法确定节点内网 IP，请显式设置 GPU_NODE_IP"; exit 1; }

# kubeconfig：优先用项目内的（由本地投放），回退到默认位置
for cand in /data/swe-rl/.kube/config "$HOME/.kube/config"; do
  if [ -f "$cand" ]; then
    export KUBECONFIG="$cand"
    break
  fi
done
[ -n "${KUBECONFIG:-}" ] || { echo "[x] 找不到 kubeconfig"; exit 1; }
echo "[i] KUBECONFIG=$KUBECONFIG"
echo "[i] 节点内网 IP=$GPU_NODE_IP"

# 连通性与权限自检 —— 早失败好过部署到一半
if ! kubectl get nodes >/dev/null 2>&1; then
  echo "[x] 无法访问集群。常见原因："
  echo "    · kubeconfig 凭证已失效（控制台重新下载）"
  echo "    · 该 kubeconfig 是外网版但集群未开公网访问（应下载内网版）"
  exit 1
fi
echo "[ok] 集群可访问"

sed "s|__GPU_NODE_IP__|${GPU_NODE_IP}|g" "$YAML" > "$RENDERED"

case "${1:-apply}" in
  --dry-run)
    echo "--- 渲染结果（未提交）---"
    kubectl apply -f "$RENDERED" --dry-run=client
    ;;
  --delete)
    kubectl delete -f "$RENDERED" --ignore-not-found
    echo "[ok] 已清理"
    ;;
  *)
    # 凭证 Secret 先行（Pod 通过 envFrom 引用）
    if [ -f "$HERE/create-secret.sh" ]; then
      bash "$HERE/create-secret.sh" || echo "[!] Secret 创建失败，Pod 可能缺少凭证"
    fi
    kubectl apply -f "$RENDERED"
    echo ""
    echo "[i] 等待 Pod 就绪（首次需拉镜像，若节点已缓存则很快）…"
    kubectl wait --for=condition=Ready pod/swe-rl-train --timeout=600s 2>&1 || true
    kubectl get pod swe-rl-train -o wide
    echo ""
    echo "后续操作："
    echo "  kubectl exec -it swe-rl-train -- bash"
    echo "  kubectl logs -f swe-rl-train"
    ;;
esac
