#!/usr/bin/env bash
# 在 GPU 宿主机上起训练容器（不经 k8s / 不用 pod）
# ================================================
#
# 为什么不用 pod：
#   · 上一轮 pod 的 /workspace **没有 host 挂载**，22G checkpoint 全在容器
#     可写层，删 pod 即全丢
#   · kubeconfig 会过期（上一轮的已失效，kubectl 完全连不上）
#   · 本轮全程用云助手从本地终端监督，pod 那层只是多余的间接层
#
# 本脚本用 containerd 原生的 ctr 起容器：
#   --net-host   直接用宿主机网络（要访问 TCR / AGS 沙箱 / COS）
#   --gpus 0     挂载 GPU
#   --mount      工作目录落宿主机 /data/swe-rl，**容器可随时重建、数据不丢**
#   /dev/shm     必须放大：容器默认仅 64MB，NCCL 每 rank 需约 31.5MB，
#                不足会导致训练零步崩溃，且运行期无法补救
#
# 用法（由 scripts/node.py 投放后在宿主机执行）：
#   bash start_train_container.sh preflight        # 冒烟测试
#   bash start_train_container.sh train            # 启动训练
#   bash start_train_container.sh shell '<cmd>'    # 在容器内执行任意命令
set -uo pipefail

NS=k8s.io
IMAGE="${IMAGE:-docker.io/verlai/verl:vllm011.latest}"
NAME="${NAME:-swe-rl-train}"
DATA=/data/swe-rl
ACTION="${1:-preflight}"
shift || true

# 清掉同名残留容器（幂等，重复执行不报错）
cleanup() {
  ctr -n $NS task kill -s SIGKILL "$NAME" >/dev/null 2>&1 || true
  sleep 1
  ctr -n $NS task rm -f "$NAME" >/dev/null 2>&1 || true
  ctr -n $NS containers rm "$NAME" >/dev/null 2>&1 || true
}

run_in_container() {
  local inner="$1"
  cleanup
  ctr -n $NS run --rm --net-host --gpus 0 \
    --env PYTHONUNBUFFERED=1 \
    --env WORKDIR=$DATA \
    --env PYTHONPATH=$DATA:$DATA/pylibs \
    --env NCCL_SHM_DISABLE=1 \
    --env NCCL_P2P_DISABLE=1 \
    --env TOKENIZERS_PARALLELISM=false \
    --mount type=bind,src=$DATA,dst=$DATA,options=rbind:rw \
    --mount type=tmpfs,dst=/dev/shm,options=rw:nosuid:nodev:size=4g \
    --cwd $DATA \
    "$IMAGE" "$NAME" \
    bash -lc "$inner"
}

case "$ACTION" in
  preflight)
    echo "=== 训练前冒烟测试 ==="
    run_in_container "python3 $DATA/scripts/preflight_gpu.py ${*:-}"
    ;;
  train)
    echo "=== 启动 GRPO 训练 ==="
    run_in_container "bash $DATA/scripts/run_grpo_training.sh"
    ;;
  shell)
    run_in_container "${1:-echo 需要提供命令}"
    ;;
  stop)
    cleanup
    echo "已清理容器 $NAME"
    ;;
  *)
    echo "用法：$0 {preflight|train|shell <cmd>|stop}"
    exit 1
    ;;
esac
