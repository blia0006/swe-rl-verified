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

# containerd namespace：**必须用 default，不能用 k8s.io**
#
# k8s.io 这个 namespace 由 kubelet 管理，它的镜像 GC 会回收「没有被任何 Pod
# 引用」的镜像。我们用 ctr 直接起容器、不经 k8s，所以训练镜像在 kubelet 看来
# 永远是"没人用"的，随时会被清掉 —— 实测跑了一整天后突然报
#     ctr: image "docker.io/verlai/verl:vllm011.latest": not found
# 而磁盘还有 65G 空闲，纯粹是被 GC 掉了。
#
# 放在 default namespace 后 kubelet 不会碰它，镜像可长期驻留。
NS="${CTR_NS:-default}"
# 镜像地址走腾讯云内网加速：Docker Hub 直连实测 i/o timeout，
# 而 mirror.ccs.tencentyun.com 拉 10.7GB 只需约 60s（177 MiB/s）。
IMAGE="${IMAGE:-mirror.ccs.tencentyun.com/verlai/verl:vllm011.latest}"
NAME="${NAME:-swe-rl-train}"
DATA=/data/swe-rl
ACTION="${1:-preflight}"
shift || true

# 清掉同名残留容器（幂等，重复执行不报错）
#
# ⚠️ 顺序与等待都有必要：ctr 的 task kill 是异步的，紧跟着 task rm 会撞上
# "cannot delete a deleted process: not found"，而该错误会让整条 `&&` 链
# 提前中止 —— 实测导致冒烟测试 2 秒就"失败"，看起来像显存不足，
# 其实训练根本没启动。因此每步都独立执行、失败即忽略，并等到确认无残留。
cleanup() {
  ctr -n $NS task kill -s SIGKILL "$NAME" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    ctr -n $NS task ls 2>/dev/null | grep -q "^$NAME " || break
    sleep 1
  done
  ctr -n $NS task rm -f "$NAME" >/dev/null 2>&1 || true
  ctr -n $NS containers rm "$NAME" >/dev/null 2>&1 || true
  # 确认容器记录真的消失，否则后续 run 会报 "already exists"
  for _ in $(seq 1 10); do
    ctr -n $NS containers ls -q 2>/dev/null | grep -qx "$NAME" || break
    sleep 1
    ctr -n $NS containers rm "$NAME" >/dev/null 2>&1 || true
  done
}

run_in_container() {
  local inner="$1"
  # 镜像可能被 kubelet 的 GC 回收（我们用 ctr 起容器，镜像在 k8s 看来未被引用）。
  # 实测跑了一整天后突然报 image not found，磁盘却还有 65G 空闲。
  if ! ctr -n $NS i ls -q 2>/dev/null | grep -qx "$IMAGE"; then
    echo "[x] 训练镜像不在位：$IMAGE"
    echo "    请先执行：bash scripts/ensure_train_image.sh（拉取约 10GB，需数分钟）"
    return 3
  fi
  cleanup
  # 训练超参必须**显式透传**进容器：ctr run 不继承宿主机环境，
  # 漏传会让 orchestrate.py 设的 MODEL_PATH / SMOKE_MAX_STEPS 等静默失效，
  # 表现为"明明指定了 7B，却还在训 3B"这类极难察觉的问题。
  local passthru=()
  for v in MODEL_PATH TRAIN_FILE VAL_FILE TOTAL_EPOCHS ROLLOUT_N TRAIN_BATCH MINI_BATCH \
           MAX_PROMPT_LEN MAX_RESP_LEN LORA_RANK LORA_ALPHA LR GPU_MEM_UTIL \
           SAVE_FREQ TEST_FREQ TEMPERATURE TOP_P ENTROPY_COEFF SMOKE_MAX_STEPS \
           REWARD_STRICT_SCORE VERIFY_TIMEOUT SANDBOX_IMAGE_TAG ACTOR_STRATEGY \
           OFFLOAD_POLICY RESUME_MODE CKPT_DIR MAX_CKPT_KEEP; do
    if [ -n "${!v:-}" ]; then
      passthru+=(--env "$v=${!v}")
    fi
  done
  ctr -n $NS run --rm --net-host --gpus 0 \
    --env PYTHONUNBUFFERED=1 \
    --env WORKDIR=$DATA \
    --env PYTHONPATH=$DATA:$DATA/pylibs \
    --env NCCL_SHM_DISABLE=1 \
    --env NCCL_P2P_DISABLE=1 \
    --env TOKENIZERS_PARALLELISM=false \
    "${passthru[@]}" \
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
  serve)
    # 线 A 的推理端：给沙箱内的 ReAct Agent 提供 OpenAI 兼容接口。
    #
    # 关键参数取值依据（5090 单卡 24G，Qwen2.5-Coder-7B ≈ 15G 权重）：
    #   --gpu-memory-utilization 0.85  权重 15G + KV cache，留约 3.6G 给显存碎片
    #                                  与 CUDA context；给到 0.9 实测偶发 OOM
    #   --max-model-len 16384          ReAct 多轮上下文需求；再大会挤掉 KV cache
    #                                  导致并发骤降
    #   --host 0.0.0.0                 配合 --net-host，让 VPC 内的沙箱可直连
    #                                  节点内网 IP:8000（无需 NodePort/Service）
    #   --served-model-name            固定别名，Agent 端不必写长路径
    echo "=== 启动 vLLM 推理服务（线 A）==="
    MODEL_PATH="${MODEL_PATH:-$DATA/model/Qwen2.5-Coder-7B-Instruct}"
    SERVED="${SERVED_NAME:-swe-rl-policy}"
    PORT="${VLLM_PORT:-8000}"
    # ⚠️ 必须用 NAME=... ; cleanup 两条语句，不能写 `NAME=xx cleanup`：
    # 命令前缀赋值只对**外部命令**生效，对 shell 函数无效 —— 函数里读到的
    # 仍是全局 NAME（swe-rl-train），会误清训练容器而放着 serve 容器不动。
    NAME="swe-rl-serve"
    cleanup
    ctr -n $NS run -d --net-host --gpus 0 \
      --env PYTHONUNBUFFERED=1 \
      --env VLLM_LOGGING_LEVEL=INFO \
      --mount type=bind,src=$DATA,dst=$DATA,options=rbind:rw \
      --mount type=tmpfs,dst=/dev/shm,options=rw:nosuid:nodev:size=4g \
      --cwd $DATA \
      "$IMAGE" swe-rl-serve \
      bash -lc "python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --served-model-name $SERVED \
        --host 0.0.0.0 --port $PORT \
        --gpu-memory-utilization 0.85 \
        --max-model-len 16384 \
        --disable-log-requests \
        > $DATA/logs/vllm_serve.log 2>&1"
    echo "已后台启动，日志：$DATA/logs/vllm_serve.log"
    echo "健康检查：curl -s http://127.0.0.1:$PORT/v1/models"
    ;;
  serve-stop)
    # 同上：前缀赋值对函数无效，必须分两条语句
    NAME="swe-rl-serve"
    cleanup
    echo "已停止 vLLM 服务容器"
    ;;
  shell)
    run_in_container "${1:-echo 需要提供命令}"
    ;;
  stop)
    cleanup
    echo "已清理容器 $NAME"
    ;;
  *)
    echo "用法：$0 {preflight|train|serve|serve-stop|shell <cmd>|stop}"
    exit 1
    ;;
esac
