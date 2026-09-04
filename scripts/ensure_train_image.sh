#!/usr/bin/env bash
# 确保训练镜像存在（不存在则拉取）
# ================================
#
# ## 为什么需要这个脚本
#
# 训练镜像放在 containerd 的 `k8s.io` namespace 里，该 namespace 由 kubelet 管理，
# **kubelet 的镜像 GC 会回收没有被任何 Pod 引用的镜像**。我们用 `ctr run` 起容器
# 而不经 k8s，所以镜像在 kubelet 看来是"未被引用"的，随时可能被清掉。
#
# 实测：训练跑了一整天都正常，某次重启突然报
#     ctr: image "docker.io/verlai/verl:vllm011.latest": not found
# 磁盘还有 65G 空闲 —— 不是空间不足，就是 GC 回收了。
#
# 因此每次启动训练前都要确认镜像在位。拉取约 10GB，需要几分钟，
# 所以必须后台执行并可轮询。
set -uo pipefail

# 与 start_train_container.sh 保持一致：用 default 而非 k8s.io
NS="${CTR_NS:-default}"
IMAGE="${IMAGE:-mirror.ccs.tencentyun.com/verlai/verl:vllm011.latest}"
LOG=/data/swe-rl/logs/pull_verl.log
PIDF=/data/swe-rl/logs/pull_verl.pid

mkdir -p /data/swe-rl/logs

if ctr -n $NS i ls -q 2>/dev/null | grep -qx "$IMAGE"; then
  echo "镜像已在位：$IMAGE"
  exit 0
fi

# 已有拉取任务在跑就不重复启动（并发拉同一镜像会互相拖慢）
if [ -f "$PIDF" ] && ps -p "$(cat "$PIDF")" > /dev/null 2>&1; then
  echo "拉取已在进行中 pid=$(cat "$PIDF")"
  tail -2 "$LOG" 2>/dev/null
  exit 0
fi

echo "镜像缺失，开始后台拉取：$IMAGE"
setsid nohup ctr -n $NS i pull "$IMAGE" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
sleep 5
echo "pid=$(cat "$PIDF")  日志=$LOG"
tail -3 "$LOG" 2>/dev/null || echo "（日志尚未产生）"
