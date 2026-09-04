#!/usr/bin/env bash
# 训练看护：崩了自动从断点续跑
# ============================
#
# ## 为什么需要
#
# 单卡 24GB 跑 7B 的 RL 训练余量只有几百 MB，中途被打断是常态。今天实测遇到过：
#   · 显存 OOM（训练侧差 400MB）
#   · 磁盘写满（checkpoint 各 13GB）
#   · 训练镜像被 kubelet GC 回收
#   · 沙箱判分偶发超时
#
# 每次人工发现→清场→重启要几十分钟，且夜间无人值守时会白等一整晚。
# 本脚本让训练具备自愈能力：崩溃后自动清场、检查前置条件、从最新 checkpoint
# 续训（靠 verl 的 resume_mode=auto）。
#
# ## 关键设计
#
# **靠 checkpoint 续训而非重跑**：verl 的 auto 模式会读
# `latest_checkpointed_iteration.txt`，从那一步继续。因此 SAVE_FREQ 越小、
# 崩溃损失越少，但 checkpoint 占盘（各 13GB）—— 用 MAX_CKPT_KEEP 限制保留数。
#
# **区分可重试与不可重试**：磁盘满、镜像缺失这类问题重试无用（会一直失败），
# 必须先修复。脚本会识别并停止重试，把原因写清。
#
# 用法：
#   bash scripts/train_guard.sh              # 默认最多重启 5 次
#   MAX_RETRY=10 bash scripts/train_guard.sh
set -uo pipefail

DATA=/data/swe-rl
cd "$DATA" || exit 1

MAX_RETRY="${MAX_RETRY:-5}"
GUARD_LOG="$DATA/logs/train_guard.log"
STATE="$DATA/logs/train_guard_state.txt"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$GUARD_LOG"; }

# 训练目标步数：用于判断"是否已跑完"，跑完就不再重启
TARGET_STEPS="${TARGET_STEPS:-60}"

current_step() {
  # 取已完成的最大步数：优先看 checkpoint（真正落盘的），日志只作参考
  local from_ckpt=0
  for d in "$DATA"/checkpoints/global_step_*; do
    [ -d "$d" ] || continue
    local n="${d##*_}"
    [ "$n" -gt "$from_ckpt" ] 2>/dev/null && from_ckpt="$n"
  done
  echo "$from_ckpt"
}

classify_failure() {
  local log="$1"
  if grep -q "available space: 0 GB\|No space left on device" "$log" 2>/dev/null; then
    echo "disk_full"          # 不可重试：必须先清盘
  elif grep -q "image .*not found" "$log" 2>/dev/null; then
    echo "image_missing"      # 不可重试：必须先拉镜像
  elif grep -qE "OutOfMemoryError|less than desired GPU memory|No available memory for the cache" "$log" 2>/dev/null; then
    echo "oom"                # 可重试：清场后往往能过
  elif grep -q "device type 'cpu' to a tensor with device type 'cuda'" "$log" 2>/dev/null; then
    echo "device_mismatch"    # 不可重试：配置问题（fsdp2 + offload + LoRA）
  else
    echo "unknown"            # 可重试一次看是否偶发
  fi
}

for attempt in $(seq 1 "$MAX_RETRY"); do
  step_before="$(current_step)"
  if [ "$step_before" -ge "$TARGET_STEPS" ]; then
    log "已达目标步数 $step_before/$TARGET_STEPS，无需继续"
    echo "done:$step_before" > "$STATE"
    exit 0
  fi

  log "=== 第 $attempt/$MAX_RETRY 次启动（当前进度 step $step_before/$TARGET_STEPS）==="

  # 前置：清场 + 确保镜像在位
  bash "$DATA/scripts/start_train_container.sh" stop      >/dev/null 2>&1 || true
  bash "$DATA/scripts/start_train_container.sh" serve-stop >/dev/null 2>&1 || true
  sleep 5
  bash "$DATA/scripts/ensure_train_image.sh" 2>&1 | tail -1 | tee -a "$GUARD_LOG"

  RUN_LOG="$DATA/logs/train_attempt_${attempt}.log"
  MODEL_PATH="${MODEL_PATH:-$DATA/model/Qwen2.5-Coder-7B-Instruct}" \
  SAVE_FREQ="${SAVE_FREQ:-10}" \
  RESUME_MODE=auto \
    bash "$DATA/scripts/start_train_container.sh" train > "$RUN_LOG" 2>&1
  rc=$?

  step_after="$(current_step)"
  log "本次退出码=$rc，进度 step $step_before → $step_after"

  if [ "$rc" -eq 0 ] && [ "$step_after" -ge "$TARGET_STEPS" ]; then
    log "✓ 训练完成，共 $step_after 步"
    echo "done:$step_after" > "$STATE"
    exit 0
  fi

  kind="$(classify_failure "$RUN_LOG")"
  log "失败类型：$kind"

  case "$kind" in
    disk_full)
      avail="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
      log "✗ 磁盘不足（可用 ${avail}GB），重试无用。请清理后再启动："
      du -sh "$DATA"/checkpoints* /var/lib/containerd 2>/dev/null | sort -rh | head -3 | tee -a "$GUARD_LOG"
      echo "blocked:disk_full" > "$STATE"
      exit 2
      ;;
    image_missing|device_mismatch)
      log "✗ $kind 属配置/环境问题，重试无用，需人工处理"
      echo "blocked:$kind" > "$STATE"
      exit 2
      ;;
  esac

  # 进度完全没推进且不是首次 → 说明卡在同一处，避免无限空转
  if [ "$attempt" -ge 2 ] && [ "$step_after" -le "$step_before" ]; then
    log "✗ 连续两次无进展（停在 step $step_after），停止重试"
    echo "stalled:$step_after" > "$STATE"
    exit 3
  fi

  log "将在 30s 后从 step $step_after 续训…"
  sleep 30
done

log "已达最大重试次数 $MAX_RETRY，最终进度 step $(current_step)"
echo "exhausted:$(current_step)" > "$STATE"
exit 4
