#!/usr/bin/env bash
# 在 GPU 节点上启动全流程编排（脱离本地会话）
# ============================================
#
# 为什么单独成脚本而不是一行 ssh 命令：
# 云助手（TAT）执行的是一段 sh -c，其中混用 `cd`、后台 `&`、`$!`、重定向时
# 极易踩到解析顺序问题 —— 实测 `cd X && setsid nohup ... & echo $! > logs/p.pid`
# 里的 `logs/p.pid` 会在 cd 生效前被解析，报 "No such file or directory"。
# 写成脚本文件后由 bash 完整解析，行为确定。
#
# setsid + nohup + 重定向三者缺一不可：
#   setsid   脱离控制终端，云助手任务结束时不会连带杀掉子进程
#   nohup    忽略 SIGHUP
#   重定向   否则进程写日志时因 stdout 关闭而崩
set -uo pipefail

DATA=/data/swe-rl
PY="$DATA/venv-orch/bin/python"
STAGES="${STAGES:-collect,train,eval}"
LOG="$DATA/logs/orchestrate.log"
PIDF="$DATA/logs/orchestrate.pid"

mkdir -p "$DATA/logs"
cd "$DATA" || exit 1

# 幂等：已有存活的编排进程就不再重复启动（重复跑会两个进程抢 GPU）
if [ -f "$PIDF" ] && ps -p "$(cat "$PIDF")" > /dev/null 2>&1; then
  echo "编排已在运行 pid=$(cat "$PIDF")，不重复启动"
  exit 0
fi

setsid nohup "$PY" scripts/orchestrate.py --stages "$STAGES" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
sleep 3

PID="$(cat "$PIDF")"
echo "pid=$PID  日志=$LOG  阶段=$STAGES"
if ps -p "$PID" > /dev/null 2>&1; then
  echo "编排进程存活 ✓"
else
  echo "✗ 进程已退出，日志尾部："
  tail -20 "$LOG"
  exit 1
fi
tail -5 "$LOG"
