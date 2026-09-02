#!/usr/bin/env bash
# 输出「step 数最多」的训练日志路径。
#
# 为什么不用 `ls -t | head -1`：训练脚本每次以新时间戳建日志，
# 启动失败的空日志反而更新，按时间挑会读到错的文件 ——
# 实测曾因此只取到 11 步，而真实最新一轮跑了 31 步。
best=""
best_n=0
for f in /data/swe-rl/logs/train_2*.log; do
  [ -f "$f" ] || continue
  # grep -c 无匹配时退出码非 0，需兜底；tr -d 去掉可能的换行，否则 [ ] 报
  # "integer expression expected"
  n=$(grep -ac 'critic/score/mean' "$f" 2>/dev/null | tr -d "[:space:]")
  [ -n "$n" ] || n=0
  if [ "$n" -gt "$best_n" ]; then
    best_n="$n"
    best="$f"
  fi
done
echo "$best"
