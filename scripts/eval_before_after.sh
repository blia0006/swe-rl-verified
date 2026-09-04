#!/usr/bin/env bash
# 闭环验收：训练前(base) vs 训练后(checkpoint) 在 SandBox 上的 pass@1 对比
# ============================================================
#
# 对应课题验收第 ⑤⑥ 条：
#   ⑤ 完成 1 轮闭环：SandBox 产出 tracing → TKE 训练 → 新模型回 SandBox 评估
#   ⑥ 训练后 pass@1 相比训练前有可观测提升（需给出对比数据）
#
# 用法（在 GPU 宿主机上，由 node.py nohup 后台启动）：
#   bash scripts/eval_before_after.sh
#
# 全程在同一进程内顺序执行，整体由外层 nohup 托管，可安全断网。
set -uo pipefail

WORKDIR=/data/swe-rl
cd "$WORKDIR"
LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"

CKPT_DIR="$WORKDIR/checkpoints_3b_v1"
BASE_MODEL="$WORKDIR/model/Qwen2.5-Coder-3B-Instruct"
CKPT_STEP="global_step_192"
MERGED_DIR="$CKPT_DIR/$CKPT_STEP/merged_hf"
PYLIB="$WORKDIR:$WORKDIR/pylibs"

# 注意：该训练容器不经 kubelet/crictl 管理（pod 已从 k8s 视角摘除，但 containerd
# 里的 task 仍在跑），crictl ps 查不到它，必须用 ctr 在 k8s.io namespace 下操作。
CID=$(ctr -n k8s.io containers ls 2>/dev/null | grep 'verlai/verl' | awk '{print $1}' | head -1)
if [ -z "$CID" ]; then
  echo "[x] 找不到 verlai/verl 训练容器，终止"
  exit 1
fi
echo "[ok] 容器 CID=$CID"

# 该训练容器走 VPC-CNI，有独立 Pod IP（与宿主机同属 10.0.0.0/24 段，
# 是 VPC 内网可路由地址），宿主机 127.0.0.1 访问不到容器内监听的端口，
# 必须用这个容器 IP 才能连上 vLLM 服务。
CONTAINER_IP=$(ctr -n k8s.io tasks exec --exec-id "eba-ip-$(date +%s%N)" "$CID" hostname -I 2>/dev/null | tr -d ' \t\r\n')
if [ -z "$CONTAINER_IP" ]; then
  echo "[x] 获取容器 IP 失败，终止"
  exit 1
fi
echo "[ok] 容器 IP=$CONTAINER_IP"

pod_exec() {
  # 前台执行、阻塞到完成（用于 merge 这类一次性同步任务）
  ctr -n k8s.io tasks exec --exec-id "eba-$(date +%s%N)" "$CID" sh -c "$1"
}

pod_exec_bg() {
  # ctr -d：分离后台，独立于本次调用连接，容器不退出则进程持续运行
  ctr -n k8s.io tasks exec -d --exec-id "eba-bg-$(date +%s%N)" "$CID" sh -c "$1"
  sleep 1
  echo "  已提交后台任务"
}

stop_vllm() {
  pod_exec "pkill -f vllm.entrypoints.openai.api_server 2>/dev/null; sleep 2; pkill -9 -f vllm.entrypoints.openai.api_server 2>/dev/null; true"
  sleep 3
}

wait_vllm_ready() {
  local name="$1"
  for i in $(seq 1 90); do
    if curl -s -m 3 "http://${CONTAINER_IP}:8000/v1/models" 2>/dev/null | grep -q "$name"; then
      echo "  ✓ vLLM 就绪（用时 ~$((i * 5))s）"
      return 0
    fi
    sleep 5
  done
  echo "  ✗ vLLM 启动超时（450s）"
  return 1
}

start_vllm() {
  local model_path="$1" served_name="$2"
  echo "  启动 vLLM：served_name=$served_name model=$model_path"
  pod_exec_bg "cd $WORKDIR && PYTHONPATH=$PYLIB python3 -m vllm.entrypoints.openai.api_server --model $model_path --served-model-name $served_name --port 8000 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 16384 --enforce-eager --trust-remote-code > $WORKDIR/logs/vllm_${served_name}.log 2>&1"
  wait_vllm_ready "$served_name"
}

run_collect_eval() {
  local run_id="$1" served_name="$2"
  echo "  运行 collect_tracing.py --split eval（run_id=$run_id）"
  "$WORKDIR/venv-orch/bin/python" "$WORKDIR/pipeline/collect_tracing.py" \
    --split eval -n 1 --temperature 0.2 --jobs 4 --max-steps 20 \
    --model "$served_name" --base-url "http://${CONTAINER_IP}:8000" \
    --run-id "$run_id" --strict-eval \
    >> "$LOG_DIR/eval_${run_id}.log" 2>&1
  echo "  → 日志: $LOG_DIR/eval_${run_id}.log"
}

echo "======================================================"
echo " 闭环评测：base（训练前） vs $CKPT_STEP（训练后）"
echo "======================================================"

echo "[1/6] 合并 checkpoint 的 LoRA 权重 → $MERGED_DIR"
if [ -f "$MERGED_DIR/config.json" ]; then
  echo "  已存在，跳过合并"
else
  pod_exec "cd $WORKDIR && PYTHONPATH=$PYLIB python3 -m verl.model_merger merge --backend fsdp --local_dir $CKPT_DIR/$CKPT_STEP/actor --target_dir $MERGED_DIR --trust-remote-code 2>&1 | tail -60"
  if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "[x] 合并失败：$MERGED_DIR/config.json 不存在，终止"
    exit 1
  fi
  echo "  ✓ 合并完成"
fi

echo "[2/6] 清理残留 vLLM 进程"
stop_vllm

echo "[3/6] 评测 base（训练前）模型：$BASE_MODEL"
start_vllm "$BASE_MODEL" "swe-rl-base" || { echo "[x] base vLLM 启动失败"; exit 1; }
run_collect_eval "before_base" "swe-rl-base"
stop_vllm

echo "[4/6] 评测 $CKPT_STEP（训练后）模型：$MERGED_DIR"
start_vllm "$MERGED_DIR" "swe-rl-ckpt192" || { echo "[x] checkpoint vLLM 启动失败"; exit 1; }
run_collect_eval "after_ckpt192" "swe-rl-ckpt192"
stop_vllm

echo "[5/6] 汇总对比"
"$WORKDIR/venv-orch/bin/python" - <<'PYEOF'
import json
from pathlib import Path

OUT = Path("/data/swe-rl/data/tracing")

def load(run_id):
    p = OUT / f"summary_{run_id}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    rewards, resolved, total = [], 0, 0
    for r in d["results"]:
        for x in r.get("rollouts", []):
            rewards.append(x["reward"])
            total += 1
            if x["resolved"]:
                resolved += 1
    return {
        "run_id": run_id,
        "n_tasks": len(d["results"]),
        "n_rollouts": total,
        "n_resolved": resolved,
        "pass_at_1": round(resolved / total, 4) if total else None,
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
    }

before = load("before_base")
after = load("after_ckpt192")
report = {"before_base": before, "after_ckpt192_step192": after}
Path("/data/swe-rl/data/pass_at_1_before_after.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
PYEOF

echo "[6/6] 全部完成。结果见 /data/swe-rl/data/pass_at_1_before_after.json"
