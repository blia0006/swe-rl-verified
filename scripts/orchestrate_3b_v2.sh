#!/bin/bash
# orchestrate_3b 修正版：
#   1. 前序脚本(verify_criteria.py / build_grpo_dataset.py)依赖 dotenv/pandas，
#      必须用 venv-orch/bin/python3，不能用系统 python3（v1 就是栽在这里）。
#   2. 【关键修正】真正的 GRPO 训练必须跑在 TKE Pod 容器内（课题硬性要求：
#      在 TKE 上部署 VERL 训练），不能像 v1 那样在宿主机裸跑 ——
#      改为 kubectl exec 进 swe-rl-train 容器内启动，且用 nohup+disown
#      让训练进程与本次 exec 会话脱钩（父进程退出后被 Pod 的 PID1 收养，
#      不会随 exec 连接断开被杀）。
set -uo pipefail
cd /data/swe-rl
PY=/data/swe-rl/venv-orch/bin/python3
export KUBECONFIG=/data/swe-rl/.kube/config
LOG=/data/swe-rl/logs/orchestrate_3b_v2.log
exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date)] orchestrate_3b_v2 启动 ==="

# ---------- 1. 补质检剩余未验证题目 ----------
echo "[1/4] 质检尚未验证的题目..."
$PY - <<'PYEOF'
import json
tasks = [json.loads(l)["task_id"] for l in open("data/tasks.jsonl") if l.strip()]
contents = json.load(open("data/file_contents.json"))
checked = set()
try:
    checked = {r["task_id"] for r in json.load(open("data/criteria_check.json"))["results"]}
except Exception:
    pass
todo = [t for t in tasks if t not in checked and contents.get(t)]
with open("/tmp/criteria_todo.txt", "w") as f:
    f.write("\n".join(todo))
print("待质检(有文件内容且未检):", len(todo), "已质检:", len(checked))
PYEOF

TODO_COUNT=$(wc -l < /tmp/criteria_todo.txt 2>/dev/null | tr -d ' ')
if [ "${TODO_COUNT:-0}" -gt 0 ]; then
  ARGS=""
  while IFS= read -r t; do
    [ -n "$t" ] && ARGS="$ARGS --task $t"
  done < /tmp/criteria_todo.txt
  $PY experiments/verify_criteria.py $ARGS --jobs 3 --out /tmp/criteria_new.json
  $PY - <<'PYEOF'
import json
old = json.load(open("data/criteria_check.json"))
try:
    new = json.load(open("/tmp/criteria_new.json"))
except Exception:
    new = {"results": []}
old_ids = {r["task_id"] for r in old["results"]}
for r in new.get("results", []):
    if r["task_id"] not in old_ids:
        old["results"].append(r)
json.dump(old, open("data/criteria_check.json", "w"), ensure_ascii=False, indent=2)
print("合并后质检总数:", len(old["results"]))
PYEOF
else
  echo "[1/4] 无需新质检"
fi

# ---------- 2. 按门禁筛出有效题目，重建 split.json ----------
echo "[2/4] 筛选有效题目并重建 split.json..."
$PY - <<'PYEOF'
import json, random

criteria = json.load(open("data/criteria_check.json"))
contents = json.load(open("data/file_contents.json"))

def ok(r):
    sc = r.get("scenarios", {})
    g = sc.get("② golden", {}).get("reward", {}) or {}
    e = sc.get("① 空解", {}).get("reward", {}) or {}
    j = sc.get("③ 垃圾patch", {}).get("reward", {}) or {}
    return (
        g.get("strict_pass") is True and g.get("reward") == 1.0
        and e.get("strict_pass") is not True
        and j.get("reward", 1) == 0.0
    )

valid = [r["task_id"] for r in criteria["results"] if ok(r) and contents.get(r["task_id"])]
valid = sorted(set(valid))
print("有效题目总数:", len(valid))

random.seed(42)
random.shuffle(valid)
n_eval = min(10, max(5, len(valid) // 6))
eval_ids = valid[:n_eval]
train_ids = valid[n_eval:]
print("train:", len(train_ids), "eval:", len(eval_ids))

split = {
    "train": train_ids,
    "eval": eval_ids,
    "criteria": {"note": "仅保留通过verify_criteria.py门禁的题目", "valid_total": len(valid)},
}
json.dump(split, open("data/split.json", "w"), ensure_ascii=False, indent=2)
PYEOF

# ---------- 3. 重建 grpo_train.parquet / grpo_eval.parquet ----------
echo "[3/4] 重建训练/评测数据集..."
TRAIN_N=$($PY -c "import json; print(len(json.load(open('data/split.json'))['train']))")
REPEAT=$(( (180 + TRAIN_N - 1) / TRAIN_N ))
[ "$REPEAT" -lt 4 ] && REPEAT=4
echo "训练题目数=$TRAIN_N，repeat=$REPEAT，预计 step 数=$((TRAIN_N * REPEAT))"

$PY pipeline/build_grpo_dataset.py --split-key train --repeat "$REPEAT" --out data/grpo_train.parquet
$PY pipeline/build_grpo_dataset.py --split-key eval --repeat 1 --out data/grpo_eval.parquet

# ---------- 4. 在 TKE Pod 容器内启动 3B 正式训练（不在宿主机裸跑） ----------
echo "[4/4] 通过 kubectl exec 在 Pod 内启动 3B GRPO 训练 $(date)"

if ! kubectl get pod swe-rl-train >/dev/null 2>&1; then
  echo "[x] swe-rl-train Pod 不存在，请先 bash deploy/apply.sh"
  exit 1
fi
POD_PHASE=$(kubectl get pod swe-rl-train -o jsonpath='{.status.phase}' 2>/dev/null)
if [ "$POD_PHASE" != "Running" ]; then
  echo "[x] swe-rl-train Pod 状态为 $POD_PHASE，不是 Running，拒绝启动"
  exit 1
fi

kubectl exec swe-rl-train -- bash -lc '
  cd /data/swe-rl
  MODEL_PATH=/data/swe-rl/model/Qwen2.5-Coder-3B-Instruct \
  CKPT_DIR=/data/swe-rl/checkpoints_3b_v1 \
  TOTAL_EPOCHS=1 \
  nohup bash scripts/run_grpo_training.sh > logs/train_pod_3b.log 2>&1 &
  disown
  sleep 2
  echo "[pod] 训练已在容器内后台启动，日志: /data/swe-rl/logs/train_pod_3b.log"
'

echo "=== [$(date)] orchestrate_3b_v2 全部完成（训练已在 Pod 内后台运行） ==="
