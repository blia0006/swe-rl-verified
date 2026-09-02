#!/usr/bin/env bash
# VERL GRPO 训练入口（RTX 5090 / Blackwell sm_120）
# =================================================
#
# ## 与上一轮的差异
#
# 【保留】所有为 5090 调通的配置 —— 这些是踩坑换来的，改动前务必想清楚：
#   · actor.strategy=fsdp                verl 0.6.1 的必需项，缺了报配置错
#   · model_dtype=bfloat16（写全名）      vLLM 0.11 严格校验，写 bf16 会挂
#   · use_orig_params=False              修 LoRA writeback 报错
#   · use_torch_compile=False            sm_120 上 compile 不稳
#   · NCCL_SHM_DISABLE=1                 容器 /dev/shm 仅 64MB，而 NCCL 每 rank
#                                        需约 31.5MB，不设会零步崩溃且运行期无法补救
#
# 【改动】针对上一轮 reward 学不动的三处根因：
#   1. 模型 1.5B → 3B（代码专精；7B 因单卡显存装不下而放弃，见下）
#   2. 任务表示 unified diff → search/replace（reward function 侧实现，不需算行号）
#   3. reward 四档细化（pipeline/reward.py），"写歪"与"写对"差 4 倍
#
# 【模型选型的实测修正】原定 7B，但 24GB 单卡跑 hybrid engine 装不下
#   （依据见下方 GPU_MEM_UTIL 处），改用 **Qwen2.5-Coder-3B-Instruct**：
#   同为代码专精系列，参数量仍是上一轮 1.5B 的 2 倍。
#   若 OOM，依次降 max_response_length → rollout.n → gpu_memory_utilization。
#
# 用法（GPU 宿主机的容器内）：
#   bash scripts/run_grpo_training.sh
#   TOTAL_EPOCHS=1 ROLLOUT_N=8 bash scripts/run_grpo_training.sh
set -uo pipefail

WORKDIR="${WORKDIR:-/data/swe-rl}"
cd "$WORKDIR"

MODEL_PATH="${MODEL_PATH:-$WORKDIR/model/Qwen2.5-Coder-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$WORKDIR/data/grpo_train.parquet}"
CKPT_DIR="${CKPT_DIR:-$WORKDIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ---------- 环境自检：早失败好过训到一半崩 ----------
[ -d "$MODEL_PATH" ] || { echo "[x] 模型目录不存在：$MODEL_PATH"; exit 1; }
[ -f "$TRAIN_FILE" ] || { echo "[x] 训练数据不存在：$TRAIN_FILE"; exit 1; }
[ -f "$WORKDIR/.env" ] || { echo "[x] 缺少 .env（reward function 要用沙箱凭证）"; exit 1; }

# ---------- NCCL：容器 /dev/shm 只有 64MB 的兜底 ----------
# 单卡训练没有跨卡通信，禁用共享内存通道零性能损失
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

# ---------- reward function 需要的配置 ----------
set -a; . "$WORKDIR/.env"; set +a
export TASKS_FILE="${TASKS_FILE:-$WORKDIR/data/tasks.jsonl}"
export FILE_CONTENTS_FILE="${FILE_CONTENTS_FILE:-$WORKDIR/data/file_contents.json}"
export REWARD_DEBUG_LOG="${REWARD_DEBUG_LOG:-$LOG_DIR/reward_debug.jsonl}"
export SANDBOX_IMAGE_TAG="${SANDBOX_IMAGE_TAG:-sbx}"
export PYTHONPATH="$WORKDIR:$WORKDIR/pylibs:${PYTHONPATH:-}"

# ---------- 超参 ----------
ROLLOUT_N="${ROLLOUT_N:-8}"                 # GRPO 组大小；组内比较产生 advantage
TRAIN_BATCH="${TRAIN_BATCH:-2}"             # 上一轮=1 使单步 reward 主要由"抽到哪题"决定
MINI_BATCH="${MINI_BATCH:-2}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-6144}"    # prompt 内嵌文件内容
MAX_RESP_LEN="${MAX_RESP_LEN:-1024}"        # search/replace 块比整份 diff 短
LORA_RANK="${LORA_RANK:-32}"                # 3B 比上一轮 1.5B 容量大，rank 相应提高
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-1e-5}"
# ⚠️ gpu_memory_utilization 是 vLLM 可用显存的**总**占比（含权重本身），
# 不是"留给 KV cache 的比例"。且 verl 是 hybrid engine：
# FSDP actor 先加载、vLLM 后启动，vLLM 只能用**剩余**显存。
#
# 【实测结论：7B 在 24GB 单卡上装不下，已放弃】
#   FSDP actor 即便开了 param_offload，残留仍约 15.8GB（LoRA 参数 +
#   激活缓冲 + CUDA context）；vLLM 再要 15.2GB 权重 → 合计 31GB > 23.4GB。
#   三档全部失败：0.35/0.68 → KV cache 为负；0.26 →
#   "Failed to create unquantized linear weights"（连权重都放不下）。
#
# 【3B 的账】权重 6.2GB，FSDP 残留约 7GB，vLLM 预算 0.45×23.4≈10.5GB，
#   其中权重 6.2GB + KV cache 约 4.3GB，余量充足。
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEMPERATURE="${TEMPERATURE:-0.9}"           # 组内必须有方差，否则 advantage 恒 0
TOP_P="${TOP_P:-0.95}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="$LOG_DIR/train_$STAMP.log"

echo "======================================================"
echo " VERL GRPO 训练"
echo "   模型      : $MODEL_PATH"
echo "   数据      : $TRAIN_FILE"
echo "   组大小    : $ROLLOUT_N   batch=$TRAIN_BATCH"
echo "   LoRA      : rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "   日志      : $TRAIN_LOG"
echo "   reward 归因: $REWARD_DEBUG_LOG"
echo "======================================================"

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# ---------- 训练前清场：残留会导致零步崩溃 ----------
ray stop >/dev/null 2>&1 || true
sleep 3
pkill -f "ray::" >/dev/null 2>&1 || true
rm -f /dev/shm/nccl-* 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TRAIN_FILE" \
  data.train_batch_size="$TRAIN_BATCH" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESP_LEN" \
  data.shuffle=False \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$TOP_P" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$WORKDIR/pipeline/verl_reward_fn.py" \
  custom_reward_function.name=compute_score \
  trainer.logger=[console] \
  trainer.project_name=swe-rl-verified \
  trainer.experiment_name="grpo_$STAMP" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=-1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.default_local_dir="$CKPT_DIR" \
  2>&1 | tee "$TRAIN_LOG"

rc=${PIPESTATUS[0]}
echo ""
echo "训练退出码：$rc"
echo "日志       ：$TRAIN_LOG"
echo "reward 归因：$REWARD_DEBUG_LOG"

# ---------- 收尾：回收沙箱实例（按实例计费，必须释放）----------
python3 - <<'PY' || true
import sys
sys.path.insert(0, "/data/swe-rl")
try:
    from pipeline.verl_reward_fn import release_all
    release_all()
except Exception as e:
    print("[warn] 回收沙箱实例失败：%s" % e)
PY

exit "$rc"
