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

MODEL_PATH="${MODEL_PATH:-$WORKDIR/model/Qwen2.5-Coder-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$WORKDIR/data/grpo_train.parquet}"
# 2026-09-03 修正：此前 val_files 一直等于 TRAIN_FILE，"验证" 实为在训练集自身
# 上打分，不是真正的 held-out 泛化评估。split.json 本就有独立 eval 题，
# 现改用 build_grpo_dataset.py --split-key eval 单独产出的 grpo_eval.parquet。
VAL_FILE="${VAL_FILE:-$WORKDIR/data/grpo_eval.parquet}"
CKPT_DIR="${CKPT_DIR:-$WORKDIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ---------- 环境自检：早失败好过训到一半崩 ----------
[ -d "$MODEL_PATH" ] || { echo "[x] 模型目录不存在：$MODEL_PATH"; exit 1; }
[ -f "$TRAIN_FILE" ] || { echo "[x] 训练数据不存在：$TRAIN_FILE"; exit 1; }
[ -f "$VAL_FILE" ] || { echo "[x] 评测数据不存在：$VAL_FILE（先跑 build_grpo_dataset.py --split-key eval）"; exit 1; }
[ -f "$WORKDIR/.env" ] || { echo "[x] 缺少 .env（reward function 要用沙箱凭证）"; exit 1; }

# ---------- NCCL：容器 /dev/shm 只有 64MB 的兜底 ----------
# 单卡训练没有跨卡通信，禁用共享内存通道零性能损失
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
# ⚠️ 不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
# OOM 报错里 PyTorch 会主动建议开这个选项来减少碎片，但**在 vLLM 场景下不可用**：
#   AssertionError: Expandable segments are not compatible with memory pool.
#   （vllm/device_allocator/cumem.py 显式 assert 该值不存在）
# vLLM 的 CuMemAllocator 用 memory pool 实现 sleep/wake（权重与 KV cache 分离
# 释放），与可扩展段机制冲突。实测一设就在 vLLM 加载模型阶段直接崩。
# 参见 https://github.com/pytorch/pytorch/issues/147851
#
# 因此显式清空，避免外部环境变量误传进来。
unset PYTORCH_CUDA_ALLOC_CONF

# ---------- reward function 需要的配置 ----------
set -a; . "$WORKDIR/.env"; set +a
export TASKS_FILE="${TASKS_FILE:-$WORKDIR/data/tasks.jsonl}"
export FILE_CONTENTS_FILE="${FILE_CONTENTS_FILE:-$WORKDIR/data/file_contents.json}"
export REWARD_DEBUG_LOG="${REWARD_DEBUG_LOG:-$LOG_DIR/reward_debug.jsonl}"
export SANDBOX_IMAGE_TAG="${SANDBOX_IMAGE_TAG:-sbx}"
export PYTHONPATH="$WORKDIR:$WORKDIR/pylibs:${PYTHONPATH:-}"

# ---------- 超参 ----------
# ⚠️ 已归档的 31 step 实测结果（README §7.3）跑的是**旧配置**：
#     ROLLOUT_N=8, TRAIN_BATCH=2, MINI_BATCH=2  → 63 行数据只出 31 step
#   下方为据其诊断结论调整后的值，尚未执行。要复现归档结果请用：
#     ROLLOUT_N=8 TRAIN_BATCH=2 MINI_BATCH=2 bash scripts/run_grpo_training.sh

# GRPO 组大小、序列长度、LoRA 秩、显存占比这四组**按模型大小分档**，
# 统一在下方 case 里定义 —— 若在这里先赋值，case 里的 `${VAR:-默认}`
# 会因变量已非空而全部失效，7B 档位形同虚设（这是个很容易踩的坑）。
#
# 每步题数。=2 时 63 行数据只跑出 31 step（行数被折半），不满足验收的 ≥50 step；
# 且每步只抽 2 题、题池仅 9 题，抽样方差压过学习信号 ——
# 首轮实测 reward 前 1/3 (0.0344) > 后 1/3 (0.0163)，
# 但满分步随机散布在 2,7,9,11,20,25,29，是抽样噪声而非真实下降趋势。
# 改回 1：step 数 = 行数，且靠加大 ROLLOUT_N 保证组内多样性。
TRAIN_BATCH="${TRAIN_BATCH:-1}"
MINI_BATCH="${MINI_BATCH:-1}"
LR="${LR:-1e-5}"
# ⚠️ gpu_memory_utilization 是 vLLM 可用显存的**总**占比（含权重本身），
# 不是"留给 KV cache 的比例"。且 verl 是 hybrid engine：
# FSDP actor 先加载、vLLM 后启动，vLLM 只能用**剩余**显存。
#
# 【上一轮实测：7B 在 24GB 单卡装不下】
#   FSDP actor 即便开了 param_offload，残留仍约 15.8GB（LoRA 参数 +
#   激活缓冲 + CUDA context）；vLLM 再要 15.2GB 权重 → 合计 31GB > 23.4GB。
#   三档全部失败：0.35/0.68 → KV cache 为负；0.26 →
#   "Failed to create unquantized linear weights"（连权重都放不下）。
#
# 【本轮为何仍先试 7B】采集与评测都用 7B（3B 实测解不对题、reward 恒 0，
#   训练必然学不动）。训练模型若与之不一致，前后对比毫无意义。
#   因此按模型大小**自动选择**省显存档位，给 7B 一次机会；
#   scripts/orchestrate.py 会先跑冒烟测试，装不下再降级 3B。
#
# 【3B 的账】权重 6.2GB，FSDP 残留约 7GB，vLLM 预算 0.45×23.4≈10.5GB，
#   其中权重 6.2GB + KV cache 约 4.3GB，余量充足。
case "$MODEL_PATH" in
  *7B*)
    # 【2026-09-03 实测确定的唯一可行窗口，改任一项都会崩】
    #   GPU_MEM_UTIL=0.75 + prompt 3072 + resp 512 + n=4 + fsdp2 + offload关
    #   → 跑过 step 12，零报错，score 出现 0.75 等实质分数
    #
    # GPU_MEM_UTIL 的窗口极窄，逐档实测（其余参数相同）：
    #   0.28/0.30 → vLLM 权重放不下（unquantized linear weights）
    #   0.40/0.62 → 权重下了，KV cache 不够（No available memory for cache blocks）
    #   0.68      → KV cache 仍不够
    #   0.75      → ✓ 唯一可行
    #   0.80      → vLLM 起来了，但训练侧 OOM（差约 400MB）
    # 注意方向**非单调**：太低连权重都放不下，太高挤掉训练侧。
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-3072}"
    MAX_RESP_LEN="${MAX_RESP_LEN:-512}"
    LORA_RANK="${LORA_RANK:-16}"
    LORA_ALPHA="${LORA_ALPHA:-32}"
    # 组大小 4 是显存与组内多样性的折中：8 会 OOM，2 则组内方差不足
    ROLLOUT_N="${ROLLOUT_N:-4}"
    ;;
  *)
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-6144}"
    MAX_RESP_LEN="${MAX_RESP_LEN:-1024}"
    LORA_RANK="${LORA_RANK:-32}"
    LORA_ALPHA="${LORA_ALPHA:-32}"
    ROLLOUT_N="${ROLLOUT_N:-16}"
    ;;
esac
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"
# 之前 test_freq=-1 只在开头做一次 initial validation，看不到 pass@1 随训练变化的过程。
# 交付需要一条曲线，改成每 TEST_FREQ 步在 held-out eval 集上跑一次。
TEST_FREQ="${TEST_FREQ:-10}"
TEMPERATURE="${TEMPERATURE:-0.9}"           # 组内必须有方差，否则 advantage 恒 0
TOP_P="${TOP_P:-0.95}"
# 2026-09-03 诊断：102 步实测里 critic/advantages/mean 超过一半的步精确为 0.0
# （同一题 4 次采样输出完全一致，GRPO 组内方差=0 → 该步梯度为死信号）。
# ROLLOUT_N 受 7B 显存上限卡在 4（8 会 OOM，已实测），没法靠加大组内采样数缓解，
# 只能靠给 entropy 一个小的正系数，抑制策略过早收缩为确定性输出、
# 延长"同题多次采样仍有差异"的窗口。0.0 → 保持旧行为；新跑建议 0.005~0.01。
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"

# ---------- 续训模式 ----------
#
# ## 为什么默认改成 auto（可断点续跑）
#
# 单卡 24G 跑 7B 的 RL 训练余量极小，中途被各种原因打断是常态（实测遇到过
# 显存 OOM、磁盘满、镜像被 GC 回收）。每次从零重跑要几小时，代价太高。
# verl 的 `resume_mode=auto` 会自动从 CKPT_DIR 里最新的 checkpoint 继续，
# 是本项目应有的默认行为。
#
# ## 之前为何一度关掉它（以及现在为何能安全打开）
#
# 早先设 disable 是因为两个问题，现在都已解决：
#
# 1) **显存**：加载 checkpoint 时权重要先落 GPU，与 actor/vLLM 叠加导致 OOM。
#    → 根因其实是当时 GPU_MEM_UTIL=0.80 本身就在临界点。改用实测确定的
#      0.75 后余量足够，续训加载不再爆。
#
# 2) **结论污染**：调参期留下一堆不同配置的半截 checkpoint，auto 模式会
#    悄悄从其中某个续训，导致"本次训练"混入失败配置的权重。
#    → 现在靠 CKPT_DIR 的纪律来防：**换配置就换目录**（见下方 CKPT_DIR），
#      同一目录内的 checkpoint 保证同配置，续训是安全的。
#
# 要强制从零开始时传 RESUME_MODE=disable。
RESUME_MODE="${RESUME_MODE:-auto}"

# 【关键】续训时跳过加载 optimizer 状态，只加载 model 权重
#
# 实测事故：GPU_MEM_UTIL=0.75 冷启动能装下（验证过 step 12），但**续训时**
# 加载 checkpoint 会在原有显存占用基础上额外申请空间来反序列化 optimizer
# 状态（Adam 的一二阶矩，体积与模型参数相当），结果：
#   torch.OutOfMemoryError: Tried to allocate 130.00 MiB ... 65.44 MiB is free
# 只差 65MB，冷启动和续训对显存的需求并不对称。
#
# 权衡：跳过 optimizer 状态意味着续训后 Adam 动量从零重新累积，
# 相当于该步有一次很小的"热重启"扰动，但远好于完全无法续训。
# save_contents 仍保留 optimizer（万一以后显存更宽松想完整续训），
# 只是 load_contents 覆盖为不读它。
LOAD_CONTENTS="${LOAD_CONTENTS:-[model,extra]}"

# checkpoint 保留数：单个约 13GB，197G 的盘放不了几个。
# 实测事故：不限制时训练到 step 30 把磁盘写满（100%），ray 报
#   file_system_monitor: available space: 0 GB
# 训练崩溃且最后一个 checkpoint 可能不完整（latest 标记落后于实际目录）。
# 保留 2 个：足够续训（用最新的），又给"最新那个写坏了"留一个退路。
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"

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

# ---------- FSDP 策略：7B 用 fsdp2，但**不开** offload_policy ----------
#
# 【为什么必须 fsdp2】fsdp1 下 7B 的 vLLM engine 初始化会失败：
#   fsdp1 的 param_offload 只在 rollout_mode() 里搬走权重，而 vLLM 初始化在那之前，
#   此时 FSDP actor 已占 15.65GB。实测三档全灭：
#     GPU_MEM_UTIL=0.28/0.30 → "Failed to create unquantized linear weights"
#     GPU_MEM_UTIL=0.72      → "Free memory (7.76/23.41) < desired (16.86)"
#
# 【为什么 offload_policy 必须关】它会让 fsdp2 用 CPUOffloadPolicy，
#   梯度落在 CPU 而参数在 GPU，反向传播时报
#     RuntimeError: attempting to assign a gradient with device type 'cpu'
#                   to a tensor with device type 'cuda'
#   实测该组合能跑到 step 10 才崩（前 10 步不触发 LoRA 的梯度写回路径），
#   极具误导性 —— 冒烟测试跑 1~3 步完全测不出来。
#   结论：**FSDP2 的 CPU offload 与 LoRA 不兼容**。
#
# 【关掉后为何还装得下】实测 max_memory_allocated 报到 35.6GB 却不崩，
#   说明驱动在做显存超额分配（把部分张量放到主机内存）。虽非最优，
#   但稳定跑通了 step 11 且 grad_norm 正常（0.30），可用。
#   代价是单步变慢，属可接受范围。
case "$MODEL_PATH" in
  *7B*)
    ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
    OFFLOAD_POLICY="${OFFLOAD_POLICY:-False}"
    ;;
  *)
    ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp}"
    OFFLOAD_POLICY="${OFFLOAD_POLICY:-False}"
    ;;
esac

# use_orig_params 是 **fsdp1 专属**参数（fsdp2 无此概念，传了会报未知配置）。
# 它当初是为修 LoRA writeback 报错才加的，因此只在 fsdp1 下传。
if [ "$ACTOR_STRATEGY" = "fsdp" ]; then
  USE_ORIG_PARAMS_ARG="actor_rollout_ref.actor.fsdp_config.use_orig_params=False"
else
  USE_ORIG_PARAMS_ARG=""
fi

# ---------- 训练前清场：残留会导致零步崩溃 ----------
ray stop >/dev/null 2>&1 || true
sleep 3
pkill -f "ray::" >/dev/null 2>&1 || true
rm -f /dev/shm/nccl-* 2>/dev/null || true

# ---------- 显存守门：占用未释放就直接失败，别浪费几十分钟 ----------
#
# 实测事故：编排被 kill 后，它起的 vLLM 服务容器（ctr run -d，无父子关系）
# 不会跟着退出，留下 `VLLM::EngineCore` 孤儿进程独占 21.5GB。训练随后报
#     Free memory on device (7.76/23.41 GiB) < desired (0.72, 16.86 GiB)
# 这个报错看起来像"7B 装不下 24G 卡"，与历史结论吻合，极易误导人放弃 7B。
# 真因只是没清干净。因此在这里**主动拦一道**，把问题指向正确的方向。
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "${GPU_USED:-0}" -gt 1500 ]; then
  echo ""
  echo "[x] 训练前 GPU 已被占用 ${GPU_USED} MiB，拒绝启动（否则必然 OOM）"
  echo "    占用者："
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/      /'
  echo ""
  echo "    请先清场：bash scripts/start_train_container.sh serve-stop"
  exit 2
fi
echo "[ok] 显存检查通过（已用 ${GPU_USED} MiB）"

# ---------- 磁盘守门：单个 checkpoint 约 14GB，很容易撑爆 ----------
#
# 实测事故：训练跑到 step 30 时磁盘 100% 满，ray 报
#     file_system_monitor.cc: /tmp/ray/... is over 95% full, available space: 0 GB
# 训练随即崩溃。真因是调参过程留下的归档 checkpoint（60G + 29G）没清，
# 加上本轮自己写的 28G，197G 的盘直接见底。
#
# 这个失败和显存无关，但表现同样是"训练突然中断"，容易被误判为 OOM。
DISK_FREE_GB="$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
MIN_DISK_GB="${MIN_DISK_GB:-40}"
if [ "${DISK_FREE_GB:-0}" -lt "$MIN_DISK_GB" ]; then
  echo ""
  echo "[x] 磁盘可用空间仅 ${DISK_FREE_GB}GB（要求 ≥${MIN_DISK_GB}GB），拒绝启动"
  echo "    单个 checkpoint 约 14GB，空间不足会在训练中途崩溃。"
  echo "    占用最大的目录："
  du -sh /data/swe-rl/checkpoints* /var/lib/containerd 2>/dev/null | sort -rh | head -5 | sed 's/^/      /'
  exit 2
fi
echo "[ok] 磁盘检查通过（可用 ${DISK_FREE_GB}GB）"

# ---------- 续训起点：明确告知从哪继续，避免"以为从零跑其实在续训" ----------
#
# 这一步不只是打印信息，还做**完整性校验**：训练被磁盘写满打断时，最后一个
# checkpoint 目录可能只写了一半，而 latest_checkpointed_iteration.txt 还停在
# 上一个（实测出现过 目录有 step_30、latest 却写着 20 的情况）。
# 直接 auto 续训会去读 latest 指向的那个，是安全的；但要让人看清实际状态，
# 否则容易像我之前那样"怕不完整"就把整个目录删了，白丢几十步成果。
if [ "$RESUME_MODE" = "auto" ] && [ -d "$CKPT_DIR" ]; then
  LATEST_FILE="$CKPT_DIR/latest_checkpointed_iteration.txt"
  DIRS_FOUND="$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
  if [ -n "$DIRS_FOUND" ]; then
    LATEST_STEP="$(cat "$LATEST_FILE" 2>/dev/null || echo '?')"
    echo "[ok] 检测到已有 checkpoint：$DIRS_FOUND"
    echo "     latest 标记 = step $LATEST_STEP → 将从这里续训"
    # 校验 latest 指向的目录是否真的存在且非空
    if [ "$LATEST_STEP" != "?" ] && [ ! -s "$CKPT_DIR/global_step_$LATEST_STEP/actor/.metadata" ] \
       && [ ! -d "$CKPT_DIR/global_step_$LATEST_STEP/actor" ]; then
      echo "     ⚠️ latest 指向的目录不完整，verl 会回退或从零开始"
    fi
  else
    echo "[ok] 无已有 checkpoint，从零开始训练"
  fi
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
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
  actor_rollout_ref.actor.strategy="$ACTOR_STRATEGY" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy="$OFFLOAD_POLICY" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  ${USE_ORIG_PARAMS_ARG} \
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
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.max_actor_ckpt_to_keep="$MAX_CKPT_KEEP" \
  actor_rollout_ref.actor.checkpoint.load_contents="$LOAD_CONTENTS" \
  ${SMOKE_MAX_STEPS:+trainer.total_training_steps=$SMOKE_MAX_STEPS} \
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
