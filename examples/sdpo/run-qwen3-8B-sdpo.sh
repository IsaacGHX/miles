#!/bin/bash

# usage:
#   bash examples/sdpo/run-qwen3-8B-sdpo.sh
#   NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=1 \
#     ROLLOUT_MAX_RESPONSE_LEN=128 GLOBAL_BATCH_SIZE=2 SAVE_INTERVAL=1 \
#     bash examples/sdpo/run-qwen3-8B-sdpo.sh

set -euo pipefail

if [ "${SDPO_XTRACE:-false}" = "true" ]; then
   set -x
fi

export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/examples/sdpo/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_ROOT}/scripts/models/qwen3-8B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${PROJECT_ROOT}/data/checkpoints/${MODEL_NAME}_hf_complete}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-${PROJECT_ROOT}/data/checkpoints/${MODEL_NAME}_torch_dist}"
MILES_CKPT="${MILES_CKPT:-${PROJECT_ROOT}/data/checkpoints/${MODEL_NAME}_miles}"
LOAD_CKPT="${LOAD_CKPT:-}"
PROMPT_DATA="${PROMPT_DATA:-${PROJECT_ROOT}/examples/sdpo/data/smoke_math.jsonl}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.3}"
QKV_FORMAT="${QKV_FORMAT:-bshd}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-false}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"
SGLANG_RL_ON_POLICY_TARGET="${SGLANG_RL_ON_POLICY_TARGET:-fsdp}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-true}"
DUMP_DETAILS="${DUMP_DETAILS:-}"
SAVE_DEBUG_ROLLOUT_DATA="${SAVE_DEBUG_ROLLOUT_DATA:-}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/${USER:-miles}-ray}"
mkdir -p "$RAY_TMPDIR"

if [ ! -f "$MODEL_CONFIG" ]; then
   echo "MODEL_CONFIG not found: $MODEL_CONFIG"
   echo "Set MODEL_CONFIG=/path/to/scripts/models/<model>.sh for ${MODEL_NAME}."
   exit 1
fi

if [ ! -e "$HF_CHECKPOINT" ]; then
   echo "HF_CHECKPOINT not found: $HF_CHECKPOINT"
   echo "Set HF_CHECKPOINT=/path/to/${MODEL_NAME}."
   exit 1
fi

if [ ! -e "$TORCH_DIST_CKPT" ]; then
   echo "TORCH_DIST_CKPT not found: $TORCH_DIST_CKPT"
   echo "Set TORCH_DIST_CKPT=/path/to/${MODEL_NAME}_torch_dist."
   exit 1
fi

if [ -z "$LOAD_CKPT" ]; then
   if [ -e "$MILES_CKPT" ]; then
      LOAD_CKPT="$MILES_CKPT"
   else
      LOAD_CKPT="$TORCH_DIST_CKPT"
   fi
fi

if [ ! -e "$LOAD_CKPT" ]; then
   echo "LOAD_CKPT not found: $LOAD_CKPT"
   echo "Set LOAD_CKPT=/path/to/an existing Megatron checkpoint."
   exit 1
fi

if [ ! -f "$PROMPT_DATA" ]; then
   echo "PROMPT_DATA not found: $PROMPT_DATA"
   echo "Set PROMPT_DATA=/path/to/dapo-math-17k.jsonl or another jsonl with prompt/label fields."
   exit 1
fi

source "$MODEL_CONFIG"

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --ref-load "$TORCH_DIST_CKPT"
   --load "$LOAD_CKPT"
   --save "$MILES_CKPT"
   --save-interval "$SAVE_INTERVAL"
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
   --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
   --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
   --rollout-temperature "$ROLLOUT_TEMPERATURE"

   --global-batch-size "$GLOBAL_BATCH_SIZE"
   --balance-data
)

if [ -n "$ROLLOUT_MAX_CONTEXT_LEN" ]; then
   ROLLOUT_ARGS+=(--rollout-max-context-len "$ROLLOUT_MAX_CONTEXT_LEN")
fi

if [ -n "$ROLLOUT_MAX_PROMPT_LEN" ]; then
   ROLLOUT_ARGS+=(--rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN")
fi

RM_ARGS=(
   --reward-key score
   --custom-rm-path examples.sdpo.sdpo.reward_func
   --custom-reward-post-process-path examples.sdpo.sdpo.post_process_rewards
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --qkv-format "$QKV_FORMAT"

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

if [ "$QKV_FORMAT" = "bshd" ]; then
   PERF_ARGS+=(--micro-batch-size "$MICRO_BATCH_SIZE")
else
   PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU")
fi

if [ "$SEQUENCE_PARALLEL" = "true" ]; then
   PERF_ARGS+=(--sequence-parallel)
fi

SDPO_ARGS=(
   --advantage-estimator on_policy_distillation
   --use-kl-loss
   --kl-loss-coef "$KL_LOSS_COEF"
   --kl-loss-type low_var_kl
   --entropy-coef "$ENTROPY_COEF"
   --eps-clip "$EPS_CLIP"
   --eps-clip-high "$EPS_CLIP_HIGH"

   # Optional Dr.GRPO-style pg-loss reducer:
   # --custom-pg-loss-reducer-function-path examples.DrGRPO.custom_reducer:get_pg_loss_reducer
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SDPO_WANDB_RESTORE_XTRACE=false
case "$-" in
   *x*)
      SDPO_WANDB_RESTORE_XTRACE=true
      set +x
      ;;
esac

USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-miles-sdpo}"
WANDB_GROUP="${WANDB_GROUP:-${MODEL_NAME}-sdpo}"
WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_DIR="${WANDB_DIR:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-${WANDB_API:-}}}"
DISABLE_WANDB_RANDOM_SUFFIX="${DISABLE_WANDB_RANDOM_SUFFIX:-true}"

if [ -n "$WANDB_API_KEY" ]; then
   export WANDB_API_KEY
fi

if [ "$SDPO_WANDB_RESTORE_XTRACE" = "true" ]; then
   set -x
fi

WANDB_ARGS=()
if [ "$USE_WANDB" = "true" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-group "$WANDB_GROUP"
   )

   if [ -n "$WANDB_EXPERIMENT_NAME" ]; then
      WANDB_ARGS+=(--wandb-experiment-name "$WANDB_EXPERIMENT_NAME")
   fi

   if [ -n "$WANDB_RUN_ID" ]; then
      WANDB_ARGS+=(--wandb-run-id "$WANDB_RUN_ID")
   fi

   if [ -n "$WANDB_DIR" ]; then
      WANDB_ARGS+=(--wandb-dir "$WANDB_DIR")
   fi

   if [ -n "$WANDB_MODE" ]; then
      WANDB_ARGS+=(--wandb-mode "$WANDB_MODE")
   fi

   if [ "$DISABLE_WANDB_RANDOM_SUFFIX" = "true" ]; then
      WANDB_ARGS+=(--disable-wandb-random-suffix)
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE"
   --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
)

if [ -n "$SGLANG_RL_ON_POLICY_TARGET" ]; then
   SGLANG_ARGS+=(--sglang-rl-on-policy-target "$SGLANG_RL_ON_POLICY_TARGET")
fi

if [ "$SGLANG_DISABLE_CUDA_GRAPH" = "true" ]; then
   SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --no-masked-softmax-fusion
   --no-rope-fusion
   --transformer-impl local
   --no-persist-layer-norm
   --no-gradient-accumulation-fusion
   --megatron-to-hf-mode raw
)

if [ -n "${ATTENTION_BACKEND:-}" ]; then
   MISC_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

DEBUG_ARGS=()
if [ -n "$DUMP_DETAILS" ]; then
   mkdir -p "$DUMP_DETAILS"
   DEBUG_ARGS+=(--dump-details "$DUMP_DETAILS")
fi

if [ -n "$SAVE_DEBUG_ROLLOUT_DATA" ]; then
   mkdir -p "$(dirname "$SAVE_DEBUG_ROLLOUT_DATA")"
   DEBUG_ARGS+=(--save-debug-rollout-data "$SAVE_DEBUG_ROLLOUT_DATA")
fi

export SDPO_SUCCESS_REWARD_THRESHOLD=${SDPO_SUCCESS_REWARD_THRESHOLD:-0.5}
export SDPO_INCLUDE_ENVIRONMENT_FEEDBACK=${SDPO_INCLUDE_ENVIRONMENT_FEEDBACK:-true}
export SDPO_ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=${SDPO_ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION:-true}
export SDPO_DONT_REPROMPT_ON_SELF_SUCCESS=${SDPO_DONT_REPROMPT_ON_SELF_SUCCESS:-true}
export SDPO_MAX_REPROMPT_LEN=${SDPO_MAX_REPROMPT_LEN:-10240}

cleanup_ray() {
   local status=$?
   if [ "${RAY_STOP_ON_EXIT:-true}" = "true" ]; then
      ray stop --force || true
   fi

   if [ "${RAY_KILL_ON_EXIT:-false}" = "true" ]; then
      pkill -9 ray || true
   fi
   exit "$status"
}
trap cleanup_ray EXIT

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "$NUM_GPUS_PER_NODE" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT" --temp-dir "$RAY_TMPDIR"

SDPO_RESTORE_XTRACE=false
case "$-" in
   *x*)
      SDPO_RESTORE_XTRACE=true
      set +x
      ;;
esac

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${PROJECT_ROOT}:${MEGATRON_PATH}:${PYTHONPATH:-}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"CUDA_HOME\":\"${CUDA_HOME:-}\",\"CUDA_PATH\":\"${CUDA_PATH:-}\",\"LD_LIBRARY_PATH\":\"${LD_LIBRARY_PATH:-}\",\"PYTORCH_ALLOC_CONF\":\"${PYTORCH_ALLOC_CONF:-expandable_segments:True}\",\"SGLANG_ENABLE_JIT_DEEPGEMM\":\"${SGLANG_ENABLE_JIT_DEEPGEMM:-0}\",\"SGLANG_ENABLE_JIT_KVCACHE\":\"${SGLANG_ENABLE_JIT_KVCACHE:-0}\",\"SGLANG_JIT_DEEPGEMM_PRECOMPILE\":\"${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}\",\"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM\":\"${SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM:-0}\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_MODE\":\"${WANDB_MODE:-}\"}}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
   --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SDPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${DEBUG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

if [ "$SDPO_RESTORE_XTRACE" = "true" ]; then
   set -x
fi
