#!/bin/bash

# Foreground DeepMath SDPO run.
# Use launch-deepmath-qwen3-8B-sdpo.sh if you want to detach it with nohup.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "${SDPO_SOURCE_DOTENV:-true}" = "true" ] && [ -f "${PROJECT_ROOT}/.env" ]; then
   set -a
   source "${PROJECT_ROOT}/.env"
   set +a
fi

RUN_NAME="${RUN_NAME:-sdpo-deepmath-qwen3-8b-short-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/sdpo_deepmath/${RUN_NAME}}"
mkdir -p "$RUN_ROOT"

export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_ROOT}/scripts/models/qwen3-8B.sh}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${PROJECT_ROOT}/data/checkpoints/${MODEL_NAME}_hf_complete}"
export TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-${PROJECT_ROOT}/data/checkpoints/${MODEL_NAME}_torch_dist}"
export LOAD_CKPT="${LOAD_CKPT:-${TORCH_DIST_CKPT}}"
export MILES_CKPT="${MILES_CKPT:-${RUN_ROOT}/checkpoints}"
export PROMPT_DATA="${PROMPT_DATA:-${PROJECT_ROOT}/data/datasets/DeepMath-103K/deepmath-103k-sdpo.jsonl}"

export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export QKV_FORMAT="${QKV_FORMAT:-bshd}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"

export KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.3}"

export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-true}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28279}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/${USER:-miles}-rsdpo}"

export DUMP_DETAILS="${DUMP_DETAILS:-${RUN_ROOT}/dump_details}"
export WANDB_PROJECT="${WANDB_PROJECT:-miles-sdpo}"
export WANDB_GROUP="${WANDB_GROUP:-qwen3-8b-deepmath-sdpo}"
export WANDB_EXPERIMENT_NAME="${WANDB_EXPERIMENT_NAME:-${RUN_NAME}}"

if [ -n "${WANDB_API_KEY:-${WANDB_KEY:-${WANDB_API:-}}}" ]; then
   export USE_WANDB="${USE_WANDB:-true}"
else
   export USE_WANDB="${USE_WANDB:-false}"
fi

if [ "${SDPO_RUN_PREFLIGHT:-true}" = "true" ]; then
   bash "${PROJECT_ROOT}/examples/sdpo/preflight.sh"
fi

echo "RUN_NAME=${RUN_NAME}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "PROMPT_DATA=${PROMPT_DATA}"
echo "MILES_CKPT=${MILES_CKPT}"
echo "DUMP_DETAILS=${DUMP_DETAILS}"
echo "USE_WANDB=${USE_WANDB}"

exec bash "${PROJECT_ROOT}/examples/sdpo/run-qwen3-8B-sdpo.sh"
