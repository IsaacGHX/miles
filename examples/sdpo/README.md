# SDPO-Style Self-Distillation

This example shows the part of SDPO that can be expressed with the existing
MILES example-level hooks:

- `examples/on_policy_distillation` already passes teacher token logprobs
  through `sample.teacher_log_probs`.
- `--advantage-estimator on_policy_distillation` already trains from
  `teacher_log_probs - student_log_probs`.
- `examples/DrGRPO/custom_reducer.py` already demonstrates a custom pg-loss
  reducer if you want Dr.GRPO-style normalization.

The helper in `sdpo.py` builds SDPO-style teacher prompts from successful
same-prompt rollouts and optional reward feedback, then asks an SGLang endpoint
for logprobs on the original response tokens.

For the cleanest teacher prompts, include the original user prompt in
`metadata.raw_prompt`, `metadata.question`, or `metadata.problem`. If none is
present, the helper falls back to `sample.prompt`, which may already be a
chat-template-formatted string when `--apply-chat-template` is used.

For broad math datasets such as DeepMath, run `examples/sdpo/preflight.sh` first
so `math-verify` is available. The SDPO reward helper uses boxed exact matching,
then `math_verify` equivalence checks when installed, and falls back to the
integer-oriented DAPO verifier.

## What This Covers

This is a lightweight SDPO-style example that does not require core MILES
changes. It supports:

- self-distillation from successful rollouts in the same prompt group
- optional feedback text from the reward function
- masking samples that have no successful demonstration or feedback
- use of the current rollout SGLang router as the teacher endpoint

## What Requires Core Changes

The SDPO repo implements more than sampled-token distillation:

- full-logit or top-k KL/JSD distillation
- EMA or trust-region teacher weights colocated with the actor
- teacher forward over reprompted inputs inside the trainer

Those pieces cannot be reproduced exactly from an `examples/` hook alone.

## Quick Start

Like `examples/DrGRPO`, the fastest way to use this example is to add a few
arguments to an existing GRPO-style launch script that already works in your
environment.

Add the reward hooks:

```bash
RM_ARGS=(
   --reward-key score
   --custom-rm-path examples.sdpo.sdpo.reward_func
   --custom-reward-post-process-path examples.sdpo.sdpo.post_process_rewards
)
```

The same minimal snippet is available as `examples/sdpo/quick_start.sh`.

Switch the training signal to the OPD path:

```bash
SDPO_ARGS=(
   --advantage-estimator on_policy_distillation
   --use-kl-loss
   --kl-loss-coef "${KL_LOSS_COEF:-0.001}"
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF:-0.0}"
   --eps-clip "${EPS_CLIP:-0.2}"
   --eps-clip-high "${EPS_CLIP_HIGH:-0.3}"
)
```

Then include both arrays in your existing `train.py` command:

```bash
python3 train.py \
   ... \
   "${SDPO_ARGS[@]}" \
   "${RM_ARGS[@]}"
```

This uses the current rollout SGLang router as the self-teacher unless you
explicitly pass `--rm-url`.

Before launching Ray, run:

```bash
bash examples/sdpo/preflight.sh
```

Set `MODEL_NAME`, `MODEL_CONFIG`, `HF_CHECKPOINT`, `TORCH_DIST_CKPT`,
`PROMPT_DATA`, and `MEGATRON_PATH` if your paths differ from the defaults.
`preflight.sh` also writes `examples/sdpo/.runtime/env.sh`; source
`examples/sdpo/env.sh` in launch scripts so the same CUDA fix is reused.

## Optional Template Script

The example reward function returns a dict, so pass `--reward-key score`.

```bash
bash examples/sdpo/run-qwen3-8B-sdpo.sh
```

The local Qwen3-8B template defaults to 4 actor GPUs / TP=4. With the current
non-TE, non-Apex environment, 2 actor GPUs loads and forwards but OOMs in Adam
optimizer step.

This script is still a template. You can point it at different model,
Megatron, checkpoint, and data paths:

```bash
MODEL_NAME=Qwen3-8B \
MODEL_CONFIG=/path/to/miles/scripts/models/qwen3-8B.sh \
HF_CHECKPOINT=/path/to/Qwen3-8B \
TORCH_DIST_CKPT=/path/to/Qwen3-8B_torch_dist \
MILES_CKPT=/path/to/Qwen3-8B_miles \
PROMPT_DATA=/path/to/dapo-math-17k.jsonl \
DUMP_DETAILS=/path/to/dump_details \
MEGATRON_PATH=/path/to/Megatron-LM \
bash examples/sdpo/run-qwen3-8B-sdpo.sh
```

Set `DUMP_DETAILS` to save per-rollout debug files under
`$DUMP_DETAILS/rollout_data/{rollout_id}.pt`, together with train and logprob
debug dumps.

To log SDPO training curves to W&B, keep the API key in your shell or `.env`
and enable logging at launch time:

```bash
export WANDB_API_KEY=...
USE_WANDB=true \
WANDB_PROJECT=miles-sdpo \
WANDB_GROUP=qwen3-8b-deepmath \
WANDB_EXPERIMENT_NAME=qwen3-8b-sdpo-deepmath \
bash examples/sdpo/run-qwen3-8B-sdpo.sh
```

The script also accepts the legacy names `WANDB_KEY` and `WANDB_API`, but does
not store keys in the repo. W&B defines `rollout/*` and `perf/*` against
`rollout/step`, and `train/*` against `train/step`.

For long-context data, set the rollout context explicitly. For example, a
35K-token prompt plus an 8K-token response needs at least about 43K context
tokens after chat-template overhead:

```bash
ROLLOUT_MAX_PROMPT_LEN=35000 \
ROLLOUT_MAX_RESPONSE_LEN=8192 \
ROLLOUT_MAX_CONTEXT_LEN=45056 \
MICRO_BATCH_SIZE=1 \
bash examples/sdpo/run-qwen3-8B-sdpo.sh
```

Only set `ROLLOUT_MAX_CONTEXT_LEN` above the model's configured context window
if the model and SGLang runtime support that length.

For a new model family/version such as Qwen3.5, keep the CUDA preflight
unchanged and provide the matching Megatron model config:

```bash
MODEL_NAME=Qwen3.5-8B \
MODEL_CONFIG=/path/to/scripts/models/qwen3.5-8B.sh \
HF_CHECKPOINT=/path/to/Qwen3.5-8B \
TORCH_DIST_CKPT=/path/to/Qwen3.5-8B_torch_dist \
MILES_CKPT=/path/to/Qwen3.5-8B_miles \
bash examples/sdpo/run-qwen3-8B-sdpo.sh
```

A short smoke test that exercises rollout, self-teacher logprobs, actor train,
and checkpoint save:

```bash
NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=1 \
ROLLOUT_MAX_RESPONSE_LEN=128 GLOBAL_BATCH_SIZE=2 MICRO_BATCH_SIZE=1 \
ACTOR_NUM_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=1 TENSOR_MODEL_PARALLEL_SIZE=4 \
SAVE_INTERVAL=999 bash examples/sdpo/run-qwen3-8B-sdpo.sh
```

Useful environment overrides:

```bash
export SDPO_SUCCESS_REWARD_THRESHOLD=0.5
export SDPO_INCLUDE_ENVIRONMENT_FEEDBACK=true
export SDPO_ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=true
export SDPO_DONT_REPROMPT_ON_SELF_SUCCESS=true
export SDPO_MAX_REPROMPT_LEN=10240
```

To try Dr.GRPO-style pg-loss normalization, add:

```bash
--custom-pg-loss-reducer-function-path examples.DrGRPO.custom_reducer:get_pg_loss_reducer
```
