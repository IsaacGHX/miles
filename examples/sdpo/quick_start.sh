#!/bin/bash

# Source this snippet or copy these arrays into an existing Miles GRPO launch script.
# The surrounding script must still provide MODEL_ARGS, CKPT_ARGS, ROLLOUT_ARGS,
# OPTIMIZER_ARGS, PERF_ARGS, SGLANG_ARGS, and the final train.py invocation.

RM_ARGS=(
   --reward-key score
   --custom-rm-path examples.sdpo.sdpo.reward_func
   --custom-reward-post-process-path examples.sdpo.sdpo.post_process_rewards
)

SDPO_ARGS=(
   --advantage-estimator on_policy_distillation
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

