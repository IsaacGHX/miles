#!/bin/bash

# Detached DeepMath SDPO launcher.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_NAME="${RUN_NAME:-sdpo-deepmath-qwen3-8b-short-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/sdpo_deepmath/${RUN_NAME}}"
mkdir -p "$RUN_ROOT"

export RUN_NAME
export RUN_ROOT

nohup bash "${PROJECT_ROOT}/examples/sdpo/run-deepmath-qwen3-8B-sdpo.sh" > "${RUN_ROOT}/run.log" 2>&1 &
pid=$!
echo "$pid" > "${RUN_ROOT}/pid"

echo "Started SDPO DeepMath run."
echo "PID: ${pid}"
echo "RUN_NAME: ${RUN_NAME}"
echo "RUN_ROOT: ${RUN_ROOT}"
echo "Log: ${RUN_ROOT}/run.log"
echo "Tail: tail -f ${RUN_ROOT}/run.log"
