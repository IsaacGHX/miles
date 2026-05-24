#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDPO_DIR="${PROJECT_ROOT}/examples/sdpo"
RUNTIME_DIR="${SDPO_RUNTIME_DIR:-${SDPO_DIR}/.runtime}"
RUNTIME_ENV_FILE="${SDPO_RUNTIME_ENV_FILE:-${RUNTIME_DIR}/env.sh}"
CUDA_LINK="${SDPO_CUDA_LINK:-${RUNTIME_DIR}/cuda}"
CUDA_WRAPPER="${RUNTIME_DIR}/cuda-toolkit-wrapper"
CUDA_LIB64_LINKS="${RUNTIME_DIR}/cuda-lib64"
PYTHON_CUDA_SHIM="${RUNTIME_DIR}/python-cuda-shim"

mkdir -p "$RUNTIME_DIR"

ok() {
   echo "[ok] $*"
}

warn() {
   echo "[warn] $*"
}

missing() {
   echo "[missing] $*"
   fail=1
}

is_writable_dir() {
   [ -n "$1" ] && mkdir -p "$1" 2>/dev/null && [ -w "$1" ]
}

ensure_writable_cache_env() {
   if ! is_writable_dir "${XDG_CACHE_HOME:-}"; then
      export XDG_CACHE_HOME="/tmp/${USER:-miles}-cache"
   fi
   if ! is_writable_dir "${HF_HOME:-}"; then
      export HF_HOME="${XDG_CACHE_HOME}/huggingface"
   fi
   if ! is_writable_dir "${TRANSFORMERS_CACHE:-}"; then
      export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
   fi
   if ! is_writable_dir "${FLASHINFER_WORKSPACE_BASE:-}"; then
      export FLASHINFER_WORKSPACE_BASE="${XDG_CACHE_HOME}"
   fi
   if ! is_writable_dir "${UV_CACHE_DIR:-}"; then
      export UV_CACHE_DIR="/tmp/uv-cache"
   fi
   mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$FLASHINFER_WORKSPACE_BASE" "$UV_CACHE_DIR"
}

readlink_f() {
   readlink -f "$1" 2>/dev/null || python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

add_candidate() {
   local path="${1:-}"
   [ -n "$path" ] || return 0
   CUDA_CANDIDATES+=("$path")
}

collect_cuda_candidates() {
   CUDA_CANDIDATES=()
   add_candidate "${CUDA_HOME:-}"
   add_candidate "${SDPO_CUDA_HOME:-}"
   add_candidate "${SDPO_RESOLVED_CUDA_HOME:-}"
   add_candidate "${CUDA13_HOME:-}"
   add_candidate "${CUDA12_HOME:-}"
   add_candidate "$CUDA_LINK"
   add_candidate /usr/local/cuda-13
   add_candidate /usr/local/cuda-13.0
   add_candidate /usr/local/cuda
   add_candidate /usr/local/cuda-12.4

   local candidate
   for candidate in "${PROJECT_ROOT}"/.venv/lib/python*/site-packages/nvidia/cu13; do
      add_candidate "$candidate"
   done
   for candidate in "${PROJECT_ROOT}"/.venv/lib/python*/site-packages/nvidia/cuda_nvcc; do
      add_candidate "$candidate"
   done
   for candidate in "${PROJECT_ROOT}"/.venv/lib/python*/site-packages/nvidia/cuda_runtime; do
      add_candidate "$candidate"
   done

   if command -v nvcc >/dev/null 2>&1; then
      local nvcc_bin
      nvcc_bin="$(command -v nvcc)"
      add_candidate "$(cd "$(dirname "$nvcc_bin")/.." && pwd)"
   fi
}

is_full_cuda_toolkit() {
   local path="$1"
   [ -d "${path}/include" ] && [ -d "${path}/lib" ] && [ -x "${path}/bin/nvcc" ] && cuda_nvcc_matches_headers "$path"
}

cuda_headers_are_complete() {
   local path="$1"
   if [ ! -f "${path}/include/cuda_fp16.h" ]; then
      return 0
   fi
   if grep -q '#include <nv/target>' "${path}/include/cuda_fp16.h" && [ ! -e "${path}/include/nv/target" ]; then
      return 1
   fi
   return 0
}

cuda_version_number() {
   local path="$1"
   if [ -f "${path}/include/cuda.h" ]; then
      sed -n 's/^#define CUDA_VERSION[[:space:]]\+\([0-9]\+\).*/\1/p' "${path}/include/cuda.h" | head -n 1
   fi
}

cuda_version_major() {
   local version
   version="$(cuda_version_number "$1")"
   [ -n "$version" ] || return 1
   echo $((version / 1000))
}

nvcc_version_major() {
   local path="$1"
   if [ -x "${path}/bin/nvcc" ]; then
      "${path}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\).*/\1/p' | head -n 1
   fi
}

cuda_nvcc_matches_headers() {
   local path="$1"
   local cuda_major nvcc_major
   cuda_major="$(cuda_version_major "$path" 2>/dev/null || true)"
   nvcc_major="$(nvcc_version_major "$path" 2>/dev/null || true)"
   if [ -z "$cuda_major" ] || [ -z "$nvcc_major" ]; then
      return 0
   fi
   [ "$nvcc_major" -le "$cuda_major" ]
}

driver_cuda_major() {
   if [ -n "${SDPO_DRIVER_CUDA_MAJOR:-}" ]; then
      echo "$SDPO_DRIVER_CUDA_MAJOR"
      return 0
   fi
   if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9]\+\).*/\1/p' | head -n 1
   fi
}

cuda_toolkit_is_supported() {
   local path="$1"
   local cuda_major driver_major

   if [ "${SDPO_ALLOW_UNSUPPORTED_CUDA_TOOLKIT:-false}" = "true" ]; then
      return 0
   fi

   cuda_major="$(cuda_version_major "$path" 2>/dev/null || true)"
   driver_major="$(driver_cuda_major 2>/dev/null || true)"
   if [ -z "$cuda_major" ] || [ -z "$driver_major" ]; then
      return 0
   fi

   # CUDA minor-version compatibility lets CUDA 12.x runtimes run on 525+ class
   # drivers, but a CUDA 13 runtime/toolchain is not compatible with a driver
   # reporting CUDA 12.x support.
   [ "$cuda_major" -le "$driver_major" ]
}

find_cuda_runtime() {
   collect_cuda_candidates
   local candidate real_runtime real_candidate
   real_runtime="$(readlink_f "$RUNTIME_DIR")"
   for candidate in "${CUDA_CANDIDATES[@]}"; do
      real_candidate="$(readlink_f "$candidate")"
      case "$real_candidate" in
         "$real_runtime"/*) continue ;;
      esac
      if [ -d "${candidate}/include" ] && [ -d "${candidate}/lib" ] && cuda_toolkit_is_supported "$candidate"; then
         readlink_f "$candidate"
         return 0
      fi
   done
   return 1
}

find_full_cuda_toolkit() {
   collect_cuda_candidates
   local candidate
   for candidate in "${CUDA_CANDIDATES[@]}"; do
      if is_full_cuda_toolkit "$candidate" && cuda_toolkit_is_supported "$candidate"; then
         readlink_f "$candidate"
         return 0
      elif is_full_cuda_toolkit "$candidate" && ! cuda_toolkit_is_supported "$candidate"; then
         warn "Skip unsupported CUDA toolkit for this driver: ${candidate}"
      fi
   done
   return 1
}

build_python_cuda_shim() {
   local site nvcc_root runtime_root
   for site in "${PROJECT_ROOT}"/.venv/lib/python*/site-packages/nvidia; do
      [ -d "$site" ] || continue

      if is_full_cuda_toolkit "${site}/cu13" && cuda_toolkit_is_supported "${site}/cu13"; then
         readlink_f "${site}/cu13"
         return 0
      fi

      nvcc_root="${site}/cuda_nvcc"
      runtime_root="${site}/cuda_runtime"
      if [ -x "${nvcc_root}/bin/nvcc" ] && [ -d "${runtime_root}/include" ] && [ -d "${runtime_root}/lib" ]; then
         mkdir -p "$PYTHON_CUDA_SHIM"
         ln -sfnT "${runtime_root}/include" "${PYTHON_CUDA_SHIM}/include"
         ln -sfnT "${runtime_root}/lib" "${PYTHON_CUDA_SHIM}/lib"
         ln -sfnT "${nvcc_root}/bin" "${PYTHON_CUDA_SHIM}/bin"
         if [ -d "${nvcc_root}/nvvm" ]; then
            ln -sfnT "${nvcc_root}/nvvm" "${PYTHON_CUDA_SHIM}/nvvm"
         fi
         readlink_f "$PYTHON_CUDA_SHIM"
         return 0
      fi
   done
   return 1
}

install_python_nvcc_if_needed() {
   if [ "${SDPO_PREFLIGHT_INSTALL_NVCC:-auto}" = "false" ]; then
      warn "SDPO_PREFLIGHT_INSTALL_NVCC=false; skip nvcc package install."
      return 1
   fi
   if ! command -v uv >/dev/null 2>&1; then
      warn "uv not found; cannot auto-install Python nvcc package."
      return 1
   fi

   local packages=()
   if [ -n "${SDPO_NVCC_PACKAGES:-}" ]; then
      # shellcheck disable=SC2206
      packages=(${SDPO_NVCC_PACKAGES})
   elif [ "$(driver_cuda_major 2>/dev/null || true)" = "12" ]; then
      local version="${SDPO_NVCC_CU12_VERSION:-12.9.86}"
      local cccl_version="${SDPO_CUDA_CCCL_CU12_VERSION:-12.9.27}"
      packages=(
         "nvidia-cuda-nvcc-cu12==${version}"
         "nvidia-cuda-cccl-cu12==${cccl_version}"
      )
      warn "CUDA 12 pip nvcc package provides ptxas/nvvm pieces on this platform; a full CUDA 12 toolkit may still be required for nvcc."
   elif compgen -G "${PROJECT_ROOT}/.venv/lib/python*/site-packages/nvidia/cu13/include/cuda.h" >/dev/null; then
      local version="${SDPO_NVCC_VERSION:-13.0.88}"
      local cccl_version="${SDPO_CUDA_CCCL_VERSION:-13.0.85}"
      packages=(
         "nvidia-cuda-nvcc==${version}"
         "nvidia-nvvm==${version}"
         "nvidia-cuda-crt==${version}"
         "nvidia-cuda-cccl==${cccl_version}"
      )
   else
      local version="${SDPO_NVCC_VERSION:-12.8.93}"
      packages=("nvidia-cuda-nvcc-cu12==${version}")
   fi

   warn "No usable nvcc found; installing ${packages[*]}"
   UV_LINK_MODE="${UV_LINK_MODE:-copy}" UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install "${packages[@]}"
}

install_python_cuda_headers_if_needed() {
   if [ "${SDPO_PREFLIGHT_INSTALL_NVCC:-auto}" = "false" ]; then
      warn "SDPO_PREFLIGHT_INSTALL_NVCC=false; skip CUDA header package install."
      return 1
   fi
   if ! command -v uv >/dev/null 2>&1; then
      warn "uv not found; cannot auto-install CUDA header package."
      return 1
   fi

   local packages=()
   if [ -n "${SDPO_CUDA_HEADER_PACKAGES:-}" ]; then
      # shellcheck disable=SC2206
      packages=(${SDPO_CUDA_HEADER_PACKAGES})
   elif [ "$(driver_cuda_major 2>/dev/null || true)" = "12" ] && compgen -G "${PROJECT_ROOT}/.venv/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda.h" >/dev/null; then
      packages=("nvidia-cuda-cccl-cu12==${SDPO_CUDA_CCCL_CU12_VERSION:-12.9.27}")
   elif compgen -G "${PROJECT_ROOT}/.venv/lib/python*/site-packages/nvidia/cu13/include/cuda.h" >/dev/null; then
      packages=("nvidia-cuda-cccl==${SDPO_CUDA_CCCL_VERSION:-13.0.85}")
   elif compgen -G "${PROJECT_ROOT}/.venv/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda.h" >/dev/null; then
      packages=("nvidia-cuda-cccl-cu12==${SDPO_CUDA_CCCL_CU12_VERSION:-12.9.27}")
   else
      warn "CUDA headers look incomplete, but no known Python header package matched this CUDA layout."
      return 1
   fi

   warn "CUDA headers incomplete; installing ${packages[*]}"
   UV_LINK_MODE="${UV_LINK_MODE:-copy}" UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install "${packages[@]}"
}

install_math_verify_if_needed() {
   if python - <<'PY' >/dev/null 2>&1
import math_verify
PY
   then
      ok "math-verify import"
      return 0
   fi

   if [ "${SDPO_PREFLIGHT_INSTALL_MATH_VERIFY:-auto}" = "false" ]; then
      warn "math-verify is not installed; SDPO_PREFLIGHT_INSTALL_MATH_VERIFY=false."
      return 1
   fi
   if ! command -v uv >/dev/null 2>&1; then
      warn "math-verify is not installed and uv was not found."
      return 1
   fi

   local version="${SDPO_MATH_VERIFY_VERSION:-0.8.0}"
   warn "math-verify is not installed; installing math-verify[antlr4_9_3]==${version}"
   UV_LINK_MODE="${UV_LINK_MODE:-copy}" UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install "math-verify[antlr4_9_3]==${version}"
}

prepare_cuda_link() {
   local cuda_root="$1"

   if [ -e "${cuda_root}/lib64/libcudart.so" ]; then
      ln -sfnT "$cuda_root" "$CUDA_LINK"
      return 0
   fi

   mkdir -p "$CUDA_WRAPPER"
   rm -rf "${CUDA_WRAPPER}/include" "${CUDA_WRAPPER}/lib" "${CUDA_WRAPPER}/lib64" "${CUDA_WRAPPER}/bin" "${CUDA_WRAPPER}/nvvm" "$CUDA_LIB64_LINKS"
   mkdir -p "$CUDA_LIB64_LINKS"
   if [ -d "${cuda_root}/include" ]; then
      ln -sfnT "${cuda_root}/include" "${CUDA_WRAPPER}/include"
   fi
   if [ -d "${cuda_root}/lib" ]; then
      ln -sfnT "${cuda_root}/lib" "${CUDA_WRAPPER}/lib"
      ln -sfnT "$CUDA_LIB64_LINKS" "${CUDA_WRAPPER}/lib64"

      local lib base soname
      for lib in "${cuda_root}"/lib/*.so*; do
         [ -e "$lib" ] || continue
         base="$(basename "$lib")"
         ln -sfn "$lib" "${CUDA_LIB64_LINKS}/${base}"
         if [[ "$base" == *.so.* ]]; then
            soname="${base%%.so.*}.so"
            ln -sfn "$lib" "${CUDA_LIB64_LINKS}/${soname}"
         fi
      done
   fi
   if [ -d "${cuda_root}/bin" ]; then
      ln -sfnT "${cuda_root}/bin" "${CUDA_WRAPPER}/bin"
   fi
   if [ -d "${cuda_root}/nvvm" ]; then
      ln -sfnT "${cuda_root}/nvvm" "${CUDA_WRAPPER}/nvvm"
   fi
   ln -sfnT "$CUDA_WRAPPER" "$CUDA_LINK"
}

write_runtime_env() {
   local cuda_root="$1"
   local enable_jit="${2:-1}"
   local enable_batch_deepgemm="${3:-0}"
   prepare_cuda_link "$cuda_root"

   cat > "$RUNTIME_ENV_FILE" <<EOF
# Generated by examples/sdpo/preflight.sh. Safe to regenerate.
export CUDA_HOME="$CUDA_LINK"
export CUDA_PATH="\${CUDA_PATH:-\$CUDA_HOME}"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib:\${LD_LIBRARY_PATH:-}"
if [ -d "\$CUDA_HOME/lib64" ]; then
   export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
fi
export PYTORCH_ALLOC_CONF="\${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_ENABLE_JIT_DEEPGEMM="\${SDPO_SGLANG_ENABLE_JIT_DEEPGEMM:-$enable_jit}"
export SGLANG_ENABLE_JIT_KVCACHE="\${SDPO_SGLANG_ENABLE_JIT_KVCACHE:-$enable_jit}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="\${SDPO_SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM="\${SDPO_SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM:-$enable_batch_deepgemm}"
export SDPO_RESOLVED_CUDA_HOME="$cuda_root"
export SDPO_HAS_SUPPORTED_NVCC="$enable_jit"
EOF

   source "$RUNTIME_ENV_FILE"
}

ensure_cuda_toolkit() {
   local cuda_root=""

   if cuda_root="$(find_full_cuda_toolkit)"; then
      ok "Found CUDA toolkit with nvcc: ${cuda_root}"
   elif cuda_root="$(build_python_cuda_shim)"; then
      ok "Built Python CUDA shim with nvcc: ${cuda_root}"
   else
      install_python_nvcc_if_needed || true
      unset CUDA_HOME
      source "${SDPO_DIR}/env.sh"
      if cuda_root="$(find_full_cuda_toolkit)"; then
         ok "Found CUDA toolkit with nvcc after install: ${cuda_root}"
      elif cuda_root="$(build_python_cuda_shim)"; then
         ok "Built Python CUDA shim with nvcc after install: ${cuda_root}"
      else
         local runtime_root=""
         if runtime_root="$(find_cuda_runtime)"; then
            write_runtime_env "$runtime_root" 0 0
            warn "No driver-compatible full CUDA toolkit with nvcc was found."
            warn "Generated runtime-only CUDA env and disabled SGLang JIT. Set SDPO_CUDA_HOME to a full CUDA 12 toolkit to enable JIT on this driver."
            [ "${SDPO_REQUIRE_SGLANG_JIT:-false}" = "true" ] && return 1
            return 0
         fi
         warn "CUDA runtime is available but nvcc is still missing."
         warn "SGLang JIT kernels will be disabled unless you set CUDA_HOME to a full toolkit."
         return 1
      fi
   fi

   if ! cuda_headers_are_complete "$cuda_root"; then
      install_python_cuda_headers_if_needed || true
      if cuda_root="$(find_full_cuda_toolkit)"; then
         ok "Found CUDA toolkit after header install: ${cuda_root}"
      elif cuda_root="$(build_python_cuda_shim)"; then
         ok "Built Python CUDA shim after header install: ${cuda_root}"
      fi
   fi

   if ! cuda_headers_are_complete "$cuda_root"; then
      warn "CUDA headers are still incomplete for DeepGEMM JIT: ${cuda_root}"
      return 1
   fi

   write_runtime_env "$cuda_root" 1 "${SDPO_ENABLE_BATCH_INVARIANT_DEEPGEMM:-0}"
   ok "Wrote runtime env: ${RUNTIME_ENV_FILE}"
   ok "CUDA_HOME symlink: ${CUDA_LINK} -> $(readlink_f "$CUDA_LINK")"
   ok "Resolved CUDA toolkit: ${cuda_root}"
}

test_nvcc_compile() {
   if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
      warn "nvcc compile test skipped; CUDA_HOME has no nvcc."
      return 1
   fi

   local tmpdir
   tmpdir="$(mktemp -d /tmp/sdpo-nvcc-test.XXXXXX)"
   printf '%s\n' '#include <cuda_fp16.h>' 'extern "C" __global__ void k(float* x) { x[threadIdx.x] += 1.0f; }' > "${tmpdir}/test.cu"
   if "${CUDA_HOME}/bin/nvcc" -std=c++17 -gencode=arch=compute_90,code=sm_90 -c "${tmpdir}/test.cu" -o "${tmpdir}/test.o" >/tmp/sdpo-nvcc-test.log 2>&1 && \
      c++ "${tmpdir}/test.o" -shared -L"${CUDA_HOME}/lib64" -lcudart -o "${tmpdir}/test.so" >>/tmp/sdpo-nvcc-test.log 2>&1; then
      ok "nvcc sm_90 compile test passed: ${CUDA_HOME}/bin/nvcc"
      return 0
   fi

   warn "nvcc compile test failed:"
   sed -n '1,80p' /tmp/sdpo-nvcc-test.log
   return 1
}

patch_sglang_deepgemm_small_batch_guard() {
   python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("sglang.srt.batch_invariant_ops.batch_invariant_ops")
if spec is None or spec.origin is None:
    print("[warn] sglang batch_invariant_ops module not found; skip DeepGEMM guard patch.")
    raise SystemExit(0)

path = Path(spec.origin)
text = path.read_text()
new_assign = """def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    M, K = a.shape
    K, N = b.shape
"""
new = """and M >= MIN_DEEPGEMM_DIM
        and N >= MIN_DEEPGEMM_DIM
        and K >= MIN_DEEPGEMM_DIM
    ):"""
if new in text and new_assign in text:
    print(f"[ok] SGLang DeepGEMM small-batch guard already patched: {path}")
    raise SystemExit(0)

old_assign = """def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    K, N = b.shape
"""
if new_assign not in text:
    if old_assign not in text:
        print(f"[warn] SGLang matmul_persistent shape pattern not found; please inspect {path}")
        raise SystemExit(1)
    text = text.replace(old_assign, new_assign, 1)

old = """and N >= MIN_DEEPGEMM_DIM
    ):"""
if new not in text and old not in text:
    print(f"[warn] SGLang DeepGEMM guard pattern not found; please inspect {path}")
    raise SystemExit(1)

if new not in text:
    text = text.replace(old, new, 1)
text = text.replace("MIN_DEEPGEMM_DIM = 16", "MIN_DEEPGEMM_DIM = 64", 1)
path.write_text(text)
print(f"[ok] Patched SGLang DeepGEMM small-batch guard: {path}")
PY
}

patch_sglang_kvcache_jit_guard() {
   python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("sglang.jit_kernel.kvcache")
if spec is None or spec.origin is None:
    print("[warn] sglang kvcache module not found; skip KV-cache JIT guard patch.")
    raise SystemExit(0)

path = Path(spec.origin)
text = path.read_text()
if "SGLANG_ENABLE_JIT_KVCACHE" in text:
    print(f"[ok] SGLang KV-cache JIT env guard already patched: {path}")
    raise SystemExit(0)

if "import logging\n" not in text:
    print(f"[warn] SGLang kvcache import pattern not found; please inspect {path}")
    raise SystemExit(1)
text = text.replace("import logging\n", "import logging\nimport os\n", 1)

old = """def can_use_store_cache(size: int) -> bool:
    logger = logging.getLogger(__name__)
"""
new = """def can_use_store_cache(size: int) -> bool:
    logger = logging.getLogger(__name__)
    if os.environ.get("SGLANG_ENABLE_JIT_KVCACHE", "1").lower() in ("0", "false", "no", "off"):
        return False
"""
if old not in text:
    print(f"[warn] SGLang kvcache guard pattern not found; please inspect {path}")
    raise SystemExit(1)

path.write_text(text.replace(old, new, 1))
print(f"[ok] Patched SGLang KV-cache JIT env guard: {path}")
PY
}

ensure_writable_cache_env
source "${SDPO_DIR}/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_ROOT}/scripts/models/qwen3-8B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${PROJECT_ROOT}/data/checkpoints/${MODEL_NAME}_hf_complete}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-${PROJECT_ROOT}/data/checkpoints/${MODEL_NAME}_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${PROJECT_ROOT}/examples/sdpo/data/smoke_math.jsonl}"

export MODEL_NAME MODEL_CONFIG HF_CHECKPOINT TORCH_DIST_CKPT PROMPT_DATA

fail=0

ensure_cuda_toolkit || fail=1
if [ "${SDPO_HAS_SUPPORTED_NVCC:-0}" = "1" ]; then
   test_nvcc_compile || fail=1
elif [ "${SDPO_REQUIRE_SGLANG_JIT:-false}" = "true" ]; then
   missing "driver-compatible CUDA toolkit with nvcc"
else
   warn "nvcc compile test skipped; SGLang JIT is disabled for this runtime."
fi
patch_sglang_deepgemm_small_batch_guard || fail=1
patch_sglang_kvcache_jit_guard || fail=1
install_math_verify_if_needed || warn "DeepMath expression rewards will fall back to exact/DAPO checks."

check_path() {
   local label="$1"
   local path="$2"
   if [ -e "$path" ]; then
      ok "${label}: ${path}"
   else
      missing "${label}: ${path}"
   fi
}

check_path "MODEL_CONFIG" "$MODEL_CONFIG"
check_path "PROMPT_DATA" "$PROMPT_DATA"
check_path "MEGATRON_PATH" "$MEGATRON_PATH"
check_path "HF_CHECKPOINT" "$HF_CHECKPOINT"
check_path "TORCH_DIST_CKPT" "$TORCH_DIST_CKPT"

if [ "${SDPO_HAS_SUPPORTED_NVCC:-0}" = "1" ] && [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
   ok "CUDA_HOME has nvcc: ${CUDA_HOME}"
   "${CUDA_HOME}/bin/nvcc" --version | tail -n 1
elif [ -n "${CUDA_HOME:-}" ]; then
   warn "CUDA_HOME has runtime headers/libs but no nvcc: ${CUDA_HOME}"
else
   warn "CUDA_HOME is not set."
fi

python - <<'PY' || fail=1
import json
import os
from pathlib import Path

path = Path(os.environ.get("PROMPT_DATA", "examples/sdpo/data/smoke_math.jsonl"))
with path.open() as f:
    rows = [json.loads(line) for line in f if line.strip()]

missing = [i for i, row in enumerate(rows) if "prompt" not in row or "label" not in row]
if missing:
    raise SystemExit(f"PROMPT_DATA rows missing prompt/label: {missing[:10]}")
print(f"[ok] PROMPT_DATA jsonl rows: {len(rows)}")
PY

python - <<'PY' || fail=1
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.server_args import ServerArgs
from megatron.training.arguments import parse_args
import examples.sdpo.sdpo
import train

print("[ok] sglang/megatron/miles imports")
try:
    import math_verify  # noqa: F401
    print("[ok] math_verify import")
except Exception as exc:
    print(f"[warn] math_verify import failed: {exc}")
PY

python - <<'PY' || fail=1
import os
import deep_gemm

print(f"[ok] deep_gemm import with CUDA_HOME={os.environ.get('CUDA_HOME')}")
PY

python - <<'PY' || fail=1
import os
import torch

print(f"[info] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[info] torch.cuda.is_available={torch.cuda.is_available()}")
print(f"[info] torch.cuda.device_count={torch.cuda.device_count()}")
print(f"[info] CUDA_HOME={os.environ.get('CUDA_HOME')}")
print(f"[info] SGLANG_ENABLE_JIT_DEEPGEMM={os.environ.get('SGLANG_ENABLE_JIT_DEEPGEMM')}")
print(f"[info] SGLANG_ENABLE_JIT_KVCACHE={os.environ.get('SGLANG_ENABLE_JIT_KVCACHE')}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; cannot run Miles/Megatron training on this node.")
PY

if [ "$fail" -ne 0 ]; then
   echo "[fail] SDPO training preflight failed."
   exit 1
fi

ok "SDPO training preflight passed."
