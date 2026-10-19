#!/usr/bin/env bash
# Install the minimal CUDA training environment used by OPTD.
#
# Prerequisites:
#   - Linux, an NVIDIA driver compatible with CUDA 12 wheels, and Conda/Mamba.
#   - A fresh Python 3.12 environment (recommended command is in README.md).
#
# This deliberately does not install SGLang, Megatron, or FlashAttention by
# default: OPTD's documented FSDP + vLLM path does not need them, and each can
# introduce an avoidable CUDA/toolchain compatibility failure.  The launchers
# support PyTorch SDPA.  Set INSTALL_FLASH_ATTN=1 to opt into FlashAttention.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: cannot find Python executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"ERROR: OPTD's tested vLLM environment requires Python 3.12; got "
        f"{sys.version.split()[0]}. Create it with: conda create -n optd python=3.12"
    )
PY

echo "==> Upgrading packaging tools"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "==> Installing vLLM ${VLLM_VERSION} and its compatible PyTorch CUDA dependencies"
"$PYTHON_BIN" -m pip install "vllm==${VLLM_VERSION}"

echo "==> Installing OPTD's vendored verl package and math evaluation dependency"
"$PYTHON_BIN" -m pip install -e "${ROOT_DIR}/verl"
"$PYTHON_BIN" -m pip install math-verify

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  echo "==> Installing optional FlashAttention (requires a matching CUDA toolkit/compiler)"
  "$PYTHON_BIN" -m pip install --no-build-isolation flash-attn
else
  echo "==> Skipping optional FlashAttention; use ATTN_IMPLEMENTATION=sdpa (the default documented path)."
fi

echo "==> Verifying imports"
"$PYTHON_BIN" - <<'PY'
import ray
import torch
import transformers
import verl
import vllm

print(f"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
print(f"vllm={vllm.__version__}, ray={ray.__version__}, transformers={transformers.__version__}")
print(f"verl={verl.__file__}")
if not torch.cuda.is_available():
    print("WARNING: CUDA is not visible. Installation completed, but GPU training will not run until a compatible NVIDIA driver/GPU is available.")
PY

echo "OPTD environment is ready."
