#!/usr/bin/env bash
# Create the NAS-side venv (.venv-nas/) and install GPU torch + ofa.
# Kept separate from setup_laptop.sh so the LUT pipeline's CPU-only
# torch is never accidentally upgraded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# CUDA wheel index. Override with TORCH_CUDA_INDEX=cu121 (or cu118 etc.)
# if a different CUDA toolkit is installed.
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu128}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}"

if [[ ! -d .venv-nas ]]; then
  python3 -m venv .venv-nas
fi

# shellcheck disable=SC1091
source .venv-nas/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install --extra-index-url "$TORCH_INDEX_URL" -r requirements-nas.txt

# CP 1.1 DoD — confirm `import ofa` works.
python - <<'PY'
import ofa, torch, numpy, yaml
print("nas env ok:",
      "torch", torch.__version__,
      "cuda", torch.cuda.is_available(),
      "ofa", ofa.__file__)
PY

echo
echo "NAS env ready. Activate with: source $ROOT/.venv-nas/bin/activate"
