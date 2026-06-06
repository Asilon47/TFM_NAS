#!/usr/bin/env bash
# Create laptop venv and install CPU-only dependencies used to generate ONNX blocks
# and orchestrate the Jetson. PyTorch is CPU-only on purpose — no CUDA on the laptop.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel
# CPU wheels for torch; other packages resolve from PyPI
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

python - <<'PY'
import torch, onnx, onnxruntime, fabric, numpy, pandas, tqdm, yaml
print("laptop env ok:",
      "torch", torch.__version__,
      "onnx", onnx.__version__,
      "ort", onnxruntime.__version__,
      "fabric", fabric.__version__)
PY

echo
echo "Laptop env ready. Activate with: source $ROOT/.venv/bin/activate"
