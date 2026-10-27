#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL=0
[[ "${1:-}" == "--install" ]] && INSTALL=1

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$INSTALL" == "1" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$ROOT/.venv-server"
  PYTHON_BIN="$ROOT/.venv-server/bin/python"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r "$ROOT/requirements-server.txt"
fi

echo "== host =="
hostname
uname -a
command -v nvidia-smi >/dev/null && nvidia-smi || echo "WARNING: nvidia-smi not found"

echo "== python =="
"$PYTHON_BIN" - <<'PY'
import importlib
import platform
print(platform.python_version())
for name in ("numpy", "scipy", "PIL", "torch", "cv2"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "ok"))
import torch
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

echo "== bundle =="
"$PYTHON_BIN" "$ROOT/deploy/verify_bundle.py" --root "$ROOT"

echo "== VIF self-test =="
mkdir -p "$ROOT/results"
"$PYTHON_BIN" "$ROOT/analysis/vif_reference_rescore.py" --selftest \
  --vis "$ROOT/data/tno_trial/vis" --ir "$ROOT/data/tno_trial/ir" \
  --fused "mean=$ROOT/data/tno_trial/vis" --limit 1 \
  --out "$ROOT/results/vif_selftest"

echo "BOOTSTRAP_OK root=$ROOT"
