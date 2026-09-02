#!/usr/bin/env bash
set -euo pipefail

echo "=== GPU-Accelerated ML Pipeline container ==="
python --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found — GPU may be unavailable in this container."
fi

python /app/scripts/check_gpu.py || true
exec python -m src "$@"
