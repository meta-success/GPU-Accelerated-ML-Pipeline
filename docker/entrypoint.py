"""Container entrypoint (Python, so Windows CRLF checkouts still run in Linux images)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    print("=== GPU-Accelerated ML Pipeline container ===")
    print("Python", sys.version.replace("\n", " "))
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found — GPU may be unavailable in this container.")

    checker = Path("/app/scripts/check_gpu.py")
    if checker.exists():
        subprocess.run([sys.executable, str(checker)], check=False)

    os.execvp(sys.executable, [sys.executable, "-m", "src", *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
