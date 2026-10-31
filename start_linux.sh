#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Startup failed. Run install_linux.sh and review the error above." >&2' ERR
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/conda_linux.sh"
if ! environment_exists; then
    echo "Environment missing. Run $ROOT/install_linux.sh first." >&2
    exit 1
fi
conda activate otoslotto-ai
python -m src.cli verify
echo "Open http://127.0.0.1:8501"
exec python -m streamlit run "$ROOT/app.py" --server.address 127.0.0.1 --server.port 8501
