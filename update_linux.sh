#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Update failed. Existing valid historical data is preserved." >&2' ERR
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/conda_linux.sh"
update_environment
python -m src.cli sync
echo "Environment and historical database updated."
