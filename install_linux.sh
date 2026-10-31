#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Installation failed at line $LINENO. Review the error above." >&2' ERR
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/conda_linux.sh"
update_environment
echo "Installation complete. Run $ROOT/start_linux.sh"
