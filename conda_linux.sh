#!/usr/bin/env bash
# Shared setup, sourced by the Linux entry points.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if ! command -v conda >/dev/null 2>&1; then
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
            source "$candidate/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found. Install Miniconda: https://docs.conda.io/projects/miniconda/"
    echo "Restart your terminal after running the installer and conda init bash."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

environment_exists() {
    conda env list | awk '$1 == "otoslotto-ai" {found=1} END {exit !found}'
}

update_environment() {
    if environment_exists; then
        conda env update -n otoslotto-ai -f "$ROOT/environment.yml" --prune
    else
        conda env create -f "$ROOT/environment.yml" -y
    fi
    conda activate otoslotto-ai
    python -m src.cli verify
    python -m src.cli init
}
