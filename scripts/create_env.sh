#!/bin/bash
set -e

ENV_NAME="gifteval"
PYTHON_VERSION="3.11.11"

# Non-interactive shells don't load conda from .bash_profile
if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh"; do
    if [ -f "$conda_sh" ]; then
        # shellcheck disable=SC1090
        source "$conda_sh"
        break
    fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not found; install Anaconda/Miniconda or add it to PATH" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
conda activate "$ENV_NAME"

pip install -e ".[baseline]"
