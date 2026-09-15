#!/bin/bash
#
# Create a conda environment named "gift_eval" with Python 3.11.11 and install
# dependencies in editable mode. 

set -e

env_name="gift_eval"
py_version="3.11.11"

# Look for conda.sh in common locations and load it if found
if ! command -v conda >/dev/null 2>&1; then
    conda_sh_locations=(
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/miniforge3/etc/profile.d/conda.sh"
    )
    for conda_sh in "${conda_sh_locations[@]}"; do
    if [ -f "$conda_sh" ]; then
        # shellcheck disable=SC1090
        source "$conda_sh"
        break
    fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not found; install Anaconda or add it to PATH" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n "$env_name" python="$py_version"
conda activate "$env_name"

pip install -e .
