#!/bin/bash
#
# Utility functions for bash scripts

# Activate a conda environment named `env` (default: "gift_eval")
activate_conda_env() {
    env="gift_eval"

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

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env"
}