#!/bin/bash
#
# Utility functions for bash scripts.

# Activate a conda environment named `env_name` (default: "gift_eval").
activate_conda_env() {
    env_name="gift_eval"
    source /sw/external/python/anaconda3/etc/profile.d/conda.sh
    conda activate "$env_name"
}