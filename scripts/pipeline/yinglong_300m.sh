#!/bin/bash

# Delta partition information:
# https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html#delta-partitions-queues

#SBATCH --job-name=yinglong_eval
#SBATCH --partition=gpuA40x4  
#SBATCH --array=0-96
#SBATCH --mem=100GB  # Total memory per node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1  # * Ensure --gpus-per-node equals --ntasks-per-node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest
#SBATCH --account=bcqc-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p ./output/logs

# Load helper functions
source ./scripts/utils/utils.sh

log_info "Starting ${SLURM_JOB_NAME}. Job ID: ${SLURM_JOB_ID}"

# Activate conda environment
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate gift

# Disable progress bars when loading Hugging Face datasets
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

WANDB_NOTES="Rough draft"  
set_common_wandb_env_vars
export WANDB_NOTES="$WANDB_NOTES"
export WANDB_RUN_GROUP="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# Use SLURM_ARRAY_TASK_ID to set task_index
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    task_index=0  
else
    task_index=$SLURM_ARRAY_TASK_ID
fi

if python yinglong_300m_eval.py; then
    log_info "Successfully finished ${SLURM_JOB_NAME}!"
    log_error "No errors!"
    
    done_file=$(get_done_file)
    end_time=$(get_timestamp)
    echo "[${end_time}] Done with ${SLURM_JOB_NAME}!" >"$done_file"
else
    log_error "ERROR: ${SLURM_JOB_NAME} failed!"
    exit 1
fi
