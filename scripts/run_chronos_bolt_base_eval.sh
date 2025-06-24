#!/bin/bash

#SBATCH --job-name=chronos_bolt_base_eval
#SBATCH --array=100-199  # * Manually set this to 0-99, 100-199, ..., 400-455,
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%x/out/%A.out
#SBATCH --error=logs/%x/err/%A.err
#SBATCH --account=jessetho_1390
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs
source ./scripts/utils.sh

source ~/.bashrc
conda activate gift

log_job_info

index="${SLURM_ARRAY_TASK_ID}"

if python chronos_bolt_eval.py --index="${index}"; then
    log_info "Successfully finished ${SLURM_JOB_NAME}!"
    log_error "No errors!"

    done_file=$(get_done_file)
    end_time=$(get_timestamp)
    echo "[${end_time}] Done with ${SLURM_JOB_NAME}!" >"$done_file"
else
    log_error "ERROR: ${SLURM_JOB_NAME}!"
    exit 1
fi
