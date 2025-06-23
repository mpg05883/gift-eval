#!/bin/bash

#SBATCH --job-name=chronos_bolt_eval
#SBATCH --array=0-455

# TODO: Configure job resources
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
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

if python chronos_bolt_eval.py --index="${SLURM_ARRAY_TASK_ID}"; then
    log_info "Successfully finished ${SLURM_JOB_NAME}!"
    log_error "No errors!"

    done_file=$(get_done_file)
    end_time=$(get_timestamp)
    echo "[${end_time}] Done with ${SLURM_JOB_NAME}!" >"$done_file"
else
    log_error "ERROR: ${SLURM_JOB_NAME}!"
    exit 1
fi
