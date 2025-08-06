#!/bin/bash

#SBATCH --job-name=load_datasets
#SBATCH --array=0-151
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

mapfile -t names < <(find datasets/pretrain -mindepth 1 -maxdepth 1 -type d ! -name '.*' -exec basename {} \;)
name="${names[$SLURM_ARRAY_TASK_ID]}"

log_job_info

if python load_dataset.py --name="${name}"; then
    log_info "Successfully finished ${SLURM_JOB_NAME}!"
    log_error "No errors!"

    done_file=$(get_done_file)
    end_time=$(get_timestamp)
    echo "[${end_time}] Done with ${SLURM_JOB_NAME}!" >"$done_file"
else
    log_error "ERROR: ${SLURM_JOB_NAME}!"
    exit 1
fi
