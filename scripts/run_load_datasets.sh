#!/bin/bash

#SBATCH --job-name=load_datasets
#SBATCH --array=0-458
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%x/out/%A/%a.out
#SBATCH --error=logs/%x/err/%A/%a.err
#SBATCH --account=jessetho_1390
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs
source ./scripts/utils.sh

source ~/.bashrc
conda activate gift

dataset_dir="/project/jessetho_1390/gift_eval/pretrain"
mapfile -t names < <(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

num_names=${#names[@]}
log_info "Number of dataset names: $num_names"

term_index=$((SLURM_ARRAY_TASK_ID / num_names))
name_index=$((SLURM_ARRAY_TASK_ID % num_names))

terms=("short" "medium" "long")
term=${terms[$term_index]}
name=${names[$name_index]}

log_job_info
message="pretraining dataset $((SLURM_ARRAY_TASK_ID + 1)): $name (${term})"
log_info "Loading $message"

if python load_dataset.py --name "$name" --term "$term"; then
    log_info "Successfully loaded $message!"
    log_error "No errors!"

    done_file=$(get_done_file)
    end_time=$(get_timestamp)
    echo "[${end_time}] Finished loading $message!" >"$done_file"
else
    log_error "ERROR: Failed to load $message!"
    exit 1
fi
