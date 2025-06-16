#!/bin/bash

#SBATCH --job-name=download_datasets
#SBATCH --array=0-1
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x/out/%A/%a.out
#SBATCH --error=logs/%x/err/%A/%a.err
#SBATCH --account=jessetho_1390 
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs
source ./scripts/utils.sh

source ~/.bashrc
conda activate gift

split=""
name="GiftEval"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    split="pretrain"
    name+="Pretrain"
else
    split="train_test"
fi


dirpath="./datasets/${split}"
mkdir -p "$dirpath"

# Count the number of datasets that've been downloaded
num_datasets=$(find "$dirpath" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && [ "$num_datasets" -eq 50 ]; then
    log_info "${split} datasets have already been downloaded! Exiting..."
    exit 0
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ] && [ "$num_datasets" -eq 28 ]; then
    log_info "${split} datasets have already been downloaded! Exiting..."
    exit 0
fi

log_job_info
log_info "Downloading ${split}"

if huggingface-cli download Salesforce/"${name}" \
        --repo-type=dataset \
        --local-dir "${dirpath}";
then
    log_info "Successfully downloaded ${split} split to ${dirpath}!"
    log_error "No errors for downloading ${split} split!"

    done_dir="logs/${SLURM_JOB_NAME}/done/${SLURM_ARRAY_JOB_ID}"
    mkdir -p "$done_dir"

    done_file="${done_dir}/${SLURM_ARRAY_TASK_ID}.done"
    touch "$done_file"

    end_time=$(get_timestamp)
    echo "[${end_time}] Done downloading ${split} split" >"$done_file"
else
    log_error "ERROR: Failed to download ${split} split!"
fi 

