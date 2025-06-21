#!/bin/bash

#SBATCH --job-name=train_test_dataset_analysis
#SBATCH --array=0-96
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
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

split="train_test"

index=$SLURM_ARRAY_TASK_ID

export RAY_TMPDIR=/scratch1/mpgee/ray_tmpdir
mkdir -p "$RAY_TMPDIR"

NUM_CPUS=$(nproc)
log_info "Number of CPUs: $NUM_CPUS"

cat <<EOF > .env
NUM_CPUS=$NUM_CPUS
HYDRA_FULL_ERROR=1
GIFT_EVAL=./datasets
EOF

export RAY_DEDUP_LOGS=0
log_job_info

if python -m cli.analysis datasets="$split" index=$index; then
    log_info "Successfully completed dataset analysis!"
    log_error "No errors!"

    done_file=$(get_done_file)
    end_time=$(get_timestamp)
    echo "[${end_time}] Done with dataset analysis!" >"$done_file"
else
    log_error "ERROR: Dataset analysis failed!"
    exit 1
fi