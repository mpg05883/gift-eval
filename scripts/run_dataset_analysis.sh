#!/bin/bash

#SBATCH --job-name=dataset_analysis
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
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

# TODO: Remove this once you finish the analysis
# Clean old outputs
rm -rf ./outputs/analysis/test

# Set Ray's temporary directory
export RAY_TMPDIR=/scratch1/mpgee/ray_tmpdir
mkdir -p "$RAY_TMPDIR"

# Set the number of CPUs to use
NUM_CPUS=$(nproc)

# Overwrite the .env file with all environment variables at once
cat <<EOF > .env
NUM_CPUS=$NUM_CPUS
HYDRA_FULL_ERROR=1
GIFT_EVAL=./datasets
RAY_DEDUP_LOGS=0
EOF

log_job_info
if python -m cli.analysis datasets=all_datasets; then
    log_info "Successfully completed dataset analysis!"
    log_error "No errors!"

    done_dir="logs/${SLURM_JOB_NAME}/done"
    mkdir -p "$done_dir"

    done_file="${done_dir}/${SLURM_JOB_ID}.done"
    touch "$done_file"

    end_time=$(get_timestamp)
    echo "[${end_time}] Done with dataset analysis!" >"$done_file"
else
    log_error "ERROR: Dataset analysis failed!"
    exit 1
fi