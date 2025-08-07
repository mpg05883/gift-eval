#!/bin/bash

# Returns the current PST timestamp in month day year, hour:minute:second AM/PM
# format.
get_timestamp() {
    TZ="America/Los_Angeles" date +"%b %d, %Y %I:%M:%S%p"
}

# Prints timestamped info messages to stdout
log_info() {
    timestamp=$(get_timestamp)
    echo "[${timestamp}] $*"
}

# Prints timestamped error messages to stderr
log_error() {
    timestamp=$(get_timestamp)
    echo "[${timestamp}] $*" >&2
}

# Deletes Lightning logs from previous runs
delete_lightning_logs() {
    if [ -d "lightning_logs" ]; then
        rm -rf lightning_logs
    fi
}

activate_tempo_env() {
    source /sw/external/python/anaconda3/etc/profile.d/conda.sh
    conda activate tempo
}

# Logs the following slurm job information:
# - `SLURM_JOB_NAME`: The job's name.
# - `SLURM_JOB_ID`: The unique job ID assigned by SLURM if the job is part of an
#   array. E.g. 123456_0 for task 0, 123456_1 for task 1, etc.
# - `SLURM_ARRAY_JOB_ID`: The array job's ID if the job's part of an array. Same
#   for all tasks in the array. E.g. 123456.
# - `SLURM_ARRAY_TASK_ID`: The specific task index within the array job (if
#   applicable). E.g. 0 for task 0, 1 for task 1, etc.
# - `SLURM_GPUS_ON_NODE`: Number of devices (e.g. GPUs) used per each node.
# - `SLURM_JOB_NUM_NODES`: Number of nodes (i.e. machines) used for the job.
log_job_info() {
    # Check if SLURM variables are set. Otherwise, use "N/A".
    {
        echo -e "Field\t|\tValue"
        echo -e "-----\t|\t-----"
        echo -e "Job name\t|\t${SLURM_JOB_NAME:-N/A}"
        echo -e "Job ID\t|\t${SLURM_JOB_ID:-N/A}"
        echo -e "Array job ID\t|\t${SLURM_ARRAY_JOB_ID:-N/A}"
        echo -e "Array task ID\t|\t${SLURM_ARRAY_TASK_ID:-N/A}"
        echo -e "Devices per node\t|\t${SLURM_GPUS_ON_NODE:-N/A}"
        echo -e "Number of nodes\t|\t${SLURM_JOB_NUM_NODES:-N/A}"
    } | column -t -s $'\t'
}

# Returns a "done" directory path for marking job completion and ensures the
# directory exists using `mkdir -p`.
#
# The directory path depends on whether the job is part of a SLURM array:
# - If part of an array, the path is:
#     ./ouput/logs/<SLURM_JOB_NAME>/done/<SLURM_ARRAY_JOB_ID>
# - Else, the path is:
#     ./ouput/logs/<SLURM_JOB_NAME>/done
#
# - `SLURM_JOB_NAME`: The name of the SLURM job.
# - `SLURM_ARRAY_JOB_ID`: The array job ID, if applicable.
get_done_dir() {
    local base_dir="./output/logs/${SLURM_JOB_NAME}/done"

    if [[ -n "$SLURM_ARRAY_JOB_ID" ]]; then
        done_dir="${base_dir}/${SLURM_ARRAY_JOB_ID}"
    else
        done_dir="$base_dir"
    fi

    mkdir -p "$done_dir"
    echo "$done_dir"
}

# Returns a file path to a "done" file to mark completion of a SLURM job or
# array task and ensures the file exists using `touch`.
#
# The file name depends on whether the job is part of a SLURM array:
# - If part of an array, the file is named:
#     <SLURM_ARRAY_TASK_ID>.done
# - Else, the file is named:
#     <SLURM_JOB_ID>.done
#
# The file is placed inside the "done" directory created by `create_done_dir`.
#
# - `SLURM_JOB_ID`: The unique SLURM job ID.
# - `SLURM_ARRAY_JOB_ID`: The ID shared across all array tasks, if applicable.
# - `SLURM_ARRAY_TASK_ID`: The index of the array task, if applicable.
# - `SLURM_JOB_NAME`: The name of the SLURM job (used to determine directory
# path).
get_done_file() {
    local done_dir
    done_dir=$(get_done_dir)

    local done_file
    if [[ -n "$SLURM_ARRAY_JOB_ID" ]]; then
        done_file="${SLURM_ARRAY_TASK_ID}.done"
    else
        done_file="${SLURM_JOB_ID}.done"
    fi

    local done_path="${done_dir}/${done_file}"
    touch "$done_path"
    echo "$done_path"
}

# Sets common W&B environment variables that'll remain the same over multiple
# runs like:
# - `WANDB_CACHE_DIR`: Directory for W&B cache.
# - `WANDB_DATA_DIR`: Directory for W&B data.
# - `WANDB_ARTIFACTS_DIR`: Directory where W&B artifacts are downloaded.
# - `WANDB_JOB_NAME`: Name of the SLURM job.
# - `WANDB_JOB_TYPE`: Type of the SLURM job.
# - `WANDB_SHOW_RUN`: Whether to show the run in W&B UI.
# - `WANDB_TAGS`: Tags for the W&B run, including SLURM job ID and name.
set_common_wandb_env_vars() {
    CACHE_DIR=./output/wandb/cache
    CONFIG_DIR=./output/wandb/config
    DATA_DIR=./output/wandb/data
    ARTIFACTS_DIR=./downloads/wandb
    WANDB_DIR=./output/wandb
    mkdir -p "$WANDB_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$ARTIFACTS_DIR"

    export WANDB_CACHE_DIR="$CACHE_DIR"
    export WANDB_CONFIG_DIR="$CONFIG_DIR"
    export WANDB_DATA_DIR="$DATA_DIR"
    export WANDB_ARTIFACTS_DIR="$ARTIFACTS_DIR"
    export WANDB_DIR="$WANDB_DIR"
    export WANDB_ENTITY="mpgee-usc"
    export WANDB_JOB_NAME="$SLURM_JOB_NAME"
    export WANDB_JOB_TYPE="$SLURM_JOB_NAME"
    export WANDB_PROJECT="TEMPO"
    export WANDB_SHOW_RUN=true
    export WANDB_TAGS="slurm_job_id=${SLURM_JOB_ID},slurm_job_name=${SLURM_JOB_NAME}"
}
