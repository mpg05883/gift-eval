#!/bin/bash
#
# Download the GIFT-Eval Pretrain dataset from Hugging Face to a local directory
# named `local_dir`.
# If the download fails,  it will retry up to `max_attempts` times with a delay of
# `delay_seconds` seconds between attempts.
#
# The dataset is available at: https://huggingface.co/datasets/Salesforce/GiftEvalPretrain

#SBATCH --job-name=download-gift-eval-pretrain
#SBATCH --partition=day
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/%x/out/%A.out
#SBATCH --error=outputs/logs/%x/err/%A.err
#SBATCH --mail-user=mike.gee@yale.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

dataset="GiftEvalPretrain"
local_dir="../data/gift_eval_pretrain"

# Retry parameters
max_attempts=100
min_delay_seconds=10
max_delay_seconds=600

# Initial attempt and delay
attempt=1
delay_seconds=$min_delay_seconds

# Download the dataset
until hf download "Salesforce/$dataset" \
    --repo-type=dataset \
    --local-dir "$local_dir"; do

    # Exit early if the download fails after `max_attempts` attempts
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "hf download failed after $max_attempts attempts" >&2
        exit 1
    fi

    # Exponential backoff
    echo "hf download failed (attempt $attempt/$max_attempts), retrying in ${delay_seconds}s..." >&2
    attempt=$((attempt + 1))
    sleep "$delay_seconds"
    delay_seconds=$((delay_seconds * 2))
    if [ "$delay_seconds" -gt "$max_delay_seconds" ]; then
        delay_seconds=$max_delay_seconds
    fi
done

size_kb=$(du -sk "$local_dir" | cut -f1)
size_gb=$(awk "BEGIN {printf \"%.2f\", ($size_kb * 1024) / 1000000000}")

echo -e "\nDownloaded $dataset to $local_dir"
echo "Size: $size_gb GB"
