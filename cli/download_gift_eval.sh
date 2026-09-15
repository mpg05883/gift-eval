#!/bin/bash
#
# Download the GIFT-Eval dataset from Hugging Face to a local directory named
# `local_dir`.
# If the download fails,  it will retry up to `max_attempts` times
# with a delay of `delay_seconds` seconds between attempts.
#
# The dataset is available at: https://huggingface.co/datasets/Salesforce/GiftEval

set -e

dataset_name="GiftEval"
local_dir="./data/$dataset_name"
max_attempts=5
delay_seconds=10

attempt=1
until hf download "Salesforce/$dataset_name" \
    --repo-type=dataset \
    --local-dir "$local_dir"; do
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "hf download failed after $max_attempts attempts" >&2
        exit 1
    fi
    echo "hf download failed (attempt $attempt/$max_attempts), retrying in $delay_seconds seconds..." >&2
    attempt=$((attempt + 1))
    sleep "$delay_seconds"
done

size_kb=$(du -sk "$local_dir" | cut -f1)
size_gb=$(awk "BEGIN {printf \"%.2f\", ($size_kb * 1024) / 1000000000}")

echo -e "\nDownloaded $dataset_name to $local_dir"
echo "Size: $size_gb GB"