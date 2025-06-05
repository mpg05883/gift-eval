#!/bin/bash
#SBATCH --job-name=eval_taskbench
#SBATCH --array=0-2
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x/%A/out/task_%a/main.out
#SBATCH --error=logs/%x/%A/err/task_%a/main.err
#SBATCH --nodelist=glamor-ruby
#SBATCH --gres=gpu:4

source ./scripts/utils.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

job_name="${SLURM_JOB_NAME}"
job_id="${SLURM_ARRAY_JOB_ID}"
task_id="task_${SLURM_ARRAY_TASK_ID}"

out_dir="logs/${job_name}/${job_id}/out/${task_id}"
err_dir="logs/${job_name}/${job_id}/err/${task_id}"

mkdir -p "$out_dir" "$err_dir"

PY_FILE_NAME="eval_taskbench.py"
model_names=("codellama/CodeLlama-13b-hf")
subsets=("dailylifeapis" "huggingface" "multimedia")

num_model_names=${#model_names[@]}
num_subsets=${#subsets[@]}

model_name_index=$(( SLURM_ARRAY_TASK_ID / num_subsets ))
subset_index=$(( SLURM_ARRAY_TASK_ID % num_subsets ))

model_name=${model_names[$model_name_index]}
subset=${subsets[$subset_index]}

log_info "Model: $model_name"
log_info "Subset: $subset"
log_job_info

export NCCL_P2P_DISABLE=1

# Spawn 4 processes, one per GPU
for gpu_id in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        python $PY_FILE_NAME \
            --model_name "$model_name" \
            --subset "$subset" \
            --shard_id $gpu_id \
            --num_shards 4 \
            --batch_size 16 \
    ) > "$out_dir/gpu_${gpu_id}.out" \
      2> "$err_dir/gpu_${gpu_id}.err" &
done

wait  # Wait for all GPU processes to finish

log_error "No errors for ${PY_FILE_NAME} $model_name ($subset)"

DONE_DIR="done/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$DONE_DIR"

sanitized_model_name="${model_name//\//_}"
DONE_NAME="${sanitized_model_name}_${subset}"
DONE_FILE="${DONE_DIR}/${DONE_NAME}.done"
touch "$DONE_FILE"

END_TIME=$(TZ="America/Los_Angeles" date +"%b-%d-%Y_%I-%M%p")
echo "[${END_TIME}] Finished ${PY_FILE_NAME} for $model_name ($subset)!" >"$DONE_FILE"
