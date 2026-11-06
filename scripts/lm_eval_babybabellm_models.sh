#!/bin/bash
#SBATCH --job-name=lm_eval_babybabellm_models
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2
#SBATCH --output=.slurm/lm_eval_babybabellm_models_%A_%a.out
#SBATCH --error=.slurm/lm_eval_babybabellm_models_%A_%a.err

set -euo pipefail

# Activate the Conda environment created from environment-hpc-hf.yml.
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV_NAME="babybabellm-eval"
echo "Using Conda environment: ${CONDA_ENV_NAME}"
conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -c "import lm_eval; print('lm-eval import succeeded')"

# Slurm executes a staged copy of this script in /var/spool/slurmd, so do not
# derive paths from BASH_SOURCE. SLURM_SUBMIT_DIR is the directory from which
# sbatch was invoked.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -d "${SUBMIT_DIR}/tasks" ]]; then
    REPO_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../tasks" ]]; then
    REPO_DIR="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "Could not locate repository tasks directory from: ${SUBMIT_DIR}" >&2
    exit 1
fi

num_fewshot=0

# Five evaluations, with at most two running at the same time:
#   0: Hugging Face Danish baseline     3: local English checkpoint
#   1: Hugging Face English baseline
#   2: local Danish checkpoint
declare -a lang_codes=(
    "dan" "eng" "dan" "eng"
)
declare -a model_sources=(
    "hf" "hf" "local" "local"
)
declare -a local_checkpoint_dirs=(
    "" "" "dan-rev05f1-a100" "eng10-revc5a3-a100"
)

# Select this array entry.
lang=${lang_codes[$SLURM_ARRAY_TASK_ID]}
model_source=${model_sources[$SLURM_ARRAY_TASK_ID]}
local_checkpoint_dir=${local_checkpoint_dirs[$SLURM_ARRAY_TASK_ID]}

task_name="babybabellm_${lang}"

if [[ "${model_source}" == "hf" ]]; then
    model_name="BabyLM-community/${lang}-baseline-small"
    # Retain the original result layout for the Hugging Face baselines.
    output_path="${REPO_DIR}/results"
    results_folder="${REPO_DIR}/results/BabyLM-community__${lang}-baseline-small"
else
    model_name="${REPO_DIR}/models/${local_checkpoint_dir}"
    # A separate parent directory prevents local-model results from mixing with
    # downloaded Hugging Face-model results.
    output_path="${REPO_DIR}/results/local-${local_checkpoint_dir}"
    results_folder="${output_path}"

    if [[ ! -f "${model_name}/config.json" ]]; then
        echo "Local checkpoint is missing: ${model_name}/config.json" >&2
        exit 1
    fi
fi

echo "Evaluating ${model_source} model for language: ${lang}"
echo "Model: ${model_name}"

# Check if results folder already exists
if [ -d "$results_folder" ]; then
    echo "Results folder already exists: $results_folder"
    echo "Skipping evaluation for ${lang}"
    exit 0
fi

echo "Results folder not found, starting evaluation..."

srun conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -m lm_eval \
    --model hf \
    --model_args "pretrained=${model_name}" \
    --tasks "${task_name}" \
    --device cuda \
    --output_path "${output_path}" \
    --batch_size auto:10 \
    --num_fewshot ${num_fewshot} \
    --log_samples \
    --include_path "${REPO_DIR}/tasks"

echo "Completed evaluation for ${lang}"
