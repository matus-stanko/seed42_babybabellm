#!/bin/bash
#SBATCH --job-name=eval-qwen3-dan
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matus@itu.dk

set -e

module load Miniconda3/25.5.1-1
eval "$(conda shell.bash hook)"
conda activate qwen-train

cd /home/matus/seminars-in-data-science/multilingual-evaluation
MODEL=/home/matus/seminars-in-data-science/multilingual-training/a100-qwen3-dan

hostname
nvidia-smi

lm_eval --model hf \
  --model_args pretrained=$MODEL,dtype=auto \
  --tasks babybabellm_dan --batch_size auto \
  --output_path ./eval-results/qwen3-dan --log_samples \
  --include_path ./tasks --num_fewshot 0