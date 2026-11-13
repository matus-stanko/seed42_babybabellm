#!/bin/bash
#SBATCH --job-name=a100-qwen3-eng-80gb
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=matus@itu.dk

set -e

module load Miniconda3/25.5.1-1
eval "$(conda shell.bash hook)"
conda activate qwen-train
export HF_HUB_OFFLINE=1

hostname
nvidia-smi
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"

python train.py --dataset BabyLM-community/babylm-eng --config ./qwen3_config.json \
  --vocab_size 8192 --epochs 5 --batch_size 64 \
  --output_dir ./a100-qwen3-eng-80gb --model_name a100-qwen3-eng-80gb