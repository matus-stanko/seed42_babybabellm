#!/bin/bash
#SBATCH --job-name=eng-baseline-small
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pier@itu.dk

set -e
export PATH="/home/pier/.conda/envs/babylm-train/bin:$PATH"
echo "python: $(which python)"
export HF_HUB_OFFLINE=1

hostname
nvidia-smi
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"

python train.py --dataset BabyLM-community/babylm-eng --config ./small_config.json \
  --vocab_size 8192 --epochs ${EPOCHS:-5} --batch_size 64 \
  --output_dir ./${OUTDIR:-eng-baseline-small} --model_name my-eng-baseline-small ${REVISION:+--revision "$REVISION"}
