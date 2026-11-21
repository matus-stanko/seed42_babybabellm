#!/bin/bash
#SBATCH --job-name=dan-baseline-small
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pier@itu.dk

set -e
export PATH="/home/pier/.conda/envs/babylm-train/bin:$PATH"
echo "python: $(which python)"
export HF_HUB_OFFLINE=1

hostname
nvidia-smi
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"

python train.py --dataset BabyLM-community/babylm-dan --config ./small_config.json \
  --vocab_size 8192 --epochs ${EPOCHS:-10} --batch_size 8 \
  --output_dir ./${OUTDIR:-dan-baseline-small} --model_name my-dan-baseline-small ${REVISION:+--revision "$REVISION"}
