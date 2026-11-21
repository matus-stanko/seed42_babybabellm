# Seed_42: Reproducing BabyBabelLM (Challenge 3)
 
Code and results for our reproduction of the English and Danish zero-shot evaluation (Table 6) of
[BabyBabelLM](https://arxiv.org/abs/2510.10159), plus a bonus with Qwen3-0.6B.
 
This repository contains **only the files that are not in the authors' original repositories**:
our SLURM job scripts, our modified `train.py`, environment files, training logs and evaluation results.
The training and evaluation code itself comes from the authors:
 
- Training: [babylm-org/multilingual-training](https://github.com/babylm-org/multilingual-training)
- Evaluation: [babylm-org/multilingual-evaluation](https://github.com/babylm-org/multilingual-evaluation)
All jobs were run on the ITU HPC cluster with SLURM. See the report for hyperparameters, dataset versions and hardware.
 
## Structure
 
```
gpt2_evaluation/   evaluation of the released HF models (lm-eval job script, environment, outputs)
gpt2_training/     our GPT-2 training: modified train.py, job scripts, frozen requirements, logs
qwen3/training/    Qwen3-0.6B training job scripts and logs (Danish 10 ep, English 5 and 10 ep)
qwen3/evaluation/  Qwen3 lm-eval job scripts, logs and result JSON files
```
 