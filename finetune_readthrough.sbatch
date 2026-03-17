#!/bin/bash
#SBATCH -A iisc
#SBATCH -p gpusinglenode          # <-- change if needed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -J mrna_finetune
#SBATCH -o /scratch/bioadi/mrna_lm_readthrough/logs/%x_%j.out
#SBATCH -e /scratch/bioadi/mrna_lm_readthrough/logs/%x_%j.err

set -euo pipefail

# Always run from scratch
cd /scratch/bioadi/mrna_lm_readthrough

# --- Environment setup ---
# Option A: if you use conda
source ~/.bashrc
conda activate mrna-lm

# Option B: if you use modules (uncomment if your cluster uses them)
# module purge
# module load cuda/<version>
# module load anaconda/<version>
# source activate mrna-lm

# Helpful diagnostics
nvidia-smi
python -V
which python

# --- Run ---
python finetune_all.py \
  --task readthrough \
  --output /scratch/bioadi/mrna_lm_readthrough/outputs/test_output_new
