#!/bin/bash
# run_window.sh — Launch windowed readthrough fine-tuning experiment.
# Uses finetune_window.py (dataload_window: last 20 CDS codons + first 200nt 3'UTR)
# Hyperparameters: r=8, alpha=16, dropout=0.1, lr=1e-4, batch=64, head_dropout=0.1
# 10 epochs as an initial test to compare against non-windowed baseline (AUROC ~0.58)

set -e

REPO=/workspace/mRNA_LM_Readthrough
OUTPUT=/workspace/runs/run_window
LOG=/workspace/train_window.log
RESULTS_FILE=$REPO/results/run_window_summary.txt

echo "=== Windowed Run Started: $(date) ===" | tee $LOG

source /opt/miniconda/etc/profile.d/conda.sh
conda activate readthrough

cd $REPO

# Pull latest code (includes finetune_window.py and dataload_window.py)
git pull origin main 2>&1 | tee -a $LOG

mkdir -p $OUTPUT

echo "Config: r=8, alpha=16, dropout=0.1, lr=1e-4, batch=64, head_dropout=0.1, epochs=10" | tee -a $LOG
echo "Window: last 20 CDS codons + first 200nt 3UTR" | tee -a $LOG

python finetune_window.py     --task readthrough     --output $OUTPUT     --lorar 8     --lalpha 16     --ldropout 0.1     --lr 1e-4     --batch 64     --head_dim 768     --head_dropout 0.1     --device 0     2>&1 | tee -a $LOG

# Extract final metrics and save summary
echo "=== Window Run Completed: $(date) ===" | tee -a $LOG
echo "Results summary:" | tee -a $LOG
grep -E "eval_auroc|test_auroc|train_loss" $LOG | tail -30 | tee -a $LOG

# Write results summary file
echo "# Windowed Run Results" > $RESULTS_FILE
echo "Window: last 20 CDS codons (60nt) + first 200nt 3UTR" >> $RESULTS_FILE
echo "Config: r=8, alpha=16, dropout=0.1, lr=1e-4, batch=64, head_dropout=0.1" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
python3 - <<'PYEOF' >> $RESULTS_FILE
import re, sys

log_path = '/workspace/train_window.log'
with open(log_path) as f:
    content = f.read()

# Parse epoch metrics
epoch_pattern = re.compile(r"'eval_loss': ([0-9.]+).*?'eval_f1': ([0-9.]+).*?'eval_auroc': ([0-9.]+).*?'epoch': ([0-9.]+)")
matches = epoch_pattern.findall(content)

print('epoch | eval_loss | eval_f1 | eval_auroc')
print('------|-----------|---------|----------')
for m in matches:
    print(f'{m[3]:>5} | {m[0]:>9} | {m[1]:>7} | {m[2]:>10}')

# Test metrics
test_m = re.search(r"'test_loss': ([0-9.]+).*?'test_f1': ([0-9.]+).*?'test_auroc': ([0-9.]+)", content)
if test_m:
    print(f'\nTest: loss={test_m.group(1)}, f1={test_m.group(2)}, auroc={test_m.group(3)}')
PYEOF

# Commit and push results
cd $REPO
git add results/run_window_summary.txt
git commit -m "Add windowed run results (window: last20CDS + first200UTR3)"
git push origin main

echo "=== Done: $(date) ==="
