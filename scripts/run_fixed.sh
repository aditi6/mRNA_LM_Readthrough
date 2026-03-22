#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate readthrough
cd /workspace/mRNA_LM_Readthrough

git checkout finetune_all.py
sed -i 's/num_train_epochs=100/num_train_epochs=15/' finetune_all.py
sed -i 's/#     metric_for_best_model=metric_for_best_model,/    metric_for_best_model=metric_for_best_model,/' finetune_all.py
sed -i 's/#     greater_is_better=greater_is_better,/    greater_is_better=greater_is_better,/' finetune_all.py
sed -i '/warmup_ratio=0.1,/a\    weight_decay=0.01,' finetune_all.py

rm -rf /workspace/outputs/run_fixed
mkdir -p /workspace/outputs/run_fixed /workspace/logs

python finetune_all.py \
    --task readthrough \
    --output /workspace/outputs/run_fixed \
    --lorar 8 --lalpha 16 --ldropout 0.1 \
    --lr 1e-4 \
    --batch 16 \
    --head_dropout 0.1 \
    > /workspace/logs/run_fixed.log 2>&1

grep -E "eval_auroc|eval_f1|eval_loss|train_loss" /workspace/logs/run_fixed.log \
    | grep -v "Map\|it/s" > results/run_fixed_summary.txt
cp /workspace/logs/run_fixed.log results/run_fixed.log
git add results/
git diff --cached --quiet || git commit -m "Results: run_fixed r8 wd0.01 bs16 drop0.1"
git push || true
echo "Done: $(date)"
