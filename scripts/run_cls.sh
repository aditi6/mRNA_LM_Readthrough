#!/bin/bash
set -e
source /opt/miniconda/etc/profile.d/conda.sh
conda activate readthrough
cd /workspace/mRNA_LM_Readthrough

git pull origin main

# Launch watcher BEFORE training starts — it waits for the log file to appear
nohup /opt/miniconda/envs/readthrough/bin/python /workspace/watch_cls.py \
    > /workspace/watcher_cls.log 2>&1 &
echo "Watcher PID: $!"

# Remove old log so watcher starts fresh
rm -f /workspace/train_cls.log

mkdir -p /workspace/runs/run_cls2
python finetune_cls.py \
    --task readthrough --output /workspace/runs/run_cls2 \
    --lorar 16 --lalpha 32 --ldropout 0.1 --lr 5e-5 \
    --batch 16 --head_dim 768 --head_dropout 0.1 \
    --epochs 25 --device 0 \
    2>&1 | tee /workspace/train_cls.log
echo "=== run_cls2 done at $(date) ==="
