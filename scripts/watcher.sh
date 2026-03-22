#!/bin/bash
cd /workspace/mRNA_LM_Readthrough
LOGFILE=/workspace/logs/run_fixed.log
LAST_EPOCH=0

while true; do
    sleep 120
    if [ ! -f "$LOGFILE" ]; then continue; fi

    CURRENT_EPOCH=$(grep -oP "'epoch': \K[0-9.]+" "$LOGFILE" | grep "\.0$" | tail -1 | xargs printf "%.0f" 2>/dev/null || echo 0)
    if [ "$CURRENT_EPOCH" -gt "$LAST_EPOCH" ] 2>/dev/null; then
        LAST_EPOCH=$CURRENT_EPOCH
        {
            echo "# Training Progress (auto-updated)"
            echo "Last update: $(date)"
            echo ""
            echo "## Config: lr=1e-4, r=8, alpha=16, batch=16, dropout=0.1, weight_decay=0.01"
            echo ""
            echo "| Epoch | AUROC | F1 | Eval Loss |"
            echo "|-------|-------|----|-----------|"
            grep "eval_auroc" "$LOGFILE" | grep -oP "'eval_auroc': [0-9.]+|'eval_f1': [0-9.]+|'eval_loss': [0-9.]+|'epoch': [0-9.]+" \
                | paste - - - - \
                | awk -F"'" '{printf "| %s | %s | %s | %s |\n", $8, $4, $6, $2}' 2>/dev/null || true
        } > results/progress.md
        git add results/progress.md
        git diff --cached --quiet || git commit -m "Progress update: epoch $CURRENT_EPOCH"
        git push || true
    fi

    # Stop watcher if training is done
    if grep -q "All runs complete\|Done:" "$LOGFILE" 2>/dev/null; then break; fi
    if ! pgrep -f "finetune_all.py" > /dev/null 2>&1; then
        sleep 60
        if ! pgrep -f "finetune_all.py" > /dev/null 2>&1; then break; fi
    fi
done
