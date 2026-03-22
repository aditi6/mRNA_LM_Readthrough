#!/bin/bash
# Pushes windowed run log to results/run_window.log on GitHub after each epoch.

cd /workspace/mRNA_LM_Readthrough
source /opt/miniconda/etc/profile.d/conda.sh
conda activate readthrough

LOGFILE=/workspace/train_window.log
RESULTS_LOG=results/run_window.log
LAST_PUSHED=0

echo "Watcher started: $(date)"

while true; do
    sleep 90

    [ ! -f "$LOGFILE" ] && continue

    # Count completed epochs from log
    CURRENT=$(python3 -c "
import re, ast
try:
    lines = open('/workspace/train_window.log').readlines()
    epochs = []
    for l in lines:
        clean = re.sub(r'[^\x20-\x7e]', '', l).strip()
        m = re.search(r\"(\{.*'eval_auroc'.*\})\", clean)
        if m:
            try:
                d = ast.literal_eval(m.group(1))
                epochs.append(float(d.get('epoch', 0)))
            except: pass
    print(int(max(epochs)) if epochs else 0)
except: print(0)
" 2>/dev/null || echo 0)

    if [ "$CURRENT" -gt "$LAST_PUSHED" ] 2>/dev/null; then
        LAST_PUSHED=$CURRENT
        cp $LOGFILE $RESULTS_LOG
        git add $RESULTS_LOG
        git diff --cached --quiet || git commit -m "Window run progress: epoch $CURRENT/10"
        git push || true
        echo "Pushed epoch $CURRENT at $(date)"
    fi

    # Exit once training is done
    if ! pgrep -f "finetune_window.py" > /dev/null 2>&1; then
        sleep 120
        cp $LOGFILE $RESULTS_LOG
        git add $RESULTS_LOG
        git diff --cached --quiet || git commit -m "Window run COMPLETE — final log"
        git push || true
        echo "Final push done at $(date)"
        break
    fi
done
