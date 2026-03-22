#!/bin/bash
cd /workspace/mRNA_LM_Readthrough
source /opt/miniconda/etc/profile.d/conda.sh
conda activate readthrough

LOGFILE=/workspace/logs/run_fixed.log
LAST_PUSHED=0

while true; do
    sleep 90

    [ ! -f "$LOGFILE" ] && continue

    # Use Python to cleanly parse the log
    python3 - << 'PYEOF'
import re, ast, os

logfile = "/workspace/logs/run_fixed.log"
lines = open(logfile).readlines()

eval_rows = []
for line in lines:
    # Strip ANSI/progress bar noise, find dict lines
    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
    clean = re.sub(r'[^\x20-\x7e]', '', clean).strip()
    m = re.search(r"(\{.*'eval_auroc'.*\})", clean)
    if m:
        try:
            d = ast.literal_eval(m.group(1))
            eval_rows.append(d)
        except:
            pass

if not eval_rows:
    print("No eval results yet")
    exit(0)

lines_out = []
lines_out.append("# Training Progress (auto-updated)")
lines_out.append(f"**Last update:** {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
lines_out.append("")
lines_out.append("**Config:** lr=1e-4, r=8, alpha=16, batch=16, dropout=0.1, weight_decay=0.01")
lines_out.append("")
lines_out.append("| Epoch | AUROC | F1 | Eval Loss |")
lines_out.append("|-------|-------|----|-----------|")
for d in eval_rows:
    lines_out.append(f"| {d.get('epoch','?')} | {d.get('eval_auroc',0):.4f} | {d.get('eval_f1',0):.4f} | {d.get('eval_loss',0):.4f} |")

best = max(eval_rows, key=lambda x: x.get('eval_auroc', 0))
lines_out.append("")
lines_out.append(f"**Best AUROC: {best.get('eval_auroc',0):.4f} at epoch {best.get('epoch','?')}**")

with open("results/progress.md", "w") as f:
    f.write("\n".join(lines_out))

print(f"Updated progress.md with {len(eval_rows)} epochs")
PYEOF

    CURRENT=$(python3 -c "
import re,ast
lines=open('/workspace/logs/run_fixed.log').readlines()
epochs=[ast.literal_eval(re.search(r\"(\{'eval_auroc'.*?\})\",re.sub(r'[^\x20-\x7e]','',l)).group(1)).get('epoch',0) for l in lines if 'eval_auroc' in l if re.search(r\"(\{'eval_auroc'.*?\})\",re.sub(r'[^\x20-\x7e]','',l))]
print(int(max(epochs)) if epochs else 0)
" 2>/dev/null || echo 0)

    if [ "$CURRENT" -gt "$LAST_PUSHED" ] 2>/dev/null; then
        LAST_PUSHED=$CURRENT
        git add results/progress.md
        git diff --cached --quiet || git commit -m "Progress: epoch $CURRENT/15"
        git push || true
    fi

    # Stop when training is done
    pgrep -f "finetune_all.py" > /dev/null 2>&1 || { sleep 120; pgrep -f "finetune_all.py" > /dev/null 2>&1 || break; }
done
