#!/usr/bin/env bash
# Run PTC pan-drug model across all context windows.
# Usage: bash run_ptc_pandrug.sh [DATA_CSV]
# Defaults to "PTC Toledano.csv" if no argument given.

set -e

DATA="${1:-PTC Toledano.csv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

echo "=== PTC Pan-Drug Regression ==="
echo "Data:   $DATA"
echo "Python: $($PY --version)"
echo ""

for CTX in 10 15 20 30 45 63; do
    echo "--- context_nt=$CTX ---"
    $PY "$SCRIPT_DIR/train_cnn_transformer_ptc_pandrug.py" \
        --data "$DATA" \
        --context_nt "$CTX" \
        --save_preds \
        --epochs 80 \
        --batch_size 256
    echo ""
done

echo "=== All context windows complete ==="
