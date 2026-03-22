#!/bin/bash

# === Setup ===
source /opt/miniconda/etc/profile.d/conda.sh
conda activate readthrough
cd /workspace/mRNA_LM_Readthrough

# Reset finetune_all.py and set 20 epochs
git checkout finetune_all.py
sed -i 's/num_train_epochs=100/num_train_epochs=20/' finetune_all.py

mkdir -p /workspace/logs /workspace/mRNA_LM_Readthrough/results

run_experiment() {
    local name=$1
    shift
    local args="$@"
    local logfile="/workspace/logs/${name}.log"
    local outdir="/workspace/outputs/${name}"

    echo "========== Starting ${name} ==========" | tee "${logfile}"
    echo "Args: ${args}" | tee -a "${logfile}"
    echo "Start: $(date)" | tee -a "${logfile}"

    mkdir -p "${outdir}"

    # Run — don't abort script if this fails
    python finetune_all.py --task readthrough --output "${outdir}" ${args} >> "${logfile}" 2>&1
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "WARNING: ${name} exited with code ${exit_code}" | tee -a "${logfile}"
    fi

    echo "End: $(date)" | tee -a "${logfile}"

    # Extract metrics summary
    local summary="results/${name}_summary.txt"
    echo "Run: ${name}" > "${summary}"
    echo "Args: ${args}" >> "${summary}"
    grep -E "eval_loss|eval_auroc|eval_f1|train_loss|epoch" "${logfile}" \
        | grep -v "Map\|it/s\|examples" >> "${summary}" || true

    # Copy log to results
    cp "${logfile}" "results/${name}.log"

    # Push to GitHub (non-blocking — won't abort if push fails)
    git add results/ || true
    git diff --cached --quiet || git commit -m "Results: ${name} (20 epochs)"
    git push || echo "WARNING: git push failed for ${name}, will retry next run"
}

# Run A: baseline
run_experiment "run_A_baseline" \
    --lr 1e-4 --lorar 32 --lalpha 32 --ldropout 0.5 --head_dropout 0.5

# Run B: correct alpha + lower dropout
run_experiment "run_B_alpha64_dropout01" \
    --lr 1e-4 --lorar 32 --lalpha 64 --ldropout 0.1 --head_dropout 0.1

# Run C: larger LoRA + lower LR + lower dropout
run_experiment "run_C_r64_lr5e5_dropout01" \
    --lr 5e-5 --lorar 64 --lalpha 128 --ldropout 0.1 --head_dropout 0.1

# Final push to make sure everything is on GitHub
git add results/ || true
git diff --cached --quiet || git commit -m "All overnight runs complete"
git push || true

echo "========== All runs complete: $(date) =========="
