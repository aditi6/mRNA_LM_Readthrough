"""
watch_cls.py — Pushes CLS pooling run progress to GitHub after each epoch.
Saves to results/run_cls_progress.txt (same format as run_window_progress.txt).
"""
import re, ast, datetime, subprocess, time, os

LOGFILE = "/workspace/train_cls.log"
OUTFILE = "/workspace/mRNA_LM_Readthrough/results/run_cls_progress.txt"
REPODIR = "/workspace/mRNA_LM_Readthrough"
LAST_PUSHED = 0

print("Watcher started:", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), flush=True)

def parse_log():
    lines = open(LOGFILE).readlines()
    evals, test = [], None
    for l in lines:
        c = re.sub(r"[^\x20-\x7e]", "", l).strip()
        m = re.search(r"(\{'eval_loss'[^}]+'eval_auroc'[^}]+\})", c)
        if m:
            try: evals.append(ast.literal_eval(m.group(1)))
            except: pass
        m2 = re.search(r"(\{'test_loss'[^}]+'test_auroc'[^}]+\})", c)
        if m2:
            try: test = ast.literal_eval(m2.group(1))
            except: pass
    return evals, test

def write_progress(evals, test, label):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = [
        "Run: run_cls",
        "Config: lr=5e-5, r=16, alpha=32, batch=16, dropout=0.1, epochs=25",
        "Sequences: full CDS + full 3'UTR (no windowing)",
        "Pooling: CLS token (not mean pooling)",
        "Last updated: " + now,
        "",
        "epoch | eval_loss | eval_f1 | eval_auroc",
        "------|-----------|---------|----------",
    ]
    for d in evals:
        rows.append("%5s | %9.4f | %7.4f | %10.4f" % (
            str(d.get("epoch", "?")),
            d.get("eval_loss", 0),
            d.get("eval_f1", 0),
            d.get("eval_auroc", 0),
        ))
    if evals:
        best = max(evals, key=lambda x: x.get("eval_auroc", 0))
        rows += [
            "",
            "Best AUROC: %.4f at epoch %s" % (best.get("eval_auroc", 0), best.get("epoch", "?")),
            "Baseline (non-windowed mean-pool): 0.5840",
        ]
    if test:
        rows += [
            "",
            "=== FINAL TEST RESULTS ===",
            "test_auroc: %.4f" % test.get("test_auroc", 0),
            "test_f1:    %.4f" % test.get("test_f1", 0),
            "test_loss:  %.4f" % test.get("test_loss", 0),
        ]
    open(OUTFILE, "w").write("\n".join(rows))

def git_push(epoch_label):
    os.chdir(REPODIR)
    subprocess.run(["git", "add", "results/run_cls_progress.txt"])
    r = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if r.returncode != 0:
        subprocess.run(["git", "commit", "-m", "CLS run progress: epoch %s/25" % epoch_label])
        # Pull remote changes before pushing to avoid rejection from local pushes
        subprocess.run(["git", "pull", "--rebase", "origin", "main"])
        subprocess.run(["git", "push"])
        print("Pushed epoch %s at %s" % (epoch_label, datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M UTC")), flush=True)
    else:
        print("No change at epoch %s" % epoch_label, flush=True)

while not os.path.exists(LOGFILE):
    print("Waiting for log file...", flush=True)
    time.sleep(30)

while True:
    time.sleep(90)

    try:
        evals, test = parse_log()
    except Exception as e:
        print("Parse error:", e, flush=True)
        continue

    if not evals:
        continue

    cur = int(max(d.get("epoch", 0) for d in evals))

    if cur > LAST_PUSHED:
        LAST_PUSHED = cur
        try:
            write_progress(evals, test, cur)
            git_push(cur)
        except Exception as e:
            print("Push error:", e, flush=True)

    r = subprocess.run(["pgrep", "-f", "finetune_cls.py"], capture_output=True)
    if r.returncode != 0:
        time.sleep(120)
        try:
            evals, test = parse_log()
            write_progress(evals, test, "FINAL")
            git_push("FINAL")
        except Exception as e:
            print("Final push error:", e, flush=True)
        print("Training complete. Watcher exiting.", flush=True)
        break
