import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

def run(emb_dir, label, drugs):
    X_all = np.load(f'{emb_dir}/embeddings.npy')
    print(f'\n=== {label} — {X_all.shape[1]}d embeddings, {X_all.shape[0]} sequences ===')
    print('CV: 5-fold stratified per drug')
    print(f'{"Drug":<12}  {"N":>6}  {"AUROC >1.0":>10}  {"AUROC median":>12}  {"% pos@1.0":>9}')
    print('-' * 58)
    for drug in drugs:
        y_raw = np.load(f'{emb_dir}/labels_{drug}.npy')
        valid = ~np.isnan(y_raw)
        X = X_all[valid]
        y_raw = y_raw[valid]
        results = {}
        for lbl, thr in [('gt1', 1.0), ('median', float(np.median(y_raw)))]:
            y = (y_raw > thr).astype(int)
            if y.sum() < 10 or (len(y) - y.sum()) < 10:
                results[lbl] = float('nan')
                continue
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
            aurocs = []
            for tr, te in skf.split(X, y):
                m = xgb.XGBClassifier(
                    objective='binary:logistic', eval_metric='auc',
                    n_estimators=1000, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=spw, random_state=42,
                    n_jobs=-1, verbosity=0, early_stopping_rounds=30)
                m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
                aurocs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
            results[lbl] = float(np.mean(aurocs))
        pos_pct = 100.0 * float((y_raw > 1.0).mean())
        g  = f"{results['gt1']:.4f}"    if not np.isnan(results.get('gt1',    float('nan'))) else '     N/A'
        m2 = f"{results['median']:.4f}" if not np.isnan(results.get('median', float('nan'))) else '      N/A'
        print(f'{drug:<12}  {len(X):>6}  {g:>10}  {m2:>12}  {pos_pct:>8.1f}%')

drugs = ['Clitocine', 'DAP', 'G418', 'SJ6986', 'SRI']
run('/workspace/embeddings_ntc_toledano_mrnaLM', 'mRNA-LM (CodonBERT + 3utrBERT) — NTC Toledano', drugs)
