"""
Regression on frozen NT-v2-500M embeddings for NTC Toledano dataset.
Reports R², Pearson r, Spearman r per drug.
Tries multiple models: XGBoost (default), XGBoost (Optuna-tuned), LightGBM, Ridge.

Results are saved after each drug completes — safe to stop mid-run.

Run:
    python regress_ntc_toledano.py \
        --emb_dir /workspace/embeddings_ntc_toledano_nt \
        --out_dir /workspace/results_ntc_toledano_regression

    # With Optuna tuning (slower):
    python regress_ntc_toledano.py ... --tune --trials 50
"""

import argparse, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import xgboost as xgb

DRUGS = ['Clitocine', 'DAP', 'G418', 'SJ6986', 'SRI']


def eval_metrics(y_true, y_pred):
    # AUROC using regression scores as ranking (median split for binary labels)
    y_bin = (y_true > np.median(y_true)).astype(int)
    auroc = float(roc_auc_score(y_bin, y_pred)) if y_bin.sum() > 0 else float('nan')
    return {
        'r2':       float(r2_score(y_true, y_pred)),
        'pearson':  float(pearsonr(y_true, y_pred)[0]),
        'spearman': float(spearmanr(y_true, y_pred)[0]),
        'rmse':     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'auroc':    auroc,
    }


def cv_model(model_fn, X, y, n_splits=5, seed=42, use_eval_set=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    for tr, te in kf.split(X):
        m = model_fn()
        if use_eval_set:
            m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        else:
            m.fit(X[tr], y[tr])
        fold_results.append(eval_metrics(y[te], m.predict(X[te])))
    mean = {k: float(np.mean([f[k] for f in fold_results])) for k in fold_results[0]}
    std  = {k: float(np.std( [f[k] for f in fold_results])) for k in fold_results[0]}
    return mean, std


def tune_xgb(X, y, n_splits=5, n_trials=50, seed=42):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            objective='reg:squarederror',
            n_estimators=trial.suggest_int('n_estimators', 300, 2000, step=100),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
            gamma=trial.suggest_float('gamma', 0.0, 2.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            random_state=seed, n_jobs=-1, verbosity=0,
            device='cuda', tree_method='hist',
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []
        for tr, te in kf.split(X):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr], verbose=False)
            scores.append(spearmanr(y[te], m.predict(X[te]))[0])
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def save_results(all_results, out_dir):
    """Save combined results.json — called after every drug completes."""
    out_path = os.path.join(out_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir', default='/workspace/embeddings_ntc_toledano_nt')
    parser.add_argument('--out_dir', default='/workspace/results_ntc_toledano_regression')
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--tune',    action='store_true')
    parser.add_argument('--trials',  type=int, default=50)
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X_all = np.load(f'{args.emb_dir}/embeddings.npy')
    print(f'Embeddings: {X_all.shape}')

    # Load any previously saved results so we can skip completed drugs
    combined_path = os.path.join(args.out_dir, 'results.json')
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            all_results = json.load(f)
        print(f'Resuming — already done: {list(all_results.keys())}')
    else:
        all_results = {}

    for drug in DRUGS:
        if drug in all_results:
            print(f'\n=== {drug} — already done, skipping ===')
            continue

        y_raw = np.load(f'{args.emb_dir}/labels_{drug}.npy')
        valid = ~np.isnan(y_raw)
        X, y = X_all[valid], y_raw[valid]
        print(f'\n=== {drug} (n={len(X)}) ===')

        drug_results = {}

        # ── Ridge (linear baseline) ───────────────────────────────────────────
        print('  Ridge...', end=' ', flush=True)
        mean_r, std_r = cv_model(lambda: Ridge(alpha=1.0), X, y, args.cv_folds, args.seed)
        drug_results['ridge'] = {'mean': mean_r, 'std': std_r}
        print(f"R²={mean_r['r2']:.4f}  Pearson={mean_r['pearson']:.4f}  Spearman={mean_r['spearman']:.4f}  AUROC={mean_r['auroc']:.4f}")

        # ── XGBoost default ───────────────────────────────────────────────────
        print('  XGBoost default...', end=' ', flush=True)
        def xgb_default():
            return xgb.XGBRegressor(
                objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8,
                random_state=args.seed, n_jobs=-1, verbosity=0,
                device='cuda', tree_method='hist')
        mean_r, std_r = cv_model(xgb_default, X, y, args.cv_folds, args.seed)
        drug_results['xgb_default'] = {'mean': mean_r, 'std': std_r}
        print(f"R²={mean_r['r2']:.4f}  Pearson={mean_r['pearson']:.4f}  Spearman={mean_r['spearman']:.4f}  AUROC={mean_r['auroc']:.4f}")

        # ── LightGBM ──────────────────────────────────────────────────────────
        try:
            import lightgbm as lgb
            print('  LightGBM...', end=' ', flush=True)
            def lgbm_fn():
                return lgb.LGBMRegressor(
                    n_estimators=1000, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=args.seed, n_jobs=-1, verbose=-1)
            mean_r, std_r = cv_model(lgbm_fn, X, y, args.cv_folds, args.seed)
            drug_results['lightgbm'] = {'mean': mean_r, 'std': std_r}
            print(f"R²={mean_r['r2']:.4f}  Pearson={mean_r['pearson']:.4f}  Spearman={mean_r['spearman']:.4f}  AUROC={mean_r['auroc']:.4f}")
        except ImportError:
            print('  LightGBM not installed, skipping.')

        # ── XGBoost Optuna-tuned ──────────────────────────────────────────────
        if args.tune:
            print(f'  XGBoost Optuna ({args.trials} trials)...', flush=True)
            best_params, best_cv = tune_xgb(X, y, args.cv_folds, args.trials, args.seed)
            print(f'  Best CV Spearman: {best_cv:.4f}  params: {best_params}')
            def xgb_tuned():
                return xgb.XGBRegressor(
                    objective='reg:squarederror', random_state=args.seed,
                    n_jobs=-1, verbosity=0, device='cuda', tree_method='hist',
                    **best_params)
            mean_r, std_r = cv_model(xgb_tuned, X, y, args.cv_folds, args.seed)
            drug_results['xgb_tuned'] = {'mean': mean_r, 'std': std_r, 'best_params': best_params}
            print(f"  Tuned  R²={mean_r['r2']:.4f}  Pearson={mean_r['pearson']:.4f}  Spearman={mean_r['spearman']:.4f}  AUROC={mean_r['auroc']:.4f}")

        all_results[drug] = drug_results

        # Save after every drug so partial results are never lost
        save_results(all_results, args.out_dir)
        print(f'  [saved {args.out_dir}/results.json]')

    # ── Summary table ─────────────────────────────────────────────────────────
    models = list(next(iter(all_results.values())).keys())
    for model in models:
        print(f'\n--- {model} ---')
        print(f'{"Drug":<12}  {"R²":>7}  {"Pearson":>8}  {"Spearman":>9}  {"AUROC":>7}  {"RMSE":>7}')
        print('-' * 60)
        for drug in DRUGS:
            if drug not in all_results or model not in all_results[drug]:
                continue
            m = all_results[drug][model]['mean']
            print(f'{drug:<12}  {m["r2"]:>7.4f}  {m["pearson"]:>8.4f}  {m["spearman"]:>9.4f}  {m["auroc"]:>7.4f}  {m["rmse"]:>7.4f}')

    print(f'\nAll done. Results in {args.out_dir}/results.json')


if __name__ == '__main__':
    main()
