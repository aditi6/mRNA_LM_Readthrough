"""
Train XGBoost regressors on frozen NT embeddings for the PTC Toledano dataset.
Predicts continuous readthrough efficiency per drug using k-fold CV.

Run after extract_nt_embeddings_ptc.py:
    python train_xgb_ptc.py \
        --emb_dir /workspace/embeddings_ptc_nt \
        --out_dir /workspace/results_ptc_xgb

    # Single drug:
    python train_xgb_ptc.py --emb_dir ... --drug Gentamicin

    # Optuna hyperparameter search:
    python train_xgb_ptc.py --emb_dir ... --tune --trials 100
"""

import argparse
import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

DRUG_COLS = ['FUr', 'Gentamicin', 'CC90009', 'G418', 'Clitocine', 'DAP',
             'SJ6986', 'SRI', 'Untreated']


def load_data(emb_dir, drug):
    X = np.load(os.path.join(emb_dir, 'embeddings.npy'))
    y = np.load(os.path.join(emb_dir, f'labels_{drug}.npy'))

    # Remove rows where label is NaN (censored rows dropped at extraction time)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    print(f'  {drug}: {X.shape[0]} samples after dropping NaNs')
    return X, y


def make_xgb_params(seed=42, **overrides):
    params = dict(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=0,
    )
    params.update(overrides)
    return params


def eval_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pearson = float(pearsonr(y_true, y_pred)[0])
    spearman = float(spearmanr(y_true, y_pred)[0])
    return {'rmse': rmse, 'pearson': pearson, 'spearman': spearman}


def cross_validate(X, y, params, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        metrics = eval_metrics(y_val, preds)
        fold_results.append(metrics)
        print(f'  fold {fold+1}/{n_splits}  '
              f'RMSE={metrics["rmse"]:.4f}  '
              f'Pearson={metrics["pearson"]:.4f}  '
              f'Spearman={metrics["spearman"]:.4f}')

    mean_r = {k: float(np.mean([f[k] for f in fold_results])) for k in fold_results[0]}
    std_r  = {k: float(np.std( [f[k] for f in fold_results])) for k in fold_results[0]}
    print(f'  CV mean  '
          f'RMSE={mean_r["rmse"]:.4f}±{std_r["rmse"]:.4f}  '
          f'Pearson={mean_r["pearson"]:.4f}±{std_r["pearson"]:.4f}  '
          f'Spearman={mean_r["spearman"]:.4f}±{std_r["spearman"]:.4f}')
    return fold_results, mean_r, std_r


def tune_and_cv(X, y, n_splits=5, n_trials=100, seed=42):
    """Optuna hyperparameter search with nested CV."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError('Install optuna: pip install optuna')

    def objective(trial):
        params = dict(
            objective='reg:squarederror',
            eval_metric='rmse',
            n_estimators=trial.suggest_int('n_estimators', 500, 3000, step=100),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
            gamma=trial.suggest_float('gamma', 0.0, 2.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            random_state=seed, n_jobs=-1, verbosity=0,
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        spearmans = []
        for tr_idx, val_idx in kf.split(X):
            m = xgb.XGBRegressor(**params, early_stopping_rounds=40)
            m.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            preds = m.predict(X[val_idx])
            spearmans.append(spearmanr(y[val_idx], preds)[0])
        return float(np.mean(spearmans))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f'  Best CV Spearman: {best.value:.4f}')
    best_params = make_xgb_params(seed=seed, **best.params)
    return best_params, best.value, best.params


def run_drug(drug, emb_dir, out_dir, args):
    print(f'\n=== {drug} ===')
    try:
        X, y = load_data(emb_dir, drug)
    except FileNotFoundError:
        print(f'  labels_{drug}.npy not found, skipping.')
        return None

    if len(X) < args.cv_folds * 2:
        print(f'  Too few samples ({len(X)}) for {args.cv_folds}-fold CV, skipping.')
        return None

    if args.tune:
        print(f'  Optuna search ({args.trials} trials)...')
        params, best_cv_spearman, best_hp = tune_and_cv(
            X, y, n_splits=args.cv_folds, n_trials=args.trials, seed=args.seed)
    else:
        params = make_xgb_params(seed=args.seed)
        best_cv_spearman = None
        best_hp = None

    print(f'  {args.cv_folds}-fold CV...')
    fold_results, mean_r, std_r = cross_validate(X, y, params,
                                                  n_splits=args.cv_folds,
                                                  seed=args.seed)

    result = {
        'drug': drug,
        'n_samples': int(len(X)),
        'cv_mean': mean_r,
        'cv_std': std_r,
        'fold_results': fold_results,
    }
    if best_hp is not None:
        result['best_params'] = best_hp
        result['best_cv_spearman_optuna'] = best_cv_spearman

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir',  default='/workspace/embeddings_ptc_nt')
    parser.add_argument('--out_dir',  default='/workspace/results_ptc_xgb')
    parser.add_argument('--drug',     default=None,
                        help='single drug to run (default: all)')
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--tune',     action='store_true',
                        help='run Optuna hyperparameter search (requires optuna)')
    parser.add_argument('--trials',   type=int, default=100,
                        help='number of Optuna trials (--tune only)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    drugs = [args.drug] if args.drug else DRUG_COLS

    all_results = {}
    for drug in drugs:
        result = run_drug(drug, args.emb_dir, args.out_dir, args)
        if result is not None:
            all_results[drug] = result

    # Summary table
    print('\n=== Summary ===')
    print(f'{"Drug":12s}  {"N":>5}  {"RMSE":>8}  {"Pearson":>8}  {"Spearman":>9}')
    print('-' * 52)
    for drug, r in all_results.items():
        m = r['cv_mean']
        print(f'{drug:12s}  {r["n_samples"]:>5}  '
              f'{m["rmse"]:>8.4f}  {m["pearson"]:>8.4f}  {m["spearman"]:>9.4f}')

    out_path = os.path.join(args.out_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
