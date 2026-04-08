"""
CNN + Transformer hybrid for NTC Toledano readthrough regression.

Architecture:
  nucleotide embedding (dim=32)
  → 2x Conv1D (64 filters, kernels 5 & 3, GELU, LayerNorm, dropout=0.2)
  → 2x TransformerEncoderLayer (d_model=64, nhead=4, ffn=128, dropout=0.2)
  → attention pooling
  → MLP head: 64 → 32 → 1

Training:
  AdamW (lr=1e-3, weight_decay=1e-4), Huber/MSE loss, gradient clipping 1.0
  Early stopping on val R² (patience=20), ReduceLROnPlateau
  5-fold KFold CV, target standardised per fold (fit on train only)
  Reports mean ± std R², Pearson, Spearman, RMSE on original scale

Usage:
  python train_cnn_transformer_ntc.py --data "../NTC Toledano.csv"
  python train_cnn_transformer_ntc.py --data "../NTC Toledano.csv" --context_nt 20 --drug DAP
"""

import argparse, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# ── constants ─────────────────────────────────────────────────────────────────
DRUGS    = ['Clitocine', 'DAP', 'G418', 'SJ6986', 'SRI']
STOP_POS = 63           # 0-indexed position of stop codon in 132-nt window
NT2IDX   = {c: i for i, c in enumerate('acgtn')}
NT2IDX.update({'u': 3, 'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4})
VOCAB_SIZE = 5


# ── sequence utilities ────────────────────────────────────────────────────────
def extract_window(seq: str, context_nt: int) -> np.ndarray:
    """Return integer array for stop-codon window of length 2*context_nt+3."""
    start  = max(0, STOP_POS - context_nt)
    end    = min(len(seq), STOP_POS + 3 + context_nt)
    window = seq[start:end]
    return np.array([NT2IDX.get(c, 4) for c in window], dtype=np.int64)


def pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), constant_values=4)


# ── model ─────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):           # x: (B, C, L)
        x = F.gelu(self.conv(x))    # (B, out_ch, L)
        x = self.drop(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # LN over channel
        return x


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):                          # x: (B, L, d_model)
        w = torch.softmax(self.score(x), dim=1)   # (B, L, 1)
        return (w * x).sum(dim=1)                 # (B, d_model)


class ReadthroughModel(nn.Module):
    def __init__(self, emb_dim=32, conv_ch=64, d_model=64,
                 nhead=4, ffn_dim=128, dropout=0.2):
        super().__init__()
        self.embed       = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=4)
        self.conv1       = ConvBlock(emb_dim,  conv_ch, kernel_size=5, dropout=dropout)
        self.conv2       = ConvBlock(conv_ch,  d_model, kernel_size=3, dropout=dropout)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pool        = AttentionPool(d_model)
        self.head        = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):               # x: (B, L) int tokens
        x = self.embed(x)               # (B, L, emb_dim)
        x = self.conv1(x.permute(0, 2, 1))   # (B, conv_ch, L)
        x = self.conv2(x)               # (B, d_model, L)
        x = self.transformer(x.permute(0, 2, 1))  # (B, L, d_model)
        return self.head(self.pool(x)).squeeze(-1) # (B,)


# ── metrics ───────────────────────────────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    return dict(
        r2       = float(r2_score(y_true, y_pred)),
        pearson  = float(pearsonr(y_true, y_pred)[0]),
        spearman = float(spearmanr(y_true, y_pred)[0]),
        rmse     = float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


# ── training helpers ──────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, scaler, device):
    """Returns (y_true, y_pred) inverse-transformed to original scale."""
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    y_pred = scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(np.concatenate(trues).reshape(-1, 1)).ravel()
    return y_true, y_pred


def train_fold(model, train_loader, val_loader, scaler, args, device):
    criterion = nn.HuberLoss() if args.loss == 'huber' else nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)

    best_r2, best_state, no_improve = -float('inf'), None, 0

    for _ in range(args.epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = predict(model, val_loader, scaler, device)
        val_r2 = float(r2_score(y_true, y_pred))
        scheduler.step(val_r2)

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    model.load_state_dict(best_state)
    return predict(model, val_loader, scaler, device)


# ── cross-validation ──────────────────────────────────────────────────────────
def run_cv(X, y, args, device):
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    pin = device.type == 'cuda'

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        set_seed(args.seed + fold)
        X_tr,  X_val  = X[tr_idx],  X[val_idx]
        y_tr,  y_val  = y[tr_idx],  y[val_idx]

        sc = StandardScaler()
        y_tr_s  = sc.fit_transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32)
        y_val_s = sc.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_s)),
            batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_s)),
            batch_size=256, shuffle=False, num_workers=0, pin_memory=pin)

        model = ReadthroughModel().to(device)
        y_true, y_pred = train_fold(model, train_loader, val_loader, sc, args, device)
        m = eval_metrics(y_true, y_pred)
        fold_metrics.append(m)
        print(f'    fold {fold}: R²={m["r2"]:.4f}  Pearson={m["pearson"]:.4f}'
              f'  Spearman={m["spearman"]:.4f}  RMSE={m["rmse"]:.4f}')

    keys = fold_metrics[0].keys()
    mean = {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys}
    std  = {k: float(np.std( [f[k] for f in fold_metrics])) for k in keys}
    return mean, std


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',         default='../NTC Toledano.csv')
    parser.add_argument('--out_dir',      default='results')
    parser.add_argument('--drug',         default='all',
                        help='drug name or "all"')
    parser.add_argument('--context_nt',   type=int, default=10,
                        help='nt flanking each side of stop codon; window = 2*N+3')
    parser.add_argument('--epochs',       type=int, default=200)
    parser.add_argument('--patience',     type=int, default=20)
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss',         default='huber', choices=['mse', 'huber'])
    parser.add_argument('--cv_folds',     type=int, default=5)
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df      = pd.read_csv(args.data)
    seq_len = 2 * args.context_nt + 3
    print(f'Context: ±{args.context_nt} nt  →  window = {seq_len} nt')
    print(f'Sequences loaded: {len(df)}')

    X_all = np.stack([
        pad_or_trim(extract_window(seq, args.context_nt), seq_len)
        for seq in df['nt_seq']
    ])  # (N, seq_len) int64

    drugs = DRUGS if args.drug == 'all' else [args.drug]
    all_results = {}

    # Load any previously saved results (resume support)
    out_path = os.path.join(args.out_dir, f'results_context{args.context_nt}nt.json')
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        print(f'Resuming — already done: {list(all_results.keys())}')

    def parse_drug_col(series):
        """Handle censored values like '>2.59' by capping at that value."""
        def _parse(v):
            if isinstance(v, str) and v.startswith('>'):
                return float(v[1:])
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan
        return np.array([_parse(v) for v in series], dtype=np.float32)

    for drug in drugs:
        if drug in all_results:
            print(f'\n=== {drug} — already done, skipping ===')
            continue

        print(f'\n=== {drug} ===')
        y_raw = parse_drug_col(df[drug])
        valid = ~np.isnan(y_raw)
        X, y  = X_all[valid], y_raw[valid]
        print(f'  n={len(X)}')

        mean_m, std_m = run_cv(X, y, args, device)
        print(f'  MEAN  R²={mean_m["r2"]:.4f}±{std_m["r2"]:.4f}'
              f'  Pearson={mean_m["pearson"]:.4f}±{std_m["pearson"]:.4f}'
              f'  Spearman={mean_m["spearman"]:.4f}±{std_m["spearman"]:.4f}'
              f'  RMSE={mean_m["rmse"]:.4f}±{std_m["rmse"]:.4f}')

        all_results[drug] = {'mean': mean_m, 'std': std_m,
                             'context_nt': args.context_nt, 'seq_len': seq_len}
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  [saved {out_path}]')

    # Summary
    print(f'\n{"Drug":<12}  {"R²":>7}  {"Pearson":>8}  {"Spearman":>9}  {"RMSE":>7}')
    print('-' * 55)
    for drug in drugs:
        if drug not in all_results:
            continue
        m, s = all_results[drug]['mean'], all_results[drug]['std']
        print(f'{drug:<12}  {m["r2"]:>7.4f}±{s["r2"]:.4f}'
              f'  {m["pearson"]:>7.4f}±{s["pearson"]:.4f}'
              f'  {m["spearman"]:>8.4f}±{s["spearman"]:.4f}'
              f'  {m["rmse"]:>6.4f}±{s["rmse"]:.4f}')


if __name__ == '__main__':
    main()
