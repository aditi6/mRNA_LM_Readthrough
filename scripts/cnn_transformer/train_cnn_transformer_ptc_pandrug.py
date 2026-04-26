"""
Pan-drug CNN + Transformer regression on the PTC Toledano dataset.

Architecture: same drug-conditioned backbone as the treatment classifier,
but with a regression head and HuberLoss.

Key design choices vs per-drug training:
  - Each sequence × drug pair is one training sample (~52K pairs from 5837 sequences)
  - Drug embedding injected at every sequence position (shifts token representations)
  - DrugConditionedQueryPool: query shifted by drug embedding → drug-specific readout
  - CV splits on unique sequences (not pairs) to avoid any sequence leakage across folds
  - Per-drug StandardScaler fit on training fold; targets normalised per drug

STOP_POS = 72 (PTC stop codon is always at position 72 in these sequences)
Drugs: FUr, Gentamicin, CC90009, G418, Clitocine, DAP, SJ6986, SRI, Untreated

Usage:
  python train_cnn_transformer_ptc_pandrug.py --data "PTC Toledano.csv" --context_nt 45
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

# ── constants ──────────────────────────────────────────────────────────────────
DRUG_NAMES = ['FUr', 'Gentamicin', 'CC90009', 'G418',
              'Clitocine', 'DAP', 'SJ6986', 'SRI', 'Untreated']
STOP_POS   = 72
NT2IDX     = {c: i for i, c in enumerate('acgtn')}
NT2IDX.update({'u': 3, 'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4})
VOCAB_SIZE  = 5


# ── sequence utilities ─────────────────────────────────────────────────────────
def extract_window(seq: str, context_nt: int, upstream_nt=None, downstream_nt=None):
    up   = context_nt if upstream_nt   is None else upstream_nt
    down = context_nt if downstream_nt is None else downstream_nt
    start     = max(0, STOP_POS - up)
    end       = min(len(seq), STOP_POS + 3 + down)
    window    = seq[start:end]
    tokens    = np.array([NT2IDX.get(c, 4) for c in window], dtype=np.int64)
    positions = np.arange(start - STOP_POS, end - STOP_POS, dtype=np.int64)
    return tokens, positions


def pad_or_trim(arr, target_len, pad_val=0):
    if len(arr) == target_len: return arr
    if len(arr) > target_len:  return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), constant_values=pad_val)


def parse_col(series):
    def _p(v):
        if isinstance(v, str) and v.startswith('>'): return float(v[1:])
        try:    return float(v)
        except: return np.nan
    return np.array([_p(v) for v in series], dtype=np.float32)


# ── model ──────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.conv(x))
        x = self.drop(x)
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)


class DrugConditionedQueryPool(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.base_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.drug_proj  = nn.Linear(32, d_model, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=nhead, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, drug_emb, return_attn=False):
        q = self.base_query.expand(x.size(0), -1, -1) \
            + self.drug_proj(drug_emb).unsqueeze(1)
        out, attn = self.cross_attn(q, x, x,
                                    need_weights=True, average_attn_weights=True)
        out = self.norm(out.squeeze(1))
        if return_attn:
            return out, attn.squeeze(1)
        return out


class PanDrugRegressor(nn.Module):
    def __init__(self, n_drugs, seq_len, context_nt,
                 emb_dim=32, drug_emb_dim=32, conv_ch=64, d_model=64,
                 nhead=4, ffn_dim=128, dropout=0.3, attn_window=7):
        super().__init__()
        self.pos_offset    = context_nt
        pos_vocab          = 2 * context_nt + 3
        self.nt_embed      = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=4)
        self.pos_embed     = nn.Embedding(pos_vocab + 2, emb_dim)
        self.drug_embed    = nn.Embedding(n_drugs, drug_emb_dim)
        self.drug_seq_proj = nn.Linear(drug_emb_dim, emb_dim, bias=False)

        self.conv1 = ConvBlock(emb_dim, conv_ch, kernel_size=5, dropout=dropout)
        self.conv2 = ConvBlock(conv_ch, d_model,  kernel_size=3, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        half       = attn_window // 2
        local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            lo = max(0, i - half)
            hi = min(seq_len, i + half + 1)
            local_mask[i, lo:hi] = False
        self.register_buffer('local_mask', local_mask)

        self.pool = DrugConditionedQueryPool(d_model, nhead=nhead)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, tokens, positions, drug_ids, return_attn=False):
        pos_idx  = (positions + self.pos_offset).clamp(min=0)
        drug_emb = self.drug_embed(drug_ids)
        x = self.nt_embed(tokens) + self.pos_embed(pos_idx) \
            + self.drug_seq_proj(drug_emb).unsqueeze(1)
        x = self.conv1(x.permute(0, 2, 1))
        x = self.conv2(x)
        x = self.transformer(x.permute(0, 2, 1), mask=self.local_mask)
        if return_attn:
            pooled, attn = self.pool(x, drug_emb, return_attn=True)
            return self.head(pooled).squeeze(-1), attn
        return self.head(self.pool(x, drug_emb)).squeeze(-1)


# ── metrics ────────────────────────────────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    if len(y_true) < 2: return dict(r2=np.nan, pearson=np.nan, spearman=np.nan, rmse=np.nan)
    return dict(
        r2       = float(r2_score(y_true, y_pred)),
        pearson  = float(pearsonr(y_true, y_pred)[0]),
        spearman = float(spearmanr(y_true, y_pred)[0]),
        rmse     = float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


# ── training helpers ───────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for tokens, positions, drug_ids, yb in loader:
        tokens, positions = tokens.to(device), positions.to(device)
        drug_ids, yb      = drug_ids.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(tokens, positions, drug_ids), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets, drug_ids_out = [], [], []
    for tokens, positions, drug_ids, yb in loader:
        out = model(tokens.to(device), positions.to(device),
                    drug_ids.to(device)).cpu().numpy()
        preds.append(out)
        targets.append(yb.numpy())
        drug_ids_out.append(drug_ids.numpy())
    return (np.concatenate(preds),
            np.concatenate(targets),
            np.concatenate(drug_ids_out))


def inverse_transform_per_drug(arr_norm, drug_ids, scalers):
    out = np.zeros_like(arr_norm)
    for did, sc in scalers.items():
        mask = drug_ids == did
        if mask.sum() > 0:
            out[mask] = sc.inverse_transform(arr_norm[mask].reshape(-1, 1)).ravel()
    return out


# ── cross-validation ───────────────────────────────────────────────────────────
def run_cv(tokens, positions, drug_ids_all, targets_all, seq_idx_all,
           n_seqs, context_nt, args, device):
    """
    CV splits on unique sequence indices so no sequence appears in both
    train and val (even under a different drug).
    """
    seq_len = tokens.shape[1]
    dropout = args.dropout if args.dropout is not None \
              else float(np.clip(0.2 + (context_nt - 10) * (0.2 / 53), 0.2, 0.4))
    print(f'  dropout={dropout:.2f}  seq_len={seq_len}  n_pairs={len(tokens)}')

    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    pin = device.type == 'cuda'

    fold_metrics     = []
    fold_drug_metrics = []
    all_true_raw, all_pred_raw, all_drug_ids_out = [], [], []

    for fold, (tr_seq, val_seq) in enumerate(kf.split(np.arange(n_seqs)), 1):
        set_seed(args.seed + fold)

        tr_mask  = np.isin(seq_idx_all, tr_seq)
        val_mask = np.isin(seq_idx_all, val_seq)

        # Fit per-drug StandardScalers on training targets
        scalers = {}
        for did in range(len(DRUG_NAMES)):
            mask_d = tr_mask & (drug_ids_all == did)
            if mask_d.sum() > 1:
                sc = StandardScaler()
                sc.fit(targets_all[mask_d].reshape(-1, 1))
                scalers[did] = sc

        # Build normalised target arrays
        def normalise(mask):
            y_norm = np.zeros(mask.sum(), dtype=np.float32)
            dids   = drug_ids_all[mask]
            tgts   = targets_all[mask]
            for did, sc in scalers.items():
                m = dids == did
                if m.sum() > 0:
                    y_norm[m] = sc.transform(tgts[m].reshape(-1, 1)).ravel()
            return y_norm

        y_tr_norm  = normalise(tr_mask)
        y_val_norm = normalise(val_mask)

        train_ds = TensorDataset(
            torch.from_numpy(tokens[tr_mask]),
            torch.from_numpy(positions[tr_mask]),
            torch.from_numpy(drug_ids_all[tr_mask]),
            torch.from_numpy(y_tr_norm))
        val_ds = TensorDataset(
            torch.from_numpy(tokens[val_mask]),
            torch.from_numpy(positions[val_mask]),
            torch.from_numpy(drug_ids_all[val_mask]),
            torch.from_numpy(y_val_norm))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=512,
                                  shuffle=False, num_workers=0, pin_memory=pin)

        model = PanDrugRegressor(
            n_drugs=len(DRUG_NAMES), seq_len=seq_len, context_nt=context_nt,
            dropout=dropout, attn_window=args.attn_window).to(device)

        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)

        best_r2, best_state, no_improve = -float('inf'), None, 0
        ckpt_path = os.path.join(args.out_dir,
            f'checkpoint_pandrug_fold{fold}_ctx{context_nt}nt.pt')

        for epoch in range(args.epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_preds_norm, val_targets_norm, val_dids = predict(model, val_loader, device)

            # Inverse transform to original scale for R² computation
            val_pred_raw = inverse_transform_per_drug(val_preds_norm, val_dids, scalers)
            val_true_raw = inverse_transform_per_drug(val_targets_norm, val_dids, scalers)

            val_r2 = float(r2_score(val_true_raw, val_pred_raw))
            scheduler.step(val_r2)

            if val_r2 > best_r2:
                best_r2    = val_r2
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state,
                    'val_r2': best_r2,
                    'context_nt': context_nt,
                    'drug_names': DRUG_NAMES,
                    'scaler_means':  {did: sc.mean_.tolist()  for did, sc in scalers.items()},
                    'scaler_scales': {did: sc.scale_.tolist() for did, sc in scalers.items()},
                    'args': vars(args),
                }, ckpt_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

        # Final eval with best model
        model.load_state_dict(best_state)
        val_preds_norm, val_targets_norm, val_dids = predict(model, val_loader, device)
        val_pred_raw = inverse_transform_per_drug(val_preds_norm, val_dids, scalers)
        val_true_raw = inverse_transform_per_drug(val_targets_norm, val_dids, scalers)

        all_true_raw.append(val_true_raw)
        all_pred_raw.append(val_pred_raw)
        all_drug_ids_out.append(val_dids)

        m = eval_metrics(val_true_raw, val_pred_raw)
        fold_metrics.append(m)

        # Per-drug metrics for this fold
        drug_m = {}
        for did, dname in enumerate(DRUG_NAMES):
            mask = val_dids == did
            if mask.sum() >= 10:
                drug_m[dname] = eval_metrics(val_true_raw[mask], val_pred_raw[mask])
        fold_drug_metrics.append(drug_m)

        print(f'  fold {fold}: R²={m["r2"]:.4f}  Pearson={m["pearson"]:.4f}'
              f'  Spearman={m["spearman"]:.4f}  RMSE={m["rmse"]:.4f}'
              f'  (epoch {epoch+1}, best_r2={best_r2:.4f})')
        for dname, dm in drug_m.items():
            print(f'    {dname:<15} R²={dm["r2"]:.4f}  Pearson={dm["pearson"]:.4f}'
                  f'  Spearman={dm["spearman"]:.4f}')

    # Aggregate across folds
    keys = ['r2', 'pearson', 'spearman', 'rmse']
    mean_m = {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys}
    std_m  = {k: float(np.std( [f[k] for f in fold_metrics])) for k in keys}

    # Per-drug mean across folds
    per_drug_agg = {}
    for dname in DRUG_NAMES:
        vals = [f[dname] for f in fold_drug_metrics if dname in f]
        if vals:
            per_drug_agg[dname] = {k: float(np.mean([v[k] for v in vals])) for k in keys}

    all_true_cat    = np.concatenate(all_true_raw)
    all_pred_cat    = np.concatenate(all_pred_raw)
    all_dids_cat    = np.concatenate(all_drug_ids_out)

    return mean_m, std_m, per_drug_agg, (all_true_cat, all_pred_cat, all_dids_cat)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',         default='PTC Toledano.csv')
    parser.add_argument('--out_dir',      default='results_ptc_pandrug')
    parser.add_argument('--context_nt',   type=int, default=10,
                        help='symmetric context window; max 72 for PTC')
    parser.add_argument('--dropout',      type=float, default=None)
    parser.add_argument('--attn_window',  type=int, default=7)
    parser.add_argument('--epochs',       type=int, default=200)
    parser.add_argument('--patience',     type=int, default=20)
    parser.add_argument('--batch_size',   type=int, default=512)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--cv_folds',     type=int, default=5)
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--save_preds',    action='store_true')
    parser.add_argument('--shuffle_seq',   action='store_true',
                        help='shuffle nucleotide tokens (control)')
    parser.add_argument('--upstream_nt',   type=int, default=None,
                        help='CDS context (overrides context_nt upstream)')
    parser.add_argument('--downstream_nt', type=int, default=None,
                        help='UTR context (overrides context_nt downstream)')
    args = parser.parse_args()

    assert args.context_nt <= 72, 'PTC sequences only have 72 nt on each side'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data).dropna(subset=['nt_seq']).reset_index(drop=True)
    context_nt  = args.context_nt
    upstream_nt   = args.upstream_nt
    downstream_nt = args.downstream_nt
    up   = context_nt if upstream_nt   is None else upstream_nt
    down = context_nt if downstream_nt is None else downstream_nt
    seq_len = up + 3 + down
    n_seqs  = len(df)
    print(f'Sequences: {n_seqs}  |  Drugs: {len(DRUG_NAMES)}'
          f'  |  Context: -{up}/+{down} nt  →  window = {seq_len} nt')

    # Encode all sequences once
    tokens_list, pos_list = [], []
    rng = np.random.default_rng(42)
    for seq in df['nt_seq']:
        tok, pos = extract_window(seq, context_nt, upstream_nt, downstream_nt)
        tok = pad_or_trim(tok, seq_len, pad_val=4)
        if args.shuffle_seq:
            tok = rng.permutation(tok)
        tokens_list.append(tok)
        pos_list.append(pad_or_trim(pos, seq_len, pad_val=0))
    X_tok = np.stack(tokens_list).astype(np.int64)  # (n_seqs, seq_len)
    X_pos = np.stack(pos_list).astype(np.int64)     # (n_seqs, seq_len)

    # Build long-form dataset: one row per (sequence, drug) pair
    all_tokens, all_positions, all_drug_ids, all_targets, all_seq_idx = [], [], [], [], []
    for did, drug in enumerate(DRUG_NAMES):
        y_raw     = parse_col(df[drug])
        valid_idx = np.where(~np.isnan(y_raw))[0]
        all_tokens.append(X_tok[valid_idx])
        all_positions.append(X_pos[valid_idx])
        all_drug_ids.append(np.full(len(valid_idx), did, dtype=np.int64))
        all_targets.append(y_raw[valid_idx])
        all_seq_idx.append(valid_idx)
        print(f'  {drug:<15} n={len(valid_idx)}')

    tokens_all   = np.concatenate(all_tokens)
    positions_all = np.concatenate(all_positions)
    drug_ids_all  = np.concatenate(all_drug_ids)
    targets_all   = np.concatenate(all_targets).astype(np.float32)
    seq_idx_all   = np.concatenate(all_seq_idx)

    print(f'\nTotal training pairs: {len(tokens_all)}'
          f'  |  Running {args.cv_folds}-fold CV (split on sequences)')

    # Build suffix for ablation naming
    if args.shuffle_seq:
        suffix = f'ctx{context_nt}nt_shuffled'
    elif upstream_nt is not None or downstream_nt is not None:
        suffix = f'up{up}_down{down}nt'
    else:
        suffix = f'ctx{context_nt}nt'

    # Check if already done
    out_path = os.path.join(args.out_dir, f'results_{suffix}.json')
    if os.path.exists(out_path):
        print(f'Results already exist at {out_path}, skipping.')
        return

    mean_m, std_m, per_drug_m, raw_preds = run_cv(
        tokens_all, positions_all, drug_ids_all, targets_all, seq_idx_all,
        n_seqs, context_nt, args, device)

    print(f'\n── OVERALL (mean ± std across {args.cv_folds} folds) ──')
    print(f'R²={mean_m["r2"]:.4f}±{std_m["r2"]:.4f}'
          f'  Pearson={mean_m["pearson"]:.4f}±{std_m["pearson"]:.4f}'
          f'  Spearman={mean_m["spearman"]:.4f}±{std_m["spearman"]:.4f}'
          f'  RMSE={mean_m["rmse"]:.4f}±{std_m["rmse"]:.4f}')

    print(f'\n── PER DRUG ──')
    print(f'{"Drug":<15}  {"R²":>7}  {"Pearson":>8}  {"Spearman":>9}  {"RMSE":>7}')
    print('-' * 55)
    for dname in DRUG_NAMES:
        if dname not in per_drug_m: continue
        m = per_drug_m[dname]
        print(f'{dname:<15}  {m["r2"]:>7.4f}  {m["pearson"]:>8.4f}'
              f'  {m["spearman"]:>9.4f}  {m["rmse"]:>7.4f}')

    results = dict(mean=mean_m, std=std_m, per_drug=per_drug_m,
                   config=vars(args), drug_names=DRUG_NAMES)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')

    if args.save_preds:
        preds_path = os.path.join(args.out_dir, f'preds_{suffix}.npz')
        np.savez(preds_path,
                 y_true=raw_preds[0], y_pred=raw_preds[1],
                 drug_ids=raw_preds[2],
                 drug_names=np.array(DRUG_NAMES))
        print(f'Predictions saved to {preds_path}')


if __name__ == '__main__':
    main()
