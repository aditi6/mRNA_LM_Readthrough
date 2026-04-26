"""
CNN + Transformer hybrid v2 — improved for larger context windows.

Changes vs v1:
  - Stop-codon-relative positional encoding: integer distance from stop codon
    is embedded and added to the nucleotide embedding, giving the model an
    inductive bias to weight proximal positions more heavily.
  - Dropout scales with context size (larger window → more regularisation).
  - Weight decay increased to 1e-3.
  - --dropout arg exposed for manual override.
  - Local attention mask: each token attends only to a band of ±attn_window//2
    neighbours (default window=7). Prevents noise-fitting from distant positions
    and reduces quadratic attention cost for large contexts.
  - Stop-codon query pooling: replaces generic attention pooling with a single
    learned query vector that cross-attends over all transformer outputs. The
    readout is always computed from the stop-codon's "perspective" rather than
    having the model discover the center from scratch.

Architecture:
  (nt embedding + pos embedding) → 2x Conv1D → 2x Transformer (local attn)
  → stop-codon cross-attn pool → MLP

Usage:
  python train_cnn_transformer_ntc_v2.py --data "../NTC Toledano.csv" --context_nt 10
  python train_cnn_transformer_ntc_v2.py --data "../NTC Toledano.csv" --context_nt 20
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
STOP_POS = 63
NT2IDX   = {c: i for i, c in enumerate('acgtn')}
NT2IDX.update({'u': 3, 'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4})
VOCAB_SIZE = 5


# ── sequence utilities ────────────────────────────────────────────────────────
def extract_window(seq: str, context_nt: int = None,
                   upstream_nt: int = None, downstream_nt: int = None):
    """
    Returns (tokens, positions) where positions are distances from stop codon.
    Stop codon positions: 0, 1, 2
    Upstream (CDS): -upstream_nt, ..., -1
    Downstream (UTR): 3, 4, ..., 3+downstream_nt-1

    Pass context_nt for symmetric windows, or upstream_nt/downstream_nt for asymmetric.
    """
    if upstream_nt is None:
        upstream_nt = context_nt
    if downstream_nt is None:
        downstream_nt = context_nt
    start  = max(0, STOP_POS - upstream_nt)
    end    = min(len(seq), STOP_POS + 3 + downstream_nt)
    window = seq[start:end]
    tokens = np.array([NT2IDX.get(c, 4) for c in window], dtype=np.int64)
    # position relative to stop codon start
    positions = np.arange(start - STOP_POS, end - STOP_POS, dtype=np.int64)
    return tokens, positions


def pad_or_trim(arr: np.ndarray, target_len: int, pad_val: int = 0) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), constant_values=pad_val)


def default_dropout(context_nt: int) -> float:
    """Scale dropout with context: 0.2 at ±10nt, up to 0.4 at ±63nt."""
    return float(np.clip(0.2 + (context_nt - 10) * (0.2 / 53), 0.2, 0.4))


# ── model ─────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):           # x: (B, C, L)
        x = F.gelu(self.conv(x))
        x = self.drop(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class StopCodonQueryPool(nn.Module):
    """
    Cross-attention pool with a single learned stop-codon query.
    The query is fixed and learned; it cross-attends over all transformer
    outputs so the readout is always centered on the stop-codon perspective.
    """
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.query      = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=nhead, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_attn=False):      # x: (B, L, d_model)
        q   = self.query.expand(x.size(0), -1, -1)   # (B, 1, d_model)
        out, attn = self.cross_attn(q, x, x,
                                    need_weights=True,
                                    average_attn_weights=True)  # attn: (B, 1, L)
        out = self.norm(out.squeeze(1))                # (B, d_model)
        if return_attn:
            return out, attn.squeeze(1)                # (B, d_model), (B, L)
        return out


class ReadthroughModelV2(nn.Module):
    def __init__(self, seq_len, context_nt=None, upstream_nt=None, downstream_nt=None,
                 emb_dim=32, conv_ch=64,
                 d_model=64, nhead=4, ffn_dim=128, dropout=0.2, attn_window=7):
        super().__init__()
        # Support both symmetric (context_nt) and asymmetric (upstream_nt/downstream_nt)
        if upstream_nt is None:
            upstream_nt = context_nt
        if downstream_nt is None:
            downstream_nt = context_nt
        # Stop-codon-relative positional embedding
        self.pos_offset = upstream_nt
        pos_vocab       = upstream_nt + downstream_nt + 3
        self.nt_embed   = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=4)
        self.pos_embed  = nn.Embedding(pos_vocab + 2, emb_dim)

        self.conv1 = ConvBlock(emb_dim, conv_ch, kernel_size=5, dropout=dropout)
        self.conv2 = ConvBlock(conv_ch, d_model,  kernel_size=3, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Local attention mask: token i attends only to [i-half, i+half]
        # Registered as buffer so it moves to the right device automatically
        half       = attn_window // 2
        local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            lo = max(0, i - half)
            hi = min(seq_len, i + half + 1)
            local_mask[i, lo:hi] = False          # False = allowed to attend
        self.register_buffer('local_mask', local_mask)

        self.pool = StopCodonQueryPool(d_model, nhead=nhead)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, tokens, positions, return_attn=False):  # both: (B, L)
        pos_idx = (positions + self.pos_offset).clamp(min=0)
        x = self.nt_embed(tokens) + self.pos_embed(pos_idx)  # (B, L, emb_dim)
        x = self.conv1(x.permute(0, 2, 1))       # (B, conv_ch, L)
        x = self.conv2(x)                         # (B, d_model, L)
        x = self.transformer(                     # (B, L, d_model)
            x.permute(0, 2, 1), mask=self.local_mask)
        if return_attn:
            pooled, attn = self.pool(x, return_attn=True)
            return self.head(pooled).squeeze(-1), attn  # (B,), (B, L)
        return self.head(self.pool(x)).squeeze(-1)  # (B,)


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
    for tokens, positions, yb in loader:
        tokens, positions, yb = tokens.to(device), positions.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(tokens, positions), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []
    for tokens, positions, yb in loader:
        preds.append(model(tokens.to(device), positions.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    y_pred = scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(np.concatenate(trues).reshape(-1, 1)).ravel()
    return y_true, y_pred


def train_fold(model, train_loader, val_loader, scaler, args, device, ckpt_path=None):
    criterion = nn.HuberLoss() if args.loss == 'huber' else nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)

    best_r2, best_state, no_improve = -float('inf'), None, 0

    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = predict(model, val_loader, scaler, device)
        val_r2 = float(r2_score(y_true, y_pred))
        scheduler.step(val_r2)
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state,
                    'val_r2': best_r2,
                    'scaler_mean': scaler.mean_.tolist(),
                    'scaler_scale': scaler.scale_.tolist(),
                }, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    model.load_state_dict(best_state)
    return predict(model, val_loader, scaler, device)


# ── cross-validation ──────────────────────────────────────────────────────────
def run_cv(X_tok, X_pos, y, seq_len, context_nt, args, device, drug='drug',
           upstream_nt=None, downstream_nt=None):
    dropout = args.dropout if args.dropout is not None else default_dropout(context_nt)
    shuffle_sfx = '_shuffled' if getattr(args, 'shuffle_seq', False) else ''
    if upstream_nt is not None or downstream_nt is not None:
        window_tag = f'up{upstream_nt}_down{downstream_nt}nt'
    else:
        window_tag = f'context{context_nt}nt'
    suffix = f'_{window_tag}{shuffle_sfx}'
    print(f'  dropout={dropout:.2f}  weight_decay={args.weight_decay}  attn_window={args.attn_window}')

    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    all_true, all_pred = [], []
    pin = device.type == 'cuda'

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tok), 1):
        set_seed(args.seed + fold)
        sc = StandardScaler()
        y_tr_s  = sc.fit_transform(y[tr_idx].reshape(-1, 1)).ravel().astype(np.float32)
        y_val_s = sc.transform(y[val_idx].reshape(-1, 1)).ravel().astype(np.float32)

        train_ds = TensorDataset(
            torch.from_numpy(X_tok[tr_idx]),
            torch.from_numpy(X_pos[tr_idx]),
            torch.from_numpy(y_tr_s))
        val_ds = TensorDataset(
            torch.from_numpy(X_tok[val_idx]),
            torch.from_numpy(X_pos[val_idx]),
            torch.from_numpy(y_val_s))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_ds, batch_size=256,
                                  shuffle=False, num_workers=0, pin_memory=pin)

        model = ReadthroughModelV2(
            seq_len=seq_len, context_nt=context_nt,
            upstream_nt=upstream_nt, downstream_nt=downstream_nt,
            dropout=dropout, attn_window=args.attn_window).to(device)
        ckpt_path = os.path.join(args.out_dir, f'checkpoint_{drug}_fold{fold}{suffix}.pt')
        y_true, y_pred = train_fold(model, train_loader, val_loader, sc, args, device, ckpt_path=ckpt_path)
        all_true.append(y_true)
        all_pred.append(y_pred)
        m = eval_metrics(y_true, y_pred)
        fold_metrics.append(m)
        print(f'    fold {fold}: R²={m["r2"]:.4f}  Pearson={m["pearson"]:.4f}'
              f'  Spearman={m["spearman"]:.4f}  RMSE={m["rmse"]:.4f}')

    keys = fold_metrics[0].keys()
    mean = {k: float(np.mean([f[k] for f in fold_metrics])) for k in keys}
    std  = {k: float(np.std( [f[k] for f in fold_metrics])) for k in keys}
    preds = (np.concatenate(all_true), np.concatenate(all_pred))
    return mean, std, preds


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',         default='../NTC Toledano.csv')
    parser.add_argument('--out_dir',      default='results_v2')
    parser.add_argument('--drug',         default='all')
    parser.add_argument('--context_nt',   type=int, default=10)
    parser.add_argument('--dropout',      type=float, default=None,
                        help='override auto-scaled dropout')
    parser.add_argument('--attn_window', type=int, default=7,
                        help='local attention band width (each token attends to ±attn_window//2 neighbours)')
    parser.add_argument('--epochs',       type=int, default=200)
    parser.add_argument('--patience',     type=int, default=20)
    parser.add_argument('--batch_size',   type=int, default=512)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--loss',         default='huber', choices=['mse', 'huber'])
    parser.add_argument('--cv_folds',      type=int, default=5)
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--shuffle_seq',   action='store_true',
                        help='shuffle nucleotide tokens within each window (control experiment)')
    parser.add_argument('--upstream_nt',   type=int, default=None,
                        help='CDS context (overrides context_nt for upstream side)')
    parser.add_argument('--downstream_nt', type=int, default=None,
                        help='UTR context (overrides context_nt for downstream side)')
    parser.add_argument('--save_preds',    action='store_true',
                        help='save observed vs predicted arrays to npz for plotting')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)

    # Resolve upstream / downstream context
    upstream_nt   = args.upstream_nt   if args.upstream_nt   is not None else args.context_nt
    downstream_nt = args.downstream_nt if args.downstream_nt is not None else args.context_nt
    seq_len = upstream_nt + 3 + downstream_nt

    if args.upstream_nt is not None or args.downstream_nt is not None:
        print(f'Asymmetric window: {upstream_nt} nt upstream + stop codon + {downstream_nt} nt downstream  →  {seq_len} nt')
    else:
        print(f'Context: ±{args.context_nt} nt  →  window = {seq_len} nt')
    print(f'Sequences loaded: {len(df)}')

    # Encode all sequences once
    tokens_list, pos_list = [], []
    rng = np.random.default_rng(args.seed)
    for seq in df['nt_seq']:
        tok, pos = extract_window(seq, upstream_nt=upstream_nt, downstream_nt=downstream_nt)
        tok = pad_or_trim(tok, seq_len, pad_val=4)
        pos = pad_or_trim(pos, seq_len, pad_val=0)
        if args.shuffle_seq:
            tok = rng.permutation(tok)   # shuffle tokens, keep positions fixed
        tokens_list.append(tok)
        pos_list.append(pos)
    X_tok = np.stack(tokens_list)   # (N, seq_len) int64
    X_pos = np.stack(pos_list)      # (N, seq_len) int64

    if args.shuffle_seq:
        print('*** SHUFFLE CONTROL: nucleotide tokens randomly permuted per sequence ***')

    def parse_drug_col(series):
        def _parse(v):
            if isinstance(v, str) and v.startswith('>'):
                return float(v[1:])
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan
        return np.array([_parse(v) for v in series], dtype=np.float32)

    drugs = DRUGS if args.drug == 'all' else [args.drug]

    shuffle_sfx = '_shuffled' if args.shuffle_seq else ''
    if args.upstream_nt is not None or args.downstream_nt is not None:
        window_tag = f'up{upstream_nt}_down{downstream_nt}nt'
    else:
        window_tag = f'context{args.context_nt}nt'
    out_path = os.path.join(args.out_dir, f'results_{window_tag}{shuffle_sfx}.json')
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        print(f'Resuming — already done: {list(all_results.keys())}')
    else:
        all_results = {}

    for drug in drugs:
        if drug in all_results:
            print(f'\n=== {drug} — already done, skipping ===')
            continue

        print(f'\n=== {drug} ===')
        y_raw = parse_drug_col(df[drug])
        valid = ~np.isnan(y_raw)
        mean_m, std_m, preds = run_cv(
            X_tok[valid], X_pos[valid], y_raw[valid],
            seq_len, args.context_nt, args, device, drug=drug,
            upstream_nt=upstream_nt, downstream_nt=downstream_nt)
        if args.save_preds:
            preds_path = os.path.join(args.out_dir, f'preds_{drug}_{window_tag}{shuffle_sfx}.npz')
            np.savez(preds_path, y_true=preds[0], y_pred=preds[1])
            print(f'  [preds saved to {preds_path}]')

        print(f'  MEAN  R²={mean_m["r2"]:.4f}±{std_m["r2"]:.4f}'
              f'  Pearson={mean_m["pearson"]:.4f}±{std_m["pearson"]:.4f}'
              f'  Spearman={mean_m["spearman"]:.4f}±{std_m["spearman"]:.4f}'
              f'  RMSE={mean_m["rmse"]:.4f}±{std_m["rmse"]:.4f}')

        all_results[drug] = {'mean': mean_m, 'std': std_m,
                             'context_nt': args.context_nt, 'seq_len': seq_len}
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  [saved {out_path}]')

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
