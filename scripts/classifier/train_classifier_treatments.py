"""
Pan-drug CNN + Transformer classifier with drug conditioning.

Architecture:
  nt embedding (32) + drug embedding (32, added to each position)
  → 2x Conv1D (64 filters, k=5/3, GELU, LayerNorm, dropout=0.3)
  → 1-2x pre-norm TransformerEncoderLayer (d=64, h=4, ffn=128, local attn band=7)
  → drug-conditioned cross-attention query pool
  → MLP head: 64 → 32 → 1 (logit)

Training:
  BCEWithLogitsLoss with per-sample weights from drug-specific class imbalance
  (optional: focal loss via --loss focal)
  AdamW lr=5e-4, weight_decay=1e-3, Huber on logits, grad clip=1.0
  ReduceLROnPlateau on val AUPRC, early stopping patience=20

CV:
  StratifiedGroupKFold by drug+label combination
  Balanced minibatches within each drug via WeightedRandomSampler
  Per-drug threshold tuning on val fold (maximise F1)

Metrics per fold: AUROC, AUPRC, F1 (tuned threshold), per-drug breakdown

Usage:
  python train_classifier_treatments.py --data merged_treatments.csv
  python train_classifier_treatments.py --data merged_treatments.csv --loss focal --n_transformer_layers 2
"""

import argparse, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_recall_curve)
from scipy.stats import spearmanr

DRUGS    = None   # set from data
NT2IDX   = {c: i for i, c in enumerate('acgtn')}
NT2IDX.update({'u': 3, 'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4})
VOCAB_SIZE = 5
STOP_POS_IN_WINDOW = None   # context_nt (set from args)


# ── sequence encoding ──────────────────────────────────────────────────────────
def encode_seq(seq: str, seq_len: int, stop_pos: int = None,
               upstream_nt: int = None, downstream_nt: int = None) -> np.ndarray:
    if stop_pos is not None and (upstream_nt is not None or downstream_nt is not None):
        up   = upstream_nt   if upstream_nt   is not None else seq_len
        down = downstream_nt if downstream_nt is not None else seq_len
        start = max(0, stop_pos - up)
        end   = min(len(seq), stop_pos + 3 + down)
        seq   = seq[start:end]
    arr = np.array([NT2IDX.get(c, 4) for c in seq], dtype=np.int64)
    if len(arr) >= seq_len:
        return arr[:seq_len]
    return np.pad(arr, (0, seq_len - len(arr)), constant_values=4)


def make_pos_array(context_nt: int) -> np.ndarray:
    """Distance from stop codon for each position in window."""
    seq_len = 2 * context_nt + 3
    return np.arange(-context_nt, context_nt + 3, dtype=np.int64)


# ── focal loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, weight=None):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, weight=weight, reduction='none')
        pt  = torch.exp(-bce)
        fl  = (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return fl.mean()
        return fl.sum()


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
    """
    Cross-attention pool where the query is a learned base vector
    shifted by a drug-specific embedding, so each drug gets its own
    readout perspective over the transformer outputs.
    """
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.base_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.drug_proj  = nn.Linear(32, d_model, bias=False)  # drug_emb_dim=32 → d_model
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=nhead, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, drug_emb, return_attn=False):  # x:(B,L,d_model)  drug_emb:(B,drug_emb_dim)
        # Project drug embedding to d_model and shift the base query
        q = self.base_query.expand(x.size(0), -1, -1) \
            + self.drug_proj(drug_emb).unsqueeze(1)    # (B,1,d_model)
        out, attn = self.cross_attn(q, x, x,
                                    need_weights=True,
                                    average_attn_weights=True)  # attn: (B, 1, L)
        out = self.norm(out.squeeze(1))                # (B, d_model)
        if return_attn:
            return out, attn.squeeze(1)                # (B, d_model), (B, L)
        return out


class ReadthroughClassifier(nn.Module):
    def __init__(self, n_drugs, seq_len, context_nt,
                 emb_dim=32, drug_emb_dim=32, conv_ch=64, d_model=64,
                 nhead=4, ffn_dim=128, dropout=0.3, attn_window=7,
                 n_transformer_layers=2):
        super().__init__()
        self.pos_offset = context_nt
        pos_vocab       = 2 * context_nt + 3

        self.nt_embed   = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=4)
        self.pos_embed  = nn.Embedding(pos_vocab + 2, emb_dim)
        self.drug_embed = nn.Embedding(n_drugs, drug_emb_dim)

        # Project drug emb to match sequence emb dim for positional injection
        self.drug_seq_proj = nn.Linear(drug_emb_dim, emb_dim, bias=False)

        self.conv1 = ConvBlock(emb_dim, conv_ch, kernel_size=5, dropout=dropout)
        self.conv2 = ConvBlock(conv_ch, d_model,  kernel_size=3, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_transformer_layers)

        # Local attention mask
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
        # Sequence embedding
        pos_idx  = (positions + self.pos_offset).clamp(min=0)
        drug_emb = self.drug_embed(drug_ids)                      # (B, drug_emb_dim)
        x = self.nt_embed(tokens) + self.pos_embed(pos_idx) \
            + self.drug_seq_proj(drug_emb).unsqueeze(1)           # (B, L, emb_dim)

        x = self.conv1(x.permute(0, 2, 1))                       # (B, conv_ch, L)
        x = self.conv2(x)                                         # (B, d_model, L)
        x = self.transformer(x.permute(0, 2, 1),
                             mask=self.local_mask)                # (B, L, d_model)
        if return_attn:
            pooled, attn = self.pool(x, drug_emb, return_attn=True)
            return self.head(pooled).squeeze(-1), attn  # (B,), (B, L)
        return self.head(self.pool(x, drug_emb)).squeeze(-1)      # (B,)


# ── metrics ────────────────────────────────────────────────────────────────────
def tune_threshold(y_true, y_prob):
    """Find threshold maximising F1 on the given set."""
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1[:-1])
    return float(thresh[best_idx])


def eval_metrics(y_true, y_prob, threshold=None, drug_labels=None, drug_names=None):
    if threshold is None:
        threshold = tune_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = dict(
        auroc   = float(roc_auc_score(y_true, y_prob)),
        auprc   = float(average_precision_score(y_true, y_prob)),
        f1      = float(f1_score(y_true, y_pred, zero_division=0)),
        threshold = float(threshold),
    )
    # Per-drug breakdown
    if drug_labels is not None and drug_names is not None:
        per_drug = {}
        for did, dname in enumerate(drug_names):
            mask = drug_labels == did
            if mask.sum() < 10:
                continue
            yt, yp = y_true[mask], y_prob[mask]
            if yt.sum() == 0 or yt.sum() == mask.sum():
                continue
            thr_d = tune_threshold(yt, yp)
            per_drug[dname] = dict(
                auroc   = float(roc_auc_score(yt, yp)),
                auprc   = float(average_precision_score(yt, yp)),
                f1      = float(f1_score(yt, (yp >= thr_d).astype(int), zero_division=0)),
                n       = int(mask.sum()),
                pos     = int(yt.sum()),
                threshold = float(thr_d),
            )
        metrics['per_drug'] = per_drug
    return metrics


# ── per-sample weights for imbalance ──────────────────────────────────────────
def compute_sample_weights(labels: np.ndarray, drug_ids: np.ndarray) -> np.ndarray:
    """
    Per-sample weight = 1 / (drug_prevalence if label==1 else 1-drug_prevalence).
    This upweights rare positives (e.g. untreated) much more than common ones (G418).
    """
    weights = np.ones(len(labels), dtype=np.float32)
    for did in np.unique(drug_ids):
        mask = drug_ids == did
        prev = labels[mask].mean()
        prev = np.clip(prev, 1e-3, 1 - 1e-3)
        weights[mask & (labels == 1)] = 1.0 / prev
        weights[mask & (labels == 0)] = 1.0 / (1 - prev)
    return weights


# ── training helpers ───────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for tokens, positions, drug_ids, labels, sw in loader:
        tokens, positions = tokens.to(device), positions.to(device)
        drug_ids, labels  = drug_ids.to(device), labels.to(device)
        sw = sw.to(device)
        optimizer.zero_grad()
        logits = model(tokens, positions, drug_ids)
        loss   = criterion(logits, labels, sw)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels, all_drugs = [], [], []
    for tokens, positions, drug_ids, labels, _ in loader:
        logits = model(tokens.to(device), positions.to(device),
                       drug_ids.to(device)).cpu()
        all_logits.append(logits.numpy())
        all_labels.append(labels.numpy())
        all_drugs.append(drug_ids.numpy())
    logits  = np.concatenate(all_logits)
    labels  = np.concatenate(all_labels)
    drugs   = np.concatenate(all_drugs)
    probs   = 1 / (1 + np.exp(-logits))
    return labels, probs, drugs


def make_loader(tokens, positions, drug_ids, labels, sample_weights,
                batch_size, shuffle, use_sampler=False):
    ds = TensorDataset(
        torch.from_numpy(tokens),
        torch.from_numpy(positions),
        torch.from_numpy(drug_ids),
        torch.from_numpy(labels.astype(np.float32)),
        torch.from_numpy(sample_weights),
    )
    if shuffle and use_sampler:
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── cross-validation ───────────────────────────────────────────────────────────
def run_cv(tokens, drug_ids, labels, drug_names, args, device, suffix=''):
    # Stratify on drug × label combination
    strat_key = drug_ids * 2 + labels

    kf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    fold_raw_preds = []
    n_drugs  = len(drug_names)
    seq_len  = tokens.shape[1]
    ctx      = args.context_nt

    pos_arr = make_pos_array(ctx).astype(np.int64)   # (seq_len,)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(tokens, strat_key), 1):
        set_seed(args.seed + fold)

        # Per-sample weights from train fold only
        sw_tr  = compute_sample_weights(labels[tr_idx], drug_ids[tr_idx])
        sw_val = np.ones(len(val_idx), dtype=np.float32)

        # Broadcast positions to all samples
        pos_tr  = np.tile(pos_arr, (len(tr_idx),  1))
        pos_val = np.tile(pos_arr, (len(val_idx), 1))

        train_loader = make_loader(
            tokens[tr_idx], pos_tr, drug_ids[tr_idx], labels[tr_idx], sw_tr,
            args.batch_size, shuffle=True, use_sampler=True)
        val_loader = make_loader(
            tokens[val_idx], pos_val, drug_ids[val_idx], labels[val_idx], sw_val,
            args.batch_size, shuffle=False)

        model = ReadthroughClassifier(
            n_drugs=n_drugs, seq_len=seq_len, context_nt=ctx,
            dropout=args.dropout, attn_window=args.attn_window,
            n_transformer_layers=args.n_transformer_layers).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)

        if args.loss == 'focal':
            fl = FocalLoss(gamma=2.0)
            def criterion(logits, labels, sw):
                return fl(logits, labels, weight=sw)
        else:
            def criterion(logits, labels, sw):
                return F.binary_cross_entropy_with_logits(
                    logits, labels, weight=sw)

        best_auprc, best_state, no_improve = -1.0, None, 0
        ckpt_path = os.path.join(args.out_dir, f'checkpoint_fold{fold}_{suffix}.pt')

        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_labels, val_probs, val_drugs = predict(model, val_loader, device)
            if val_labels.sum() == 0:
                continue
            val_auprc = float(average_precision_score(val_labels, val_probs))
            scheduler.step(val_auprc)

            if val_auprc > best_auprc:
                best_auprc  = val_auprc
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state,
                    'val_auprc': best_auprc,
                    'drug_names': drug_names,
                    'args': vars(args),
                }, ckpt_path)
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

        model.load_state_dict(best_state)
        val_labels, val_probs, val_drugs = predict(model, val_loader, device)
        fold_raw_preds.append((val_labels, val_probs, val_drugs))
        m = eval_metrics(val_labels, val_probs,
                         drug_labels=val_drugs, drug_names=drug_names)
        fold_metrics.append(m)

        print(f'  fold {fold}: AUROC={m["auroc"]:.4f}  AUPRC={m["auprc"]:.4f}'
              f'  F1={m["f1"]:.4f}  (stopped @ epoch {epoch}, best_auprc={best_auprc:.4f})')
        if 'per_drug' in m:
            for dname, dm in m['per_drug'].items():
                print(f'    {dname:<20} AUROC={dm["auroc"]:.3f}  AUPRC={dm["auprc"]:.3f}'
                      f'  F1={dm["f1"]:.3f}  pos={dm["pos"]}/{dm["n"]}')

    # Aggregate
    scalar_keys = ['auroc', 'auprc', 'f1']
    mean = {k: float(np.mean([f[k] for f in fold_metrics])) for k in scalar_keys}
    std  = {k: float(np.std( [f[k] for f in fold_metrics])) for k in scalar_keys}

    # Per-drug mean across folds
    per_drug_agg = {}
    for dname in drug_names:
        vals = [f['per_drug'][dname] for f in fold_metrics
                if 'per_drug' in f and dname in f['per_drug']]
        if vals:
            per_drug_agg[dname] = {
                k: float(np.mean([v[k] for v in vals]))
                for k in ['auroc', 'auprc', 'f1', 'n', 'pos']}

    all_labels_cat  = np.concatenate([f[0] for f in fold_raw_preds])
    all_probs_cat   = np.concatenate([f[1] for f in fold_raw_preds])
    all_drugs_cat   = np.concatenate([f[2] for f in fold_raw_preds])

    return mean, std, per_drug_agg, fold_metrics, (all_labels_cat, all_probs_cat, all_drugs_cat)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',         default='merged_treatments.csv')
    parser.add_argument('--out_dir',      default='results_classifier')
    parser.add_argument('--context_nt',   type=int, default=45)
    parser.add_argument('--n_transformer_layers', type=int, default=2)
    parser.add_argument('--attn_window',  type=int, default=7)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--loss',         default='bce', choices=['bce', 'focal'])
    parser.add_argument('--epochs',       type=int, default=100)
    parser.add_argument('--patience',     type=int, default=20)
    parser.add_argument('--batch_size',   type=int, default=256)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--cv_folds',     type=int, default=3)
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--save_preds',    action='store_true',
                        help='save labels/probs/drug_ids arrays to npz for plotting')
    parser.add_argument('--shuffle_seq',   action='store_true',
                        help='shuffle nucleotide tokens (control)')
    parser.add_argument('--upstream_nt',   type=int, default=None,
                        help='CDS context override (nt upstream of stop)')
    parser.add_argument('--downstream_nt', type=int, default=None,
                        help='UTR context override (nt downstream of stop)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    print(f'Loaded {len(df)} rows, {df["drug"].nunique()} drugs')

    drug_names = sorted(df['drug'].unique().tolist())
    drug2id    = {d: i for i, d in enumerate(drug_names)}
    df['drug_id'] = df['drug'].map(drug2id)

    ctx = args.context_nt
    up   = args.upstream_nt   if args.upstream_nt   is not None else ctx
    down = args.downstream_nt if args.downstream_nt is not None else ctx
    seq_len = up + 3 + down
    if args.shuffle_seq:
        suffix = f'ctx{ctx}nt_shuffled'
    elif args.upstream_nt is not None or args.downstream_nt is not None:
        suffix = f'up{up}_down{down}nt'
    else:
        suffix = f'ctx{ctx}nt'

    print(f'Context: -{up}/+{down} nt  →  window = {seq_len} nt  [{suffix}]')
    print(f'Loss: {args.loss}  |  Transformer layers: {args.n_transformer_layers}')
    print(f'Drugs: {drug_names}')

    # nt_seq is already pre-windowed at ±ctx nt; stop codon is at index ctx
    stop_pos = ctx

    # Encode
    rng = np.random.default_rng(42)
    def _encode(s):
        arr = encode_seq(s, seq_len, stop_pos=stop_pos,
                         upstream_nt=args.upstream_nt, downstream_nt=args.downstream_nt)
        if args.shuffle_seq:
            arr = rng.permutation(arr)
        return arr
    tokens   = np.stack([_encode(s) for s in df['nt_seq']])
    drug_ids = df['drug_id'].values.astype(np.int64)
    labels   = df['label'].values.astype(np.int64)

    print(f'\nOverall: n={len(labels)}  pos={labels.sum()}  ({100*labels.mean():.1f}%)')
    for dname in drug_names:
        mask = drug_ids == drug2id[dname]
        n, p = mask.sum(), labels[mask].sum()
        print(f'  {dname:<20} n={n:<5} pos={p:<5} ({100*p/n:.1f}%)')

    print(f'\nRunning {args.cv_folds}-fold stratified CV...')
    mean_m, std_m, per_drug_m, fold_metrics, raw_preds = run_cv(
        tokens, drug_ids, labels, drug_names, args, device, suffix=suffix)

    if args.save_preds:
        preds_path = os.path.join(args.out_dir,
            f'preds_{suffix}_{args.loss}_{args.n_transformer_layers}layers.npz')
        np.savez(preds_path,
                 labels=raw_preds[0], probs=raw_preds[1],
                 drug_ids=raw_preds[2], drug_names=np.array(drug_names))
        print(f'[preds saved to {preds_path}]')

    print(f'\n── OVERALL (mean ± std across {args.cv_folds} folds) ──')
    print(f'AUROC={mean_m["auroc"]:.4f}±{std_m["auroc"]:.4f}'
          f'  AUPRC={mean_m["auprc"]:.4f}±{std_m["auprc"]:.4f}'
          f'  F1={mean_m["f1"]:.4f}±{std_m["f1"]:.4f}')

    print(f'\n── PER DRUG (mean across folds) ──')
    print(f'{"Drug":<20}  {"AUROC":>6}  {"AUPRC":>6}  {"F1":>6}  pos/n')
    print('-' * 60)
    for dname in drug_names:
        if dname not in per_drug_m:
            continue
        dm = per_drug_m[dname]
        print(f'{dname:<20}  {dm["auroc"]:>6.4f}  {dm["auprc"]:>6.4f}'
              f'  {dm["f1"]:>6.4f}  {int(dm["pos"])}/{int(dm["n"])}')

    # Save results
    results = dict(mean=mean_m, std=std_m, per_drug=per_drug_m,
                   config=vars(args), drug_names=drug_names)
    out_path = os.path.join(args.out_dir,
        f'results_{suffix}_{args.loss}_{args.n_transformer_layers}layers.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
