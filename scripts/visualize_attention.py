"""
Attention visualization for ReadthroughClassifier and ReadthroughModelV2.

Loads a saved checkpoint, runs inference on the dataset with return_attn=True,
then plots mean cross-attention weight per sequence position, broken down by drug.

The attention comes from the query pool's cross-attention layer — the single vector
that decides how much each sequence position contributes to the final prediction.
For the classifier this query is drug-conditioned, so each drug gets its own map.

Usage:
  # Classifier (drug-specific attention maps):
  python visualize_attention.py \\
      --model classifier \\
      --ckpt results_classifier/checkpoint_fold1.pt \\
      --data merged_treatments.csv \\
      --out_dir attention_plots

  # Regression v2 (single attention map across all sequences):
  python visualize_attention.py \\
      --model regression \\
      --ckpt results_v2/checkpoint_DAP_fold1.pt \\
      --data "NTC Toledano.csv" \\
      --drug DAP \\
      --context_nt 45 \\
      --out_dir attention_plots
"""

import argparse, os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── import model classes ───────────────────────────────────────────────────────
from train_classifier_treatments import (
    ReadthroughClassifier, encode_seq, make_pos_array)
from train_cnn_transformer_ntc_v2 import (
    ReadthroughModelV2, extract_window, pad_or_trim, DRUGS)

NT2IDX = {c: i for i, c in enumerate('acgtn')}
NT2IDX.update({'u': 3, 'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4})


# ── helpers ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_attention_classifier(model, tokens, positions, drug_ids, batch_size=256, device='cpu'):
    """Returns mean attention per drug: dict {drug_id: np.array (L,)}"""
    model.eval()
    all_attn  = []
    all_drugs = []
    for i in range(0, len(tokens), batch_size):
        t  = torch.from_numpy(tokens[i:i+batch_size]).to(device)
        p  = torch.from_numpy(positions[i:i+batch_size]).to(device)
        d  = torch.from_numpy(drug_ids[i:i+batch_size]).to(device)
        _, attn = model(t, p, d, return_attn=True)   # attn: (B, L)
        all_attn.append(attn.cpu().numpy())
        all_drugs.append(drug_ids[i:i+batch_size])
    all_attn  = np.concatenate(all_attn)   # (N, L)
    all_drugs = np.concatenate(all_drugs)  # (N,)

    drug_mean = {}
    for did in np.unique(all_drugs):
        drug_mean[int(did)] = all_attn[all_drugs == did].mean(axis=0)
    return drug_mean


@torch.no_grad()
def extract_attention_regression(model, tokens, positions, batch_size=256, device='cpu'):
    """Returns mean attention across all sequences: np.array (L,)"""
    model.eval()
    all_attn = []
    for i in range(0, len(tokens), batch_size):
        t = torch.from_numpy(tokens[i:i+batch_size]).to(device)
        p = torch.from_numpy(positions[i:i+batch_size]).to(device)
        _, attn = model(t, p, return_attn=True)   # attn: (B, L)
        all_attn.append(attn.cpu().numpy())
    return np.concatenate(all_attn).mean(axis=0)   # (L,)


# ── plotting ───────────────────────────────────────────────────────────────────
def plot_classifier_attention(drug_mean, drug_names, positions, out_path):
    """Heatmap: drugs (rows) × sequence positions (cols)."""
    n_drugs = len(drug_names)
    mat = np.stack([drug_mean[i] for i in range(n_drugs)])  # (n_drugs, L)

    fig, axes = plt.subplots(n_drugs, 1, figsize=(14, 1.6 * n_drugs + 1.5),
                              sharex=True)
    if n_drugs == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    stop_positions = [p for p in positions if 0 <= p <= 2]
    stop_idx_start = np.where(positions == 0)[0][0] if 0 in positions else None

    for i, (ax, dname) in enumerate(zip(axes, drug_names)):
        attn = mat[i]
        ax.fill_between(positions, attn, alpha=0.7, color=colors[i % 10])
        ax.plot(positions, attn, color=colors[i % 10], lw=1.2)

        # shade stop codon region
        if stop_idx_start is not None:
            ax.axvspan(-0.5, 2.5, alpha=0.12, color='red', zorder=0)

        ax.axvline(0, color='red', lw=0.8, ls='--', alpha=0.6)
        ax.set_ylabel(dname, fontsize=9, rotation=0, ha='right', va='center')
        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axes[-1].set_xlabel('Position relative to stop codon (nt)', fontsize=10)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(5))

    fig.suptitle('Cross-attention weight by sequence position\n'
                 '(drug-conditioned query pool, averaged over all sequences)',
                 fontsize=11, y=1.01)

    # Add region labels
    axes[0].text(-0.5, axes[0].get_ylim()[1] * 0.85, '← CDS',
                 fontsize=8, color='#555555', ha='right')
    axes[0].text(3.5, axes[0].get_ylim()[1] * 0.85, "3'UTR →",
                 fontsize=8, color='#555555', ha='left')
    axes[0].text(1.0, axes[0].get_ylim()[1] * 0.85, 'stop',
                 fontsize=7, color='red', ha='center')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def plot_regression_attention(mean_attn, positions, drug, context_nt, out_path):
    """Line plot of attention vs position for regression model."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(positions, mean_attn, alpha=0.5, color='steelblue')
    ax.plot(positions, mean_attn, color='steelblue', lw=1.5)
    ax.axvspan(-0.5, 2.5, alpha=0.15, color='red', zorder=0)
    ax.axvline(0, color='red', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Position relative to stop codon (nt)', fontsize=10)
    ax.set_ylabel('Mean attention weight', fontsize=10)
    ax.set_title(f'Cross-attention weight — {drug}  (±{context_nt} nt context)', fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


def plot_classifier_heatmap(drug_mean, drug_names, positions, out_path):
    """2D heatmap: drugs × positions as a grid."""
    mat = np.stack([drug_mean[i] for i in range(len(drug_names))])
    # Normalise each drug's attention to [0, 1] for comparability
    mat_norm = (mat - mat.min(axis=1, keepdims=True)) / \
               (mat.max(axis=1, keepdims=True) - mat.min(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(14, max(3, len(drug_names) * 0.6 + 1)))
    im = ax.imshow(mat_norm, aspect='auto', cmap='YlOrRd',
                   extent=[positions[0] - 0.5, positions[-1] + 0.5,
                           len(drug_names) - 0.5, -0.5])

    # Stop codon overlay
    ax.axvspan(-0.5, 2.5, alpha=0.15, color='blue', zorder=2)
    ax.axvline(0, color='blue', lw=0.8, ls='--', alpha=0.5)

    ax.set_yticks(range(len(drug_names)))
    ax.set_yticklabels(drug_names, fontsize=9)
    ax.set_xlabel('Position relative to stop codon (nt)', fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Norm. attention\n(per drug)', fontsize=8)

    ax.set_title('Drug-conditioned cross-attention heatmap\n'
                 '(row-normalised; red = stop codon region)', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True, choices=['classifier', 'regression'],
                        help='which model type the checkpoint is from')
    parser.add_argument('--ckpt',       required=True, help='path to .pt checkpoint')
    parser.add_argument('--data',       required=True, help='CSV data file')
    parser.add_argument('--out_dir',    default='attention_plots')
    parser.add_argument('--context_nt', type=int, default=45,
                        help='context window used during training')
    parser.add_argument('--drug',       default=None,
                        help='(regression only) which drug column to use')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    df   = pd.read_csv(args.data)

    if args.model == 'classifier':
        saved_args  = ckpt.get('args', {})
        drug_names  = list(ckpt['drug_names'])
        drug2id     = {d: i for i, d in enumerate(drug_names)}
        ctx         = saved_args.get('context_nt', args.context_nt)
        seq_len     = 2 * ctx + 3

        model = ReadthroughClassifier(
            n_drugs=len(drug_names), seq_len=seq_len, context_nt=ctx,
            dropout=saved_args.get('dropout', 0.3),
            attn_window=saved_args.get('attn_window', 7),
            n_transformer_layers=saved_args.get('n_transformer_layers', 2),
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded classifier checkpoint (val AUPRC={ckpt.get("val_auprc", "?"):.4f})')

        # Encode all sequences
        df['drug_id'] = df['drug'].map(drug2id)
        tokens   = np.stack([encode_seq(s, seq_len) for s in df['nt_seq']])
        drug_ids = df['drug_id'].values.astype(np.int64)
        positions = np.tile(make_pos_array(ctx).astype(np.int64), (len(tokens), 1))

        drug_mean = extract_attention_classifier(
            model, tokens, positions, drug_ids,
            batch_size=args.batch_size, device=device)

        pos_vec = make_pos_array(ctx)

        # Line plot (one panel per drug)
        plot_classifier_attention(
            drug_mean, drug_names, pos_vec,
            os.path.join(args.out_dir, 'attention_by_drug_lines.png'))

        # Heatmap
        plot_classifier_heatmap(
            drug_mean, drug_names, pos_vec,
            os.path.join(args.out_dir, 'attention_by_drug_heatmap.png'))

        # Save raw arrays
        np.savez(os.path.join(args.out_dir, 'attention_arrays.npz'),
                 drug_names=np.array(drug_names),
                 positions=pos_vec,
                 **{f'attn_{drug_names[i]}': drug_mean[i]
                    for i in range(len(drug_names))})
        print('Saved raw attention arrays to attention_arrays.npz')

    else:  # regression
        ctx      = args.context_nt
        seq_len  = 2 * ctx + 3
        drug     = args.drug or 'DAP'

        model = ReadthroughModelV2(seq_len=seq_len, context_nt=ctx).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded regression checkpoint (val R²={ckpt.get("val_r2", "?"):.4f})')

        tokens_list, pos_list = [], []
        for seq in df['nt_seq']:
            tok, pos = extract_window(seq, context_nt=ctx)
            tokens_list.append(pad_or_trim(tok, seq_len, pad_val=4))
            pos_list.append(pad_or_trim(pos, seq_len, pad_val=0))
        tokens    = np.stack(tokens_list)
        positions = np.stack(pos_list)

        mean_attn = extract_attention_regression(
            model, tokens, positions, batch_size=args.batch_size, device=device)

        pos_vec = np.arange(-ctx, ctx + 3, dtype=np.int64)
        plot_regression_attention(
            mean_attn, pos_vec, drug, ctx,
            os.path.join(args.out_dir, f'attention_{drug}_ctx{ctx}nt.png'))

        np.savez(os.path.join(args.out_dir, f'attention_{drug}_ctx{ctx}nt.npz'),
                 positions=pos_vec, mean_attn=mean_attn)
        print('Saved raw attention arrays.')


if __name__ == '__main__':
    main()
