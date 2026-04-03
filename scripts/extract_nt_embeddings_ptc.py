"""
Extract frozen Nucleotide Transformer embeddings for the PTC Toledano dataset.

The CSV contains a pre-windowed nt_seq (~246 nt around the stop codon) and
per-drug readthrough efficiency values. Embeddings are extracted once; labels
are saved per drug.

Censored values (">3.2") are capped at --cap_value (default 3.2). Use
--drop_censored to exclude those rows entirely.

Run:
    python extract_nt_embeddings_ptc.py \
        --data "/workspace/PTC Toledano.csv" \
        --out_dir /workspace/embeddings_ptc_nt \
        --batch 32 --device 0

    # Drop rows with censored values instead of capping:
    python extract_nt_embeddings_ptc.py ... --drop_censored
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

DRUG_COLS = ['FUr', 'Gentamicin', 'CC90009', 'G418', 'Clitocine', 'DAP',
             'SJ6986', 'SRI', 'Untreated']


def parse_drug_value(val, cap_value):
    """Convert a readthrough value to float; handle censored '>X' strings."""
    s = str(val).strip()
    if s.startswith('>'):
        return cap_value
    try:
        return float(s)
    except ValueError:
        return float('nan')


def load_ptc_data(csv_path, cap_value, drop_censored):
    df = pd.read_csv(csv_path)
    # Drop unnamed index columns that appear as empty column names
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
    df = df.dropna(subset=['nt_seq'])

    sequences = df['nt_seq'].str.upper().str.replace('U', 'T', regex=False).tolist()
    meta = df[['mutation_identifier', 'GENEINFO', 'stop_type']].reset_index(drop=True)

    labels = {}
    for drug in DRUG_COLS:
        if drug not in df.columns:
            continue
        vals = df[drug].apply(lambda v: parse_drug_value(v, cap_value))
        if drop_censored:
            # Mark rows with '>X' as NaN so they can be excluded per-drug
            censored_mask = df[drug].astype(str).str.startswith('>')
            vals[censored_mask] = float('nan')
        labels[drug] = vals.values.astype(float)

    lens = [len(s) for s in sequences]
    print(f'Loaded {len(sequences)} sequences — '
          f'min len: {min(lens)}, median: {int(np.median(lens))}, max: {max(lens)}')
    for drug, y in labels.items():
        n_valid = (~np.isnan(y)).sum()
        n_censored = (df[drug].astype(str).str.startswith('>') if drug in df.columns
                      else pd.Series(False, index=df.index)).sum()
        print(f'  {drug:12s}  n={n_valid}  censored={n_censored}  '
              f'mean={np.nanmean(y):.3f}  std={np.nanstd(y):.3f}')

    return sequences, labels, meta


class SeqDataset(TorchDataset):
    def __init__(self, seqs, tokenizer, max_length):
        self.seqs = seqs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

    def collate(self, batch):
        enc = self.tokenizer(
            list(batch),
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
        )
        return enc


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    all_emb = []

    for enc in loader:
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)

        # CLS token from last hidden state
        cls_emb = out.hidden_states[-1][:, 0, :]  # (B, hidden_size)
        all_emb.append(cls_emb.cpu().float().numpy())
        print(f'  processed {sum(len(x) for x in all_emb)} sequences...', end='\r')

    print()
    return np.concatenate(all_emb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        default='/workspace/PTC Toledano.csv')
    parser.add_argument('--model',       default=MODEL_NAME)
    parser.add_argument('--out_dir',     default='/workspace/embeddings_ptc_nt')
    parser.add_argument('--batch',       type=int, default=32)
    parser.add_argument('--device',      type=int, default=0)
    parser.add_argument('--max_length',  type=int, default=128,
                        help='max tokens (NT uses 6-mer tokens; 128 tokens = 768 nt)')
    parser.add_argument('--cap_value',   type=float, default=3.2,
                        help='replace ">X" censored values with this float')
    parser.add_argument('--drop_censored', action='store_true',
                        help='set censored values to NaN instead of capping')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading data from {args.data}...')
    sequences, labels, meta = load_ptc_data(args.data, args.cap_value, args.drop_censored)

    print(f'\nLoading {args.model}...')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True,
                                                  output_hidden_states=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hidden_size = model.config.hidden_size
    print(f'Model hidden size: {hidden_size}')
    print(f'Max tokens: {args.max_length} (~{args.max_length * 6} nt)')

    ds = SeqDataset(sequences, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        collate_fn=ds.collate, num_workers=0)

    print('\nExtracting embeddings...')
    embeddings = extract(model, loader, device)
    print(f'Embeddings shape: {embeddings.shape}')

    # Save embeddings and per-drug labels
    np.save(os.path.join(args.out_dir, 'embeddings.npy'), embeddings)
    for drug, y in labels.items():
        np.save(os.path.join(args.out_dir, f'labels_{drug}.npy'), y)

    # Save metadata for reference
    meta.to_csv(os.path.join(args.out_dir, 'metadata.csv'), index=False)

    print(f'\nDone. Saved to {args.out_dir}/')
    print(f'  embeddings.npy        {embeddings.shape}')
    print(f'  labels_<drug>.npy     one file per drug: {list(labels.keys())}')
    print(f'  metadata.csv          mutation_identifier, gene, stop_type')


if __name__ == '__main__':
    main()
