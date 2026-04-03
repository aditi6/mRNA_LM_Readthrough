"""
Extract frozen mRNA-LM embeddings (CodonBERT + 3'UTRBERT) for NTC Toledano dataset.

The nt_seq is split at the stop codon (located via stop_type/up_123nt/down_123nt):
  - CDS tail  → CodonBERT  (codon tokenized, mean pool) → 768d
  - 3'UTR head → 3'UTRBERT (nt tokenized,    mean pool) → 768d
  - Concatenated → 1536d embedding per sequence

Run:
    python extract_mrnaLM_embeddings_ntc.py \
        --data "/workspace/NTC Toledano.csv" \
        --cds_model /workspace/codonbert \
        --utr_model /workspace/mrna_3utr_model_p2_cp99900_best \
        --out_dir /workspace/embeddings_ntc_mrnaLM \
        --batch 32 --device 0
"""

import argparse, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import BertForMaskedLM, BertTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer

DRUG_COLS = ['Clitocine', 'DAP', 'G418', 'SJ6986', 'SRI']


# ── Tokenisers (identical to extract_embeddings.py) ───────────────────────────

def build_tokenizers():
    lst_ele = list('AUGC')

    # CDS: codon (3-mer) tokenizer
    lst_voc_cds = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    for a1 in lst_ele:
        for a2 in lst_ele:
            for a3 in lst_ele:
                lst_voc_cds.append(f'{a1}{a2}{a3}')
    dic_voc_cds = {t: i for i, t in enumerate(lst_voc_cds)}

    tok_cds_obj = Tokenizer(WordLevel(vocab=dic_voc_cds, unk_token='[UNK]'))
    tok_cds_obj.normalizer = BertNormalizer(lowercase=False, strip_accents=False)
    tok_cds_obj.pre_tokenizer = Whitespace()
    tok_cds_obj.post_processor = BertProcessing(
        ('[SEP]', dic_voc_cds['[SEP]']), ('[CLS]', dic_voc_cds['[CLS]'])
    )
    tokenizer_cds = BertTokenizerFast(
        tokenizer_object=tok_cds_obj,
        unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]',
        cls_token='[CLS]', mask_token='[MASK]'
    )

    # 3'UTR: single-nucleotide tokenizer
    lst_voc_utr = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list('AUGC')
    dic_voc_utr = {t: i for i, t in enumerate(lst_voc_utr)}

    tok_utr_obj = Tokenizer(WordLevel(vocab=dic_voc_utr, unk_token='[UNK]'))
    tok_utr_obj.normalizer = BertNormalizer(lowercase=False, strip_accents=False)
    tok_utr_obj.pre_tokenizer = Whitespace()
    tok_utr_obj.post_processor = BertProcessing(
        ('[SEP]', dic_voc_utr['[SEP]']), ('[CLS]', dic_voc_utr['[CLS]'])
    )
    tokenizer_utr = BertTokenizerFast(
        tokenizer_object=tok_utr_obj,
        unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]',
        cls_token='[CLS]', mask_token='[MASK]'
    )
    return tokenizer_cds, tokenizer_utr


def mytok(seq, kmer_len, s):
    seq = seq.upper().replace('T', 'U')
    return [seq[j:j + kmer_len] for j in range(0, (len(seq) - kmer_len) + 1, s)]


def mean_pool(hidden, mask):
    """Mean pool excluding [CLS] and [SEP]; mask handles padding."""
    h = hidden[:, 1:-1, :]
    m = mask[:, 1:-1].unsqueeze(-1).float()
    denom = m.sum(1).clamp(min=1e-9)
    return (h * m).sum(1) / denom


# ── Stop codon splitting ───────────────────────────────────────────────────────

def find_stop_pos(seq, stop_type, up_3, down_3):
    """Return index of first nucleotide of stop codon in seq (DNA, lowercase)."""
    seq = seq.lower()
    stop_dna = stop_type.lower().replace('u', 't')
    up_dna   = up_3.lower().replace('u', 't')
    dn_dna   = down_3.lower().replace('u', 't')
    target   = up_dna + stop_dna + dn_dna
    idx = seq.find(target)
    if idx == -1:
        return None
    return idx + 3   # start of stop codon


def split_at_stop(row):
    """Return (cds_tail, utr_head) as DNA strings; None pair on failure."""
    if pd.isna(row['up_123nt']) or pd.isna(row['down_123nt']):
        return None, None
    pos = find_stop_pos(row['nt_seq'], row['stop_type'], row['up_123nt'], row['down_123nt'])
    if pos is None:
        return None, None
    cds = row['nt_seq'][:pos]
    utr = row['nt_seq'][pos + 3:]
    return cds, utr


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeqDataset(TorchDataset):
    def __init__(self, cds_list, utr_list, tokenizer_cds, tokenizer_utr):
        self.cds     = cds_list
        self.utr     = utr_list
        self.tok_cds = tokenizer_cds
        self.tok_utr = tokenizer_utr

    def __len__(self):
        return len(self.cds)

    def __getitem__(self, idx):
        return self.cds[idx], self.utr[idx]

    def collate(self, batch):
        cds_seqs, utr_seqs = zip(*batch)
        self.tok_cds.truncation_side = 'left'
        enc_cds = self.tok_cds(
            list(cds_seqs), truncation=True, padding='max_length',
            max_length=1024, return_tensors='pt'
        )
        self.tok_utr.truncation_side = 'right'
        enc_utr = self.tok_utr(
            list(utr_seqs), truncation=True, padding='max_length',
            max_length=1024, return_tensors='pt'
        )
        return enc_cds, enc_utr


@torch.no_grad()
def extract(model_cds, model_utr, loader, device):
    model_cds.eval()
    model_utr.eval()
    all_emb = []

    for enc_cds, enc_utr in loader:
        ids2  = enc_cds['input_ids'].to(device)
        mask2 = enc_cds['attention_mask'].to(device)
        ids3  = enc_utr['input_ids'].to(device)
        mask3 = enc_utr['attention_mask'].to(device)

        h_cds = model_cds(input_ids=ids2, attention_mask=mask2,
                           output_hidden_states=True)['hidden_states'][-1]
        h_utr = model_utr(input_ids=ids3, attention_mask=mask3,
                           output_hidden_states=True)['hidden_states'][-1]

        emb_cds = mean_pool(h_cds, mask2)   # (B, 768)
        emb_utr = mean_pool(h_utr, mask3)   # (B, 768)
        joint   = torch.cat([emb_cds, emb_utr], dim=1)  # (B, 1536)

        all_emb.append(joint.cpu().float().numpy())
        print(f'  processed {sum(len(x) for x in all_emb)} sequences...', end='\r', flush=True)

    print()
    return np.concatenate(all_emb)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      default='/workspace/NTC Toledano.csv')
    parser.add_argument('--cds_model', default='/workspace/codonbert')
    parser.add_argument('--utr_model', default='/workspace/mrna_3utr_model_p2_cp99900_best')
    parser.add_argument('--out_dir',   default='/workspace/embeddings_ntc_mrnaLM')
    parser.add_argument('--batch',     type=int, default=32)
    parser.add_argument('--device',    type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading {args.data}...')
    df = pd.read_csv(args.data)
    print(f'  {len(df)} sequences, drugs: {DRUG_COLS}')

    # Split sequences at stop codon
    print('Splitting sequences at stop codon...')
    splits = df.apply(split_at_stop, axis=1)
    cds_list = [x[0] for x in splits]
    utr_list = [x[1] for x in splits]

    failed = sum(c is None for c in cds_list)
    if failed:
        print(f'  WARNING: {failed} sequences where stop codon not found — will use empty string')
    cds_list = [c if c is not None else '' for c in cds_list]
    utr_list = [u if u is not None else '' for u in utr_list]

    # Tokenize
    cds_tok = [' '.join(mytok(s, 3, 3)) if s else '[UNK]' for s in cds_list]
    utr_tok = [' '.join(mytok(s, 1, 1)) if s else '[UNK]' for s in utr_list]

    lens_cds = [len(s) for s in cds_list]
    lens_utr = [len(s) for s in utr_list]
    print(f'  CDS len — min:{min(lens_cds)} median:{int(np.median(lens_cds))} max:{max(lens_cds)}')
    print(f'  UTR len — min:{min(lens_utr)} median:{int(np.median(lens_utr))} max:{max(lens_utr)}')

    # Labels
    for drug in DRUG_COLS:
        y = pd.to_numeric(df[drug], errors='coerce').values.astype(float)
        np.save(os.path.join(args.out_dir, f'labels_{drug}.npy'), y)
        n_valid = (~np.isnan(y)).sum()
        print(f'  {drug}: n={n_valid}  mean={np.nanmean(y):.3f}  std={np.nanstd(y):.3f}')

    # Build tokenizers and models
    print('\nBuilding tokenizers...')
    tok_cds, tok_utr = build_tokenizers()

    print('Loading CodonBERT...')
    model_cds = BertForMaskedLM.from_pretrained(args.cds_model).to(device).eval()
    for p in model_cds.parameters(): p.requires_grad_(False)

    print('Loading 3\'UTRBERT...')
    model_utr = BertForMaskedLM.from_pretrained(args.utr_model).to(device).eval()
    for p in model_utr.parameters(): p.requires_grad_(False)

    ds     = SeqDataset(cds_tok, utr_tok, tok_cds, tok_utr)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        collate_fn=ds.collate, num_workers=0)

    print('\nExtracting embeddings...')
    embeddings = extract(model_cds, model_utr, loader, device)
    print(f'Embeddings shape: {embeddings.shape}')

    np.save(os.path.join(args.out_dir, 'embeddings.npy'), embeddings)
    df[['GENEINFO', 'stop_type']].to_csv(os.path.join(args.out_dir, 'metadata.csv'), index=False)

    print(f'\nDone. Saved to {args.out_dir}/')
    print(f'  embeddings.npy  {embeddings.shape}  (CDS 768d + UTR 768d mean pool)')
    print(f'  labels_<drug>.npy  drugs: {DRUG_COLS}')


if __name__ == '__main__':
    main()
