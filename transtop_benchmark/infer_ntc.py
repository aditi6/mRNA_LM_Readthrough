import torch, json, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, average_precision_score
from huggingface_hub import hf_hub_download

HF_MODEL  = "Dichopsis/TransStop"
DRUG_NAME = "G418"
CONTEXT_N = 6   # ±6 nt around stop codon (15 nt total)

class PanDrugTransformer(torch.nn.Module):
    def __init__(self, base_model, num_drugs, head_hidden_size=768,
                 drug_embed_dim=64, num_attention_heads=8, dropout_rate=0.17):
        super().__init__()
        self.base_model = base_model
        hidden = base_model.config.hidden_size
        self.drug_embedding   = torch.nn.Embedding(num_drugs, drug_embed_dim)
        self.query_projection = torch.nn.Linear(drug_embed_dim, hidden)
        self.cross_attention  = torch.nn.MultiheadAttention(hidden, num_attention_heads, batch_first=True)
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, head_hidden_size),
            torch.nn.ReLU(), torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(head_hidden_size, 1))
    def forward(self, input_ids, attention_mask, drug_id):
        seq  = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        q    = self.query_projection(self.drug_embedding(drug_id)).unsqueeze(1)
        out, _ = self.cross_attention(query=q, key=seq, value=seq)
        return self.reg_head(out.squeeze(1)).squeeze(-1)

class NTCDataset(Dataset):
    def __init__(self, seqs, tokenizer, drug_id):
        self.seqs = seqs; self.tokenizer = tokenizer; self.drug_id = drug_id
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.seqs[idx], return_tensors="pt", truncation=True)
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "drug_id": torch.tensor(self.drug_id, dtype=torch.long)}

def collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    return {"input_ids":      pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0),
            "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "drug_id":        torch.stack([b["drug_id"] for b in batch])}

def extract_window(cds, utr3, n=CONTEXT_N):
    pre  = cds[-(n+3):-3] if len(cds) >= n+3 else cds[:-3].rjust(n, "N")
    stop = cds[-3:]
    post = utr3[:n]       if len(utr3) >= n   else utr3.ljust(n, "N")
    return (pre + stop + post).upper().replace("U", "T")

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
drug_map  = json.load(open("/workspace/TransStop/results/drug_map.json"))
drug_id   = drug_map[DRUG_NAME]

base  = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
model = PanDrugTransformer(base.base_model, num_drugs=len(drug_map))
weights_path = hf_hub_download(repo_id=HF_MODEL, filename="model.safetensors")
missing, _ = model.load_state_dict(load_file(weights_path), strict=False)
if missing: print("Missing keys:", missing)

device = torch.device("cuda")
model  = model.to(device).eval()
print(f"Model loaded. Drug: {DRUG_NAME} (id={drug_id})")

# Load your NTC test split
df      = pd.read_csv("/workspace/mRNA_LM_Readthrough/data/readthrough_data.csv")
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"NTC test samples: {len(test_df)}  (pos={test_df['rrts'].sum()}, neg={(test_df['rrts']==0).sum()})")

test_df["nt_seq"] = test_df.apply(lambda r: extract_window(r["cds"], r["3utr"]), axis=1)
print(f"Sample window: {test_df['nt_seq'].iloc[0]}  (len={len(test_df['nt_seq'].iloc[0])})")

loader = DataLoader(NTCDataset(test_df["nt_seq"].tolist(), tokenizer, drug_id),
                    batch_size=128, collate_fn=collate, num_workers=2)

preds = []
with torch.no_grad():
    for batch in loader:
        out = model(batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["drug_id"].to(device))
        preds.extend(out.cpu().numpy())
        print(f"  {len(preds)}/{len(test_df)}", end="\r")

preds  = np.array(preds)
labels = test_df["rrts"].values.astype(int)

auroc = roc_auc_score(labels, preds)
auprc = average_precision_score(labels, preds)
print(f"\n\n=== TransStop on Wangen NTC G418 test set ===")
print(f"AUROC = {auroc:.4f}")
print(f"AUPRC = {auprc:.4f}")
print(f"(Baseline AUROC from prior best model: 0.6905)")

test_df["transtop_pred"] = preds
test_df[["cds","3utr","rrts","nt_seq","transtop_pred"]].to_csv(
    "/workspace/TransStop/ntc_predictions.csv", index=False)
print("Predictions saved to /workspace/TransStop/ntc_predictions.csv")
