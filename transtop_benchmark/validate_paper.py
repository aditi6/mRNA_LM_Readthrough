import torch, json, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from safetensors.torch import load_file
from sklearn.metrics import r2_score, mean_absolute_error
from huggingface_hub import hf_hub_download

HF_MODEL = "Dichopsis/TransStop"

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

class PTCDataset(Dataset):
    def __init__(self, df, tokenizer, drug_map):
        self.df = df; self.tokenizer = tokenizer; self.drug_map = drug_map
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(row["seq_context_12"].replace("U","T"), return_tensors="pt", truncation=True)
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "drug_id": torch.tensor(self.drug_map[row["drug"]], dtype=torch.long),
                "label": torch.tensor(float(row["RT_transformed"]), dtype=torch.float)}

def collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    return {"input_ids":      pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0),
            "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "drug_id":        torch.stack([b["drug_id"] for b in batch]),
            "label":          torch.stack([b["label"] for b in batch])}

print("Loading tokenizer + base model...")
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
drug_map  = json.load(open("/workspace/TransStop/results/drug_map.json"))

base  = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True)
model = PanDrugTransformer(base.base_model, num_drugs=len(drug_map))

print("Downloading pretrained TransStop weights from HuggingFace...")
weights_path = hf_hub_download(repo_id=HF_MODEL, filename="model.safetensors")
missing, unexpected = model.load_state_dict(load_file(weights_path), strict=False)
print(f"Weights loaded. Missing={len(missing)}, Unexpected={len(unexpected)}")
if missing: print("  Missing:", missing[:5])

device = torch.device("cuda")
model  = model.to(device).eval()

# Test set = held-out 10% never seen during training (train+val were combined for final model)
test_df = pd.read_csv("/workspace/TransStop/processed_data/test_df.csv")
print(f"\nTest set: {len(test_df)} rows")

loader = DataLoader(PTCDataset(test_df, tokenizer, drug_map),
                    batch_size=128, collate_fn=collate, num_workers=2)

preds, labels = [], []
with torch.no_grad():
    for batch in loader:
        out = model(batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["drug_id"].to(device))
        preds.extend(out.cpu().numpy())
        labels.extend(batch["label"].numpy())
        print(f"  {len(preds)}/{len(test_df)}", end="\r")

preds  = np.expm1(np.array(preds));  preds[preds < 0] = 0
labels = np.expm1(np.array(labels))
test_df["pred"]  = preds
test_df["label"] = labels

overall_r2  = r2_score(labels, preds)
overall_mae = mean_absolute_error(labels, preds)
print(f"\n\n=== Paper reproduction on PTC held-out test set ===")
print(f"Overall R² = {overall_r2:.4f}   (paper: 0.94)")
print(f"Overall MAE = {overall_mae:.4f}")
print()

paper_r2 = {"G418":0.86,"Gentamicin":0.77,"SRI":0.92,"SJ6986":0.91,
            "DAP":0.92,"Clitocine":0.90,"CC90009":0.79,"FUr":"N/A","Untreated":"N/A"}
print(f"{'Drug':<14} {'Our R²':>8}  {'Paper R²':>10}  {'n':>5}")
print("-" * 42)
for drug, grp in test_df.groupby("drug"):
    r2d = r2_score(grp["label"], grp["pred"])
    pr  = paper_r2.get(drug, "N/A")
    match = "✓" if pr != "N/A" and abs(r2d - pr) < 0.05 else ("~" if pr != "N/A" and abs(r2d - pr) < 0.10 else "")
    print(f"  {drug:<12} {r2d:>8.3f}  {str(pr):>10}  {len(grp):>5}  {match}")
