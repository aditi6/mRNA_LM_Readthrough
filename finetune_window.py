"""
finetune_window.py — Stop-codon-windowed readthrough fine-tuning script.

This file is a copy of finetune_all.py with one key change:
  - Imports from dataload_window instead of dataload.

dataload_window.build_readthrough_dataset() trims each sequence to a
stop-codon-centred window (last 20 codons of CDS + first 200nt of 3'UTR)
before tokenisation. All other dataset builders are identical to dataload.py.

Rationale: the original finetune_all.py fed full CDS and 3'UTR sequences to
the model. Under mean pooling, the stop codon signal (the primary determinant
of readthrough) is diluted across hundreds of irrelevant tokens. Windowing
focuses the model exclusively on the biologically relevant region, eliminating
noise and ensuring no sequences are truncated by the tokeniser (max_length=1024).

See dataload_window.py for detailed documentation of the windowing approach.
"""

import os
import argparse
import numpy as np

from transformers import TrainingArguments, Trainer
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score

from FullModel import FullModel

# --- CHANGED: import from dataload_window instead of dataload ---
# dataload_window is identical to dataload except build_readthrough_dataset
# applies stop-codon-centred windowing before tokenisation.
from dataload_window import *
# ----------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "true"

######### Arguments Processing
parser = argparse.ArgumentParser(description='FullModel — windowed readthrough')

parser.add_argument('--task',   '-t', required=True, type=str, default="", help='task')
parser.add_argument('--output', '-o', required=True, type=str, default="", help='output dir')

parser.add_argument('--lorar',    type=int,   default=32,   help='Lora rank')
parser.add_argument('--lalpha',   type=int,   default=32,   help='Lora alpha')
parser.add_argument('--ldropout', type=float, default=0.5,  help='Lora dropout')
parser.add_argument('--lr',       type=float, default=1e-4, help='learning rate')

parser.add_argument('--head_dim',     type=int,   default=768, help='prediction head dimension')
parser.add_argument('--head_dropout', type=float, default=0.5, help='prediction head dropout')

parser.add_argument('--device', '-d', type=int, default=0,  help='device')
parser.add_argument('--batch',  '-b', type=int, default=64, help='batch size')

parser.add_argument('--useCLIP',     '-clip',  type=bool,  default=False, help='use CLIP')
parser.add_argument('--temperature', '-temp',  type=float, default=0.07,  help='temperature')
parser.add_argument('--coefficient', '-coeff', type=float, default=0.2,   help='coefficient')

args = parser.parse_args()

########### GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.device

######### Task
if args.task not in ["tr", "halflife", "5class", "liver", "readthrough"]:
    print("Wrong task!", args.task)
    exit(0)

num_labels = 1
class_weights = []
metric_for_best_model = ""
greater_is_better = True
if args.task == "5class":
    num_labels = 5
    class_weights = [0.97326057, 0.48056585, 1.24829396, 1.44412955, 2.51197183]
    metric_for_best_model = "auroc"
elif args.task == "readthrough":
    num_labels = 2
    # Class weights computed as n_samples / (n_classes * n_class_i) on the
    # training set (7187 samples: 2462 class-0, 4725 class-1).
    # This corrects for the ~2:1 class imbalance without artificially
    # inflating the minority class contribution.
    class_weights = [1.46, 0.76]
    metric_for_best_model = "auroc"
else:
    metric_for_best_model = "spearmanr"

########### loading pretrained model and downstream task model
model = FullModel(num_labels, class_weights,
                  args.lorar, args.lalpha, args.ldropout,
                  args.head_dim, args.head_dropout,
                  args.useCLIP, args.temperature, args.coefficient)

########### loading dataset and dataloader
# --- NOTE: build_readthrough_dataset() here comes from dataload_window,
# which applies stop-codon-centred windowing. All other builders are unchanged.
if args.task == "tr":
    ds_train, ds_valid, ds_test = build_dp_dataset()
elif args.task == "halflife":
    ds_train, ds_valid, ds_test = build_saluki_dataset(0)
elif args.task == "5class":
    ds_train, ds_valid, ds_test = build_class_dataset()
elif args.task == "liver":
    ds_train, ds_valid, ds_test = build_liver_dataset()
elif args.task == "readthrough":
    ds_train, ds_valid, ds_test = build_readthrough_dataset()

train_loader = ds_train.map(model.encode_string, batched=True)
val_loader   = ds_valid.map(model.encode_string, batched=True)
test_loader  = ds_test.map(model.encode_string,  batched=True)

######### Training Settings & Metrics
training_args = TrainingArguments(
    optim='adamw_torch',
    learning_rate=args.lr,
    output_dir=args.output,
    eval_strategy="epoch",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=1,
    save_strategy="epoch",
    save_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    save_total_limit=3,
    eval_steps=1,
    logging_steps=50,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="none",
    save_safetensors=False,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if num_labels == 1:
        logits = logits.flatten()
        labels = labels.flatten()
        try:
            pearson_corr  = pearsonr(logits, labels)[0].item()
            spearman_corr = spearmanr(logits, labels)[0].item()
            return {"pearson": pearson_corr, "spearmanr": spearman_corr}
        except:
            return {"pearson": 0.0, "spearmanr": 0.0}
    else:
        predictions = np.argmax(logits, axis=-1)
        probs = softmax(logits, axis=1)
        # Use positive-class probability for AUROC (binary classification)
        positive_class_probs = probs[:, 1]
        f1    = f1_score(labels, predictions, average="macro")
        auroc = roc_auc_score(labels, positive_class_probs)
        return {"f1": f1, "auroc": auroc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader,
    eval_dataset=val_loader,
    compute_metrics=compute_metrics
)

######### Training & Evaluation & Prediction
trainer.train()

metrics = trainer.evaluate()
print(metrics)

pred, _, metrics = trainer.predict(test_loader)
print(metrics)
