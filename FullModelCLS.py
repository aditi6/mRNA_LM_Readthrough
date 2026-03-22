"""
FullModelCLS.py — Copy of FullModel.py with one key change: CLS token pooling.

The only difference from FullModel.py is in forward():

  FullModel.py  (mean pooling):
      cds_embeds[:, 1:-1, :]  → average over all non-special tokens
      utr3_embeds[:, 1:-1, :] → average over all non-special tokens

  FullModelCLS.py (CLS pooling):
      cds_embeds[:, 0, :]   → take only the [CLS] token at index 0
      utr3_embeds[:, 0, :]  → take only the [CLS] token at index 0

Rationale: mean pooling treats every token equally, so the stop codon signal
(3nt) gets averaged against hundreds of irrelevant CDS tokens. BERT's [CLS]
token is designed as a single sequence-level summary — the entire self-attention
stack can route information from any position (including the stop codon) into
it. For a sequence-level classification task like readthrough, [CLS] pooling
lets the model learn to concentrate relevant signal rather than forcing a
uniform average.

get_mean_token_embeddings() is kept for compatibility but is no longer called
in forward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu
from torch.nn.functional import softmax, log_softmax

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer
from transformers import BertTokenizerFast

from transformers import BertForMaskedLM, BertTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model

class FullModel(torch.nn.Module):
    def __init__(self, num_labels, class_weights, lorar, lalpha, ldropout, head_dim=768, head_droupout=0.5, useCLIP=False, temperature=0.07, clip_coeff=0.2):
        super(FullModel, self).__init__()

        self.tokenizer_cds = None
        self.tokenizer_3utr = None
        self.build_tokenizer()
        self.CLIP = useCLIP

        print("Loading models for CDS and 3'UTR...")
        self.utr3 = BertForMaskedLM.from_pretrained("/workspace/mrna_3utr_model_p2_cp99900_best")
        self.cds = BertForMaskedLM.from_pretrained("/workspace/codonbert")

        if lorar > 0:
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=lorar,
                lora_alpha=lalpha,
                lora_dropout=ldropout,
                use_rslora=True
            )
            self.utr3 = get_peft_model(self.utr3, peft_config)
            self.utr3.print_trainable_parameters()
            self.cds = get_peft_model(self.cds, peft_config)
            self.cds.print_trainable_parameters()

        self.dense_cds1 = nn.Linear(768, 768)
        self.dense_cds2 = nn.Linear(768, 768)
        self.dense_utr3 = nn.Linear(768, 768)

        self.final_dense = nn.Linear(768 * 2, head_dim)

        self.transform_act_fn = gelu
        self.LayerNorm = torch.nn.LayerNorm(head_dim, eps=1e-12)
        self.dropout = nn.Dropout(head_droupout)

        self.decoder = nn.Linear(head_dim, num_labels, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self.decoder.bias = self.bias

        if num_labels == 1:
            self.loss_fn = nn.MSELoss()
        else:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, reduction='mean')

        self.temperature = temperature
        self.clip_coeff = clip_coeff
        self.is_first_epoch = True

    def build_tokenizer(self):
        lst_ele = list('AUGC')

        lst_voc_cds = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for a1 in lst_ele:
            for a2 in lst_ele:
                for a3 in lst_ele:
                    lst_voc_cds.extend([f'{a1}{a2}{a3}'])
        dic_voc_cds = {token: i for i, token in enumerate(lst_voc_cds)}

        tokenizer_cds_obj = Tokenizer(WordLevel(vocab=dic_voc_cds, unk_token="[UNK]"))
        tokenizer_cds_obj.normalizer = BertNormalizer(lowercase=False, strip_accents=False)
        tokenizer_cds_obj.pre_tokenizer = Whitespace()
        tokenizer_cds_obj.post_processor = BertProcessing(
            ("[SEP]", dic_voc_cds['[SEP]']),
            ("[CLS]", dic_voc_cds['[CLS]']),
        )
        self.tokenizer_cds = BertTokenizerFast(
            tokenizer_object=tokenizer_cds_obj,
            unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]'
        )

        lst_voc_utr = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for a1 in lst_ele:
            lst_voc_utr.extend([f'{a1}'])
        dic_voc_utr = {token: i for i, token in enumerate(lst_voc_utr)}

        tokenizer_3utr_obj = Tokenizer(WordLevel(vocab=dic_voc_utr, unk_token="[UNK]"))
        tokenizer_3utr_obj.normalizer = BertNormalizer(lowercase=False, strip_accents=False)
        tokenizer_3utr_obj.pre_tokenizer = Whitespace()
        tokenizer_3utr_obj.post_processor = BertProcessing(
            ("[SEP]", dic_voc_utr['[SEP]']),
            ("[CLS]", dic_voc_utr['[CLS]']),
        )
        self.tokenizer_3utr = BertTokenizerFast(
            tokenizer_object=tokenizer_3utr_obj,
            unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]'
        )

    def get_mean_token_embeddings(self, token_embeddings, token_mask):
        # Kept for compatibility; not used in this CLS-pooling variant.
        input_mask_expanded = token_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / sum_mask
        return sum_embeddings

    def forward(self, input_ids2, attention_mask2, input_ids3, attention_mask3, labels, **kwargs):
        cds_embeds  = self.cds(input_ids=input_ids2, attention_mask=attention_mask2, output_hidden_states=True)["hidden_states"][-1]
        utr3_embeds = self.utr3(input_ids=input_ids3, attention_mask=attention_mask3, output_hidden_states=True)["hidden_states"][-1]

        # --- CHANGED: CLS token pooling instead of mean pooling ---
        # Index 0 is always the [CLS] token (added by BertProcessing post-processor).
        # BERT's self-attention allows the [CLS] token to attend to every other
        # token, so after fine-tuning it aggregates task-relevant information
        # (e.g. stop codon identity) without the dilution effect of mean pooling.
        cds_sum_embeddings  = cds_embeds[:, 0, :]   # [CLS] token for CDS
        utr3_sum_embeddings = utr3_embeds[:, 0, :]  # [CLS] token for 3'UTR
        # ----------------------------------------------------------

        joint_embed = torch.cat([cds_sum_embeddings, utr3_sum_embeddings], dim=1)

        hidden_states = self.final_dense(joint_embed)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.decoder(hidden_states)

        if self.decoder.out_features == 1:
            loss = self.loss_fn(logits.squeeze(dim=-1), labels.float())
        else:
            loss = self.loss_fn(logits, labels.long())

        return loss, logits

    def encode_string(self, data):
        self.tokenizer_cds.truncation_side = 'left'
        tok_cds = self.tokenizer_cds(
            data['cds'],
            truncation=True,
            padding="max_length",
            max_length=1024
        )

        self.tokenizer_3utr.truncation_side = 'right'
        tok_3utr = self.tokenizer_3utr(
            data['3utr'],
            truncation=True,
            padding="max_length",
            max_length=1024
        )

        return {
            'input_ids2': tok_cds['input_ids'],
            'attention_mask2': tok_cds['attention_mask'],
            'input_ids3': tok_3utr['input_ids'],
            'attention_mask3': tok_3utr['attention_mask'],
            'labels': data['label']
        }
