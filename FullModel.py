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

from transformers import BertForMaskedLM, BertTokenizerFast # CHANGED: Using BertTokenizerFast for consistency
from peft import LoraConfig, TaskType, get_peft_model

class FullModel(torch.nn.Module):
    def __init__(self, num_labels, class_weights, lorar, lalpha, ldropout, head_dim=768, head_droupout=0.5, useCLIP=False, temperature=0.07, clip_coeff=0.2):
        super(FullModel, self).__init__()
        
        # Initialize tokenizer attributes that will be built by the method below
        self.tokenizer_cds = None
        self.tokenizer_3utr = None
        self.build_tokenizer()
        self.CLIP = useCLIP
        
        # Load only the necessary models
        print("Loading models for CDS and 3'UTR...")
        self.utr3 = BertForMaskedLM.from_pretrained("/workspace/mrna_3utr_model_p2_cp99900_best")
        self.cds = BertForMaskedLM.from_pretrained("/workspace/codonbert")

        # Adjust LoRA setup for the two models
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
            
        # Dense layers for CLIP-style structure
        self.dense_cds1 = nn.Linear(768, 768)
        self.dense_cds2 = nn.Linear(768, 768)
        self.dense_utr3 = nn.Linear(768, 768)

        # CRITICAL: Change input dimension from 768*3 to 768*2
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

    # --- ADDED: The corrected build_tokenizer method ---
  

    def build_tokenizer(self):
        lst_ele = list('AUGC')

        # 1. Build Codon Tokenizer for CDS
        lst_voc_cds = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for a1 in lst_ele:
            for a2 in lst_ele:
                for a3 in lst_ele:
                    lst_voc_cds.extend([f'{a1}{a2}{a3}'])
        dic_voc_cds = {token: i for i, token in enumerate(lst_voc_cds)}
        
        tokenizer_cds_obj = Tokenizer(WordLevel(vocab=dic_voc_cds, unk_token="[UNK]"))
        
        # --- ADDED: Define a Normalizer that doesn't alter the sequence ---
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

        # 2. Build Nucleotide Tokenizer for 3'UTR
        lst_voc_utr = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for a1 in lst_ele:
            lst_voc_utr.extend([f'{a1}'])
        dic_voc_utr = {token: i for i, token in enumerate(lst_voc_utr)}
        
        tokenizer_3utr_obj = Tokenizer(WordLevel(vocab=dic_voc_utr, unk_token="[UNK]"))
        
        # --- ADDED: Define a Normalizer that doesn't alter the sequence ---
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
        input_mask_expanded = token_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / sum_mask
        return sum_embeddings

    def forward(self, input_ids2, attention_mask2, input_ids3, attention_mask3, labels, **kwargs):
        # Get embeddings for only CDS and 3'UTR
        cds_embeds  = self.cds(input_ids=input_ids2, attention_mask=attention_mask2, output_hidden_states=True)["hidden_states"][-1]
        utr3_embeds = self.utr3(input_ids=input_ids3, attention_mask=attention_mask3, output_hidden_states=True)["hidden_states"][-1]

        cds_sum_embeddings  = self.get_mean_token_embeddings(cds_embeds[:, 1:-1, :], attention_mask2[:, 1:-1])
        utr3_sum_embeddings = self.get_mean_token_embeddings(utr3_embeds[:, 1:-1, :], attention_mask3[:, 1:-1])

        # Concatenate only two embeddings
        joint_embed = torch.cat([cds_sum_embeddings, utr3_sum_embeddings], dim=1)

        hidden_states = self.final_dense(joint_embed)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.decoder(hidden_states)
        
        # --- CHANGED: Conditional label type conversion ---
        # Check if this is a regression task (output is 1) or classification (output > 1)
        if self.decoder.out_features == 1:
        # For MSELoss in regression, labels must be floats
            loss = self.loss_fn(logits.squeeze(dim=-1), labels.float())
        else:
        # For CrossEntropyLoss in classification, labels must be long integers
            loss = self.loss_fn(logits, labels.long())

        return loss, logits

    # --- ADDED: The corrected encode_string method ---
    def encode_string(self, data):
        # CDS: truncate from the LEFT (remove beginning of long sequences).
        # Rationale: the stop codon is at the END of the CDS and is the primary
        # sequence feature for readthrough prediction. Default right-truncation
        # would discard the stop codon for ~12% of sequences (CDS > 3072nt),
        # directly destroying the most informative signal for this task.
        self.tokenizer_cds.truncation_side = 'left'
        tok_cds = self.tokenizer_cds(
            data['cds'],
            truncation=True,
            padding="max_length",
            max_length=1024
        )

        # 3'UTR: keep default right-truncation (truncate from the END).
        # Rationale: the most relevant readthrough context is at the START of
        # the 3'UTR (immediately after the stop codon), so preserving the
        # beginning is correct. ~49% of 3'UTRs exceed 1024nt so truncation
        # is common, but the important region is retained.
        self.tokenizer_3utr.truncation_side = 'right'
        tok_3utr = self.tokenizer_3utr(
            data['3utr'],
            truncation=True,
            padding="max_length",
            max_length=1024
        )

        # Return a dictionary with only the necessary inputs AND the labels
        return {
            'input_ids2': tok_cds['input_ids'],
            'attention_mask2': tok_cds['attention_mask'],
            'input_ids3': tok_3utr['input_ids'],
            'attention_mask3': tok_3utr['attention_mask'],
            'labels': data['label'] # RE-ADDED THIS CRITICAL LINE
        }

    # NOTE: The other methods like `contrastive_loss` can remain, but the main `forward`
    #       method's logic for CLIP would need to be re-evaluated if you plan to use it.
    #       For now, the non-CLIP path is fully functional with this code.