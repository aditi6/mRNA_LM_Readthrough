import torch
import torch.nn as nn
from transformers.activations import gelu

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import *
from tokenizers.processors import BertProcessing

# --- EDIT: Import AutoConfig to modify the model's configuration before loading it ---
from transformers import BertForSequenceClassification, PreTrainedTokenizerFast, AutoConfig

########### PEFT
from peft import LoraConfig, TaskType
from peft import get_peft_model

class OneModel(torch.nn.Module):
    def __init__(self, region, num_labels, class_weights, lorar, lalpha, ldropout, output_hidden_states=False):
        super(OneModel, self).__init__()
        
        self.region = region 
        self.max_length = 1024
        if self.region == "5utr" or self.region == "3utr":
            self.max_length = 512
        
        # --- This is the original logic: build the tokenizer from scratch ---
        self.tokenizer = None
        self.build_tokenizer()
        
        # --- EDIT: Changed model paths to match your Colab environment ---
        if self.region == "5utr":
            model_dir = "/content/drive/MyDrive/Readthrough_project/mrna_3utr_model"
        elif self.region == "3utr":
            model_dir = "/content/drive/MyDrive/Readthrough_project/mrna_3utr_model"
        elif self.region == "cds":
            model_dir = "/content/drive/MyDrive/Readthrough_project/codonbert"
        else:
            print("wrong region!!", self.region)
            exit(0)
        
        # --- This block loads the model's configuration and updates it ---
        print(f"Loading config from {model_dir} and updating parameters...")
        config = AutoConfig.from_pretrained(model_dir)
        config.vocab_size = self.tokenizer.vocab_size # Sync vocab size
        config.num_labels = num_labels # Set num_labels on the config object
        config.output_hidden_states = output_hidden_states # Set output_hidden_states on the config object
        
        # Load the model using the MODIFIED config.
        print(f"Loading model from local directory: {model_dir}")
        self.model = BertForSequenceClassification.from_pretrained(
            model_dir, 
            config=config, # Pass the fully updated config here
            # --- EDIT: Add this flag to resolve the size mismatch error ---
            # This will load all weights except for the word_embeddings layer,
            # which will be re-initialized to match the new vocabulary size.
            ignore_mismatched_sizes=True
        )
        # --- END OF EDIT BLOCK ---

        # --- LoRA Configuration (original logic) ---
        if lorar > 0:
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                     r=lorar, 
                                     lora_alpha=lalpha, 
                                     lora_dropout=ldropout,
                                     use_rslora=True)

            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
    def build_tokenizer(self):
        """
        Builds the tokenizer from scratch based on region.
        This is the original logic from your script.
        """
        lst_ele = list('ATGC')
        lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        if self.region == "cds":
            for a1 in lst_ele:
                for a2 in lst_ele:
                    for a3 in lst_ele:
                        lst_voc.extend([f'{a1}{a2}{a3}'])
        else: # For '5utr' or '3utr'
            for a1 in lst_ele:
                lst_voc.extend([f'{a1}'])
                
        dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
        tokenizer_obj = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
        tokenizer_obj.add_special_tokens(['[PAD]','[CLS]', '[UNK]', '[SEP]','[MASK]'])
        tokenizer_obj.pre_tokenizer = Whitespace()
        tokenizer_obj.post_processor = BertProcessing(
            ("[SEP]", dic_voc['[SEP]']),
            ("[CLS]", dic_voc['[CLS]']),
        )

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj, 
            unk_token='[UNK]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            cls_token='[CLS]',
            mask_token='[MASK]'
        )

    def encode_string(self, data):
        """Tokenizes a dictionary of sequences based on the model's region."""
        sequence_text = data.get(self.region)
        if sequence_text is None:
            raise KeyError(f"The key '{self.region}' was not found in the input data dictionary.")
            
        return self.tokenizer(
            sequence_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt" # Ensure PyTorch tensors are returned
        )

    def forward(self, **kwargs):
        """Passes tokenized inputs directly to the underlying model."""
        return self.model(**kwargs)




