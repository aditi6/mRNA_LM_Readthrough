import pandas as pd
from datasets import Dataset

def mytok(seq, kmer_len, s):
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq)-kmer_len)+1, s):
        kmer_list.append(seq[j:j+kmer_len])
    return kmer_list

########### loading dp dataset (unchanged from dataload.py)
def build_dp_dataset():
    def load_dataset(data_path, split):
        df = pd.read_csv(data_path)
        df = df[df["split"] == split]
        df = df.dropna(subset=["bp_zscore"])

        utr5 = df["UTR5"].values.tolist()
        utr3 = df["UTR3"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["bp_zscore"].values.tolist()

        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))

        assert len(seqs) == len(ys)
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/translation_rate.csv", "train")
    valid_seqs, valid_ys = load_dataset("data/translation_rate.csv", "valid")
    test_seqs, test_ys   = load_dataset("data/translation_rate.csv", "test")

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_class_dataset():
    def load_dataset(data_path, split):
        df = pd.read_csv(data_path)
        df = df[df["split"] == split]

        utr5 = df["5' UTR"].values.tolist()
        utr3 = df["3' UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["ClassificationID"].values.tolist()

        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))

        assert len(seqs) == len(ys)
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/protein_expression_5class.csv", "train")
    valid_seqs, valid_ys = load_dataset("data/protein_expression_5class.csv", "valid")
    test_seqs, test_ys   = load_dataset("data/protein_expression_5class.csv", "test")

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_liver_dataset():
    def load_dataset(data_path, split):
        df = pd.read_csv(data_path)
        df = df[df["split"] == split]

        utr5 = df["5' UTR"].values.tolist()
        utr3 = df["3' UTR"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["Liver_norm"].values.tolist()

        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))

        assert len(seqs) == len(ys)
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/transcript_expression_liver.csv", "train")
    valid_seqs, valid_ys = load_dataset("data/transcript_expression_liver.csv", "valid")
    test_seqs, test_ys   = load_dataset("data/transcript_expression_liver.csv", "test")

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test

def build_saluki_dataset(cross):
    def load_dataset(data_path):
        df = pd.read_csv(data_path)
        df = df.fillna('')
        df = df.dropna(subset=["y"])

        utr5 = df["UTR5"].values.tolist()
        utr3 = df["UTR3"].values.tolist()
        cds = df["CDS"].values.tolist()
        ys = df["y"].values.tolist()

        utr5 = [" ".join(mytok(seq, 1, 1)) for seq in utr5]
        cds  = [" ".join(mytok(seq, 3, 3)) for seq in cds]
        utr3 = [" ".join(mytok(seq, 1, 1)) for seq in utr3]
        seqs = list(zip(utr5, cds, utr3))

        assert len(seqs) == len(ys)
        return seqs, ys

    train_seqs, train_ys = load_dataset("data/mrna_half-life.csv", "train")
    valid_seqs, valid_ys = load_dataset("data/mrna_half-life.csv", "valid")
    test_seqs, test_ys   = load_dataset("data/mrna_half-life.csv", "test")

    ds_train = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"5utr": seq[0], "cds": seq[1], "3utr": seq[2], "label": y} for seq, y in zip(test_seqs, test_ys)])

    return ds_train, ds_valid, ds_test


# =============================================================================
# WINDOW PARAMETERS
# These define the sequence region fed to the model for readthrough prediction.
# Only build_readthrough_dataset() uses these — all other dataset builders are
# unchanged from dataload.py.
#
# Biology: stop codon readthrough (SCR) is determined by a localised sequence
# context around the stop codon. The two most influential regions are:
#   (1) The final few codons of the CDS leading up to and including the stop
#       codon — local codon usage, the stop codon identity (TGA > TAG > TAA),
#       and the upstream amino acid context all modulate decoding efficiency.
#   (2) The immediate downstream 3'UTR — readthrough-promoting motifs such as
#       CARYYA (Philipson et al.) and mRNA secondary structures that slow
#       ribosome release are concentrated within ~100-200nt of the stop codon.
#
# Feeding the full CDS (mean 1743nt, ~581 codons) and full 3'UTR (mean 1697nt)
# to a mean-pooling model dilutes the stop codon signal to near-zero. Data
# quality analysis showed 49% of 3'UTRs exceed 1024nt (the model's max_length),
# meaning the most relevant region was being kept but surrounded by irrelevant
# sequence. For CDS, 12% exceeded 3072nt and had their stop codon truncated
# entirely under the default right-side truncation.
#
# Windowing eliminates both problems: all sequences fit within max_length with
# no truncation, and the model sees only the biologically relevant region.
# =============================================================================

# Last N codons of the CDS to keep (right-aligned, so the stop codon is always
# the final token). 20 codons = 60nt upstream context + stop codon (3nt) = 63nt
# total. This is generous relative to the minimal ±6nt context from literature,
# while remaining short enough to avoid dilution under mean pooling.
CDS_WINDOW_CODONS = 20

# First N nucleotides of the 3'UTR to keep (left-aligned, immediately after
# the stop codon). 200nt captures the CARYYA motif search space and any
# proximal secondary structures, while excluding the distal 3'UTR that has
# no known role in readthrough.
UTR3_WINDOW_NT = 200


def build_readthrough_dataset():
    """
    Loads the readthrough dataset and applies stop-codon-centred windowing
    before tokenisation.

    Key differences from dataload.py:
      - CDS is trimmed to the last CDS_WINDOW_CODONS codons before tokenising,
        ensuring the stop codon and its upstream context are always present.
      - 3'UTR is trimmed to the first UTR3_WINDOW_NT nucleotides, focusing the
        model on the region immediately downstream of the stop codon.
      - After windowing, no sequence exceeds max_length=1024 tokens, so
        tokeniser truncation never occurs and no information is lost.
    """

    def extract_cds_window(cds_seq):
        """
        Return the last CDS_WINDOW_CODONS codons of cds_seq (right-aligned).

        Right-alignment is critical: the stop codon occupies the last 3nt of
        the CDS. By taking the suffix, we guarantee the stop codon is always
        the final codon in the window regardless of CDS length. If the CDS is
        shorter than the window, the full sequence is returned unchanged.
        """
        window_nt = CDS_WINDOW_CODONS * 3
        return cds_seq[-window_nt:] if len(cds_seq) > window_nt else cds_seq

    def extract_utr3_window(utr3_seq):
        """
        Return the first UTR3_WINDOW_NT nucleotides of utr3_seq (left-aligned).

        Left-alignment preserves the nucleotides immediately downstream of the
        stop codon — precisely where readthrough-promoting motifs are found.
        If the 3'UTR is shorter than the window, it is returned unchanged.
        """
        return utr3_seq[:UTR3_WINDOW_NT]

    def load_dataset(data_path, split):
        df = pd.read_csv(data_path)
        df = df[df["split"] == split]
        df = df.dropna(subset=["cds", "3utr", "rrts"])

        cds_sequences  = df["cds"].values.tolist()
        utr3_sequences = df["3utr"].values.tolist()
        labels         = df["rrts"].values.tolist()

        # --- CHANGED: apply stop-codon-centred windows before tokenisation ---
        # In dataload.py the full sequences were passed directly to mytok.
        # Here we first trim each sequence to its relevant window so the model
        # receives focused, noise-free input.
        cds_sequences  = [extract_cds_window(seq)  for seq in cds_sequences]
        utr3_sequences = [extract_utr3_window(seq) for seq in utr3_sequences]
        # ---------------------------------------------------------------------

        # Tokenise: CDS as codon triplets, 3'UTR as single nucleotides.
        # Unchanged from dataload.py — only the input sequences are shorter.
        cds_tokenized  = [" ".join(mytok(seq, 3, 3)) for seq in cds_sequences]
        utr3_tokenized = [" ".join(mytok(seq, 1, 1)) for seq in utr3_sequences]

        all_seqs = list(zip(cds_tokenized, utr3_tokenized))
        assert len(all_seqs) == len(labels)

        return all_seqs, labels

    csv_path = "/workspace/mRNA_LM_Readthrough/data/readthrough_data.csv"

    train_seqs, train_ys = load_dataset(csv_path, "train")
    valid_seqs, valid_ys = load_dataset(csv_path, "valid")
    test_seqs,  test_ys  = load_dataset(csv_path, "test")

    ds_train = Dataset.from_list([{"cds": seq[0], "3utr": seq[1], "label": y} for seq, y in zip(train_seqs, train_ys)])
    ds_valid = Dataset.from_list([{"cds": seq[0], "3utr": seq[1], "label": y} for seq, y in zip(valid_seqs, valid_ys)])
    ds_test  = Dataset.from_list([{"cds": seq[0], "3utr": seq[1], "label": y} for seq, y in zip(test_seqs,  test_ys)])

    return ds_train, ds_valid, ds_test
