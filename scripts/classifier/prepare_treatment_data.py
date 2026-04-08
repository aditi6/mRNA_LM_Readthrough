"""
Prepare treatment dataset for classification.

For each drug CSV in --data_dir:
  - Reads transcript ID + RRTS (binarised: >0 → 1, else 0)
  - Fetches canonical cDNA from Ensembl REST API (with retry + cache)
  - Locates stop codon position in cDNA using the stopcodon column
  - Extracts ±context_nt window around stop codon
  - Outputs a single merged CSV: transcript, drug, label, stop_codon, nt_seq

Usage:
    python prepare_treatment_data.py --data_dir data/treatments --out merged_treatments.csv

Output columns: transcript, drug, label, stop_codon, stop_pos, nt_seq (full cDNA)
Context windowing is done at training time, not here.
"""

import argparse, json, os, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd
import numpy as np
import requests
from pathlib import Path

ENSEMBL_API = "https://rest.ensembl.org/sequence/id/{}"
HEADERS = {"Content-Type": "application/json"}
STOP_CODONS = {"taa", "tag", "tga", "uaa", "uag", "uga"}


# ── Ensembl fetching with cache ───────────────────────────────────────────────
def fetch_cdna_single(tid: str, retries: int = 4) -> tuple[str, str | None]:
    """Fetch one transcript (no version suffix). Returns (tid, seq_or_None)."""
    url = ENSEMBL_API.format(tid) + "?type=cdna"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return tid, r.json()["seq"].lower()
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                time.sleep(wait)
            elif r.status_code in (400, 404):
                return tid, None
            else:
                time.sleep(1 * (attempt + 1))
        except Exception:
            time.sleep(2)
    return tid, None


def fetch_all_parallel(transcript_ids: list, cache: dict, cache_file: str,
                       max_workers: int = 10) -> dict:
    """Fetch all uncached transcript IDs in parallel, update cache in place."""
    need = [t.split(".")[0] for t in transcript_ids
            if t.split(".")[0] not in cache]
    need = list(dict.fromkeys(need))   # deduplicate preserving order

    if not need:
        return cache

    print(f"  Fetching {len(need)} sequences with {max_workers} threads...")
    lock = Lock()
    done = [0]

    def fetch_and_update(tid):
        tid, seq = fetch_cdna_single(tid)
        with lock:
            cache[tid] = seq
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"    {done[0]}/{len(need)}")
                with open(cache_file, "w") as f:
                    json.dump(cache, f)
        return tid

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_and_update, tid) for tid in need]
        for _ in as_completed(futures):
            pass

    with open(cache_file, "w") as f:
        json.dump(cache, f)
    print(f"  Done fetching. Cache now has {len(cache)} entries.")
    return cache


# ── Stop codon localisation ───────────────────────────────────────────────────
def find_stop_in_cdna(seq: str, stop_codon: str) -> int | None:
    """
    Find the last in-frame stop codon in the CDS region.
    Strategy: scan from 3' end of likely CDS for the stop codon,
    ensuring it's in-frame with a valid start (divisible by 3 from ATG).
    Returns 0-indexed position of stop codon start in seq.
    """
    stop = stop_codon.lower().replace("u", "t")
    if stop not in STOP_CODONS:
        stop = stop[:3]

    # Find ATG start
    atg = seq.find("atg")
    if atg == -1:
        return None

    # Scan all occurrences of the stop codon in-frame from ATG
    best = None
    pos = atg
    while pos < len(seq) - 2:
        codon = seq[pos:pos+3]
        if codon == stop:
            best = pos
        pos += 3

    return best


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/treatments")
    parser.add_argument("--out",         default="merged_treatments.csv")
    parser.add_argument("--cache_file",  default="seq_cache.json",
                        help="JSON file to cache fetched sequences")
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    # Load or init sequence cache
    if os.path.exists(args.cache_file):
        with open(args.cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached sequences")
    else:
        cache = {}

    drug_files = sorted(Path(args.data_dir).glob("*.csv"))
    print(f"Found {len(drug_files)} drug files: {[f.stem for f in drug_files]}")

    all_rows = []

    for drug_file in drug_files:
        drug = drug_file.stem
        print(f"\n── {drug} ──")

        df = pd.read_csv(drug_file)
        df.columns = [c.lstrip("#") for c in df.columns]
        df["label"] = (df["RRTS"] > 0).astype(int)

        n_total = len(df)
        n_pos   = df["label"].sum()
        print(f"  {n_total} transcripts, {n_pos} positive ({100*n_pos/n_total:.1f}%)")

        # Fetch all unique transcripts for this drug in parallel
        fetch_all_parallel(
            df["transcript"].tolist(), cache, args.cache_file,
            max_workers=args.max_workers)

        # Build rows — save full cDNA + stop position, no windowing here
        n_ok, n_fail = 0, 0
        for _, row in df.iterrows():
            tid  = row["transcript"]
            seq  = cache.get(tid.split(".")[0])
            stop = str(row.get("stopcodon", "")).lower().replace("u", "t")

            if seq is None:
                n_fail += 1
                continue

            stop_pos = find_stop_in_cdna(seq, stop)
            if stop_pos is None:
                n_fail += 1
                continue

            all_rows.append({
                "transcript": tid,
                "drug":       drug,
                "label":      int(row["label"]),
                "stop_codon": stop,
                "stop_pos":   stop_pos,
                "nt_seq":     seq,          # full cDNA sequence
            })
            n_ok += 1

        print(f"  OK={n_ok}  failed/skipped={n_fail}")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} rows to {args.out}")

    # Summary
    print("\nPer-drug summary:")
    for drug, grp in out_df.groupby("drug"):
        n = len(grp)
        pos = grp["label"].sum()
        print(f"  {drug:<20} n={n:<5} pos={pos:<5} ({100*pos/n:.1f}%)")


if __name__ == "__main__":
    main()
