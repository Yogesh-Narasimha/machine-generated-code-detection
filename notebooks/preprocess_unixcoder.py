"""
preprocess_unixcoder.py
=======================
Tokenizes the Task_A dataset for UniXcoder fine-tuning.

Usage:
  # Step 1 — Validate on 1,000 samples (fast, ~10 seconds)
  source venv/bin/activate && python preprocess_unixcoder.py --mode validate

  # Step 2 — Process the full dataset (saves to disk)
  source venv/bin/activate && python preprocess_unixcoder.py --mode full --num_proc 4
"""

import os
import sys
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/unixcoder-base"
MAX_LENGTH   = 512
OUTPUT_DIR   = "processed_data/unixcoder"
DATA_DIR     = "Task_A"

# ── Tokenizer ──────────────────────────────────────────────────────────────
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(examples):
    """Tokenize a batch of code samples."""
    return tokenizer(
        examples["code"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,          # Padding is done by the DataCollator at training time
    )


def load_split(split: str, n_samples: int = None) -> Dataset:
    """Load a parquet split into a HuggingFace Dataset."""
    path = os.path.join(DATA_DIR, f"{split}.parquet")
    df   = pd.read_parquet(path)
    if n_samples:
        df = df.head(n_samples)

    # Test set has no labels — fill with -1 as sentinel
    if "label" not in df.columns:
        df["label"] = -1
    if "language" not in df.columns:
        df["language"] = "unknown"

    # Keep only necessary columns
    df = df[["code", "label", "language"]].reset_index(drop=True)
    return Dataset.from_pandas(df)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "full"], default="validate",
                        help="'validate' runs on 1000 samples; 'full' processes all splits.")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of parallel workers for tokenization (full mode only).")
    args = parser.parse_args()

    if args.mode == "validate":
        print("\n── VALIDATE MODE (1,000 samples from train) ──")
        ds = load_split("train", n_samples=1000)
        print(f"Loaded {len(ds)} samples.  Columns: {ds.column_names}")

        tokenized = ds.map(tokenize_fn, batched=True, batch_size=64)
        print(f"Tokenization done. Features: {tokenized.features}")
        print(f"\nSample 0:")
        s = tokenized[0]
        print(f"  input_ids length : {len(s['input_ids'])}")
        print(f"  attention_mask   : {s['attention_mask'][:10]}...")
        print(f"  label            : {s['label']}")

        # Sanity: max token length
        lengths = [len(x["input_ids"]) for x in tokenized]
        print(f"\nToken lengths — min: {min(lengths)}, max: {max(lengths)}, "
              f"mean: {sum(lengths)/len(lengths):.1f}")
        print("\n✅ UniXcoder validation passed!")

    elif args.mode == "full":
        print(f"\n── FULL MODE — saving to '{OUTPUT_DIR}' ──")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        splits = {"train": None, "validation": None, "test": None}
        tokenized_splits = {}

        for split_name in splits:
            print(f"\nProcessing split: {split_name}")
            ds = load_split(split_name)
            print(f"  Loaded {len(ds):,} samples")

            tok_ds = ds.map(
                tokenize_fn,
                batched=True,
                batch_size=256,
                num_proc=args.num_proc,
                desc=f"Tokenizing {split_name}",
            )
            tok_ds = tok_ds.remove_columns(["code"])   # Save disk space; raw code no longer needed
            tokenized_splits[split_name] = tok_ds

        dataset_dict = DatasetDict(tokenized_splits)
        dataset_dict.save_to_disk(OUTPUT_DIR)
        print(f"\n✅ Full dataset saved to: {OUTPUT_DIR}")
        print(dataset_dict)


if __name__ == "__main__":
    main()
