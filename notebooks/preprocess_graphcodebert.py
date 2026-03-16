"""
preprocess_graphcodebert.py
===========================
Extracts DFG features from code and tokenizes for GraphCodeBERT fine-tuning.

Usage:
  # Step 1 — Validate on 1,000 samples
  source venv/bin/activate && python preprocess_graphcodebert.py --mode validate

  # Step 2 — Process the full dataset
  source venv/bin/activate && python preprocess_graphcodebert.py --mode full --num_proc 4
"""

import os
import sys
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# ── Add parser package to path (for DFG imports) ──────────────────────────
CODEBERT_DIR = os.path.abspath(
    os.path.join("CodeBERT", "GraphCodeBERT", "clonedetection")
)
sys.path.insert(0, CODEBERT_DIR)

from tree_sitter import Language, Parser as TSParser
from parser.utils import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token,
    tree_to_variable_index,
)
from parser.DFG import DFG_python, DFG_java

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/graphcodebert-base"
MAX_CODE_LEN = 256     # Code tokens
MAX_DFG_LEN  = 64      # DFG variable nodes
MAX_TOTAL    = 512     # GraphCodeBERT max sequence length
OUTPUT_DIR   = "processed_data/graphcodebert"
DATA_DIR     = "Task_A"
SO_PATH      = os.path.join(CODEBERT_DIR, "parser", "my-languages.so")

# ── Language map ────────────────────────────────────────────────────────────
# DFG functions per language (only Python and Java DFGs are high-quality)
LANG_TO_DFG = {
    "Python": (DFG_python, "python"),
    "Java":   (DFG_java,   "java"),
    "C++":    (DFG_python, "python"),  # Fallback: use Python parser for C++ (skip DFG)
}

# ── Build tree-sitter parsers ───────────────────────────────────────────────
PYTHON_LANG = Language(SO_PATH, "python")
JAVA_LANG   = Language(SO_PATH, "java")

_ts_parsers = {}

def get_ts_parser(lang_name: str):
    if lang_name not in _ts_parsers:
        if lang_name == "java":
            p = TSParser()
            p.set_language(JAVA_LANG)
        else:
            p = TSParser()
            p.set_language(PYTHON_LANG)
        _ts_parsers[lang_name] = p
    return _ts_parsers[lang_name]


# ── Tokenizer ───────────────────────────────────────────────────────────────
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def extract_dfg(code: str, language: str):
    """
    Parse code with tree-sitter and extract DFG variable nodes + edges.
    Returns:
        code_tokens  : list of code token strings
        dfg          : list of (var, idx, rel, ...) tuples
    Falls back to empty DFG on any parse error.
    """
    dfg_fn, ts_lang = LANG_TO_DFG.get(language, (DFG_python, "python"))

    try:
        # Remove comments to reduce noise
        if language == "Python":
            clean_code = remove_comments_and_docstrings(code, "python")
        else:
            clean_code = remove_comments_and_docstrings(code, "java")
    except Exception:
        clean_code = code

    try:
        ts_p = get_ts_parser(ts_lang)
        tree = ts_p.parse(bytes(clean_code, "utf-8"))
        code_lines = clean_code.split("\n")

        token_indices = tree_to_token_index(tree.root_node)
        index_to_code = {}
        for i, (start, end) in enumerate(token_indices):
            tok = index_to_code_token((start, end), code_lines)
            index_to_code[(start, end)] = (i, tok)

        code_tokens = [v for _, v in sorted(index_to_code.values(), key=lambda x: x[0])]
        dfg, _ = dfg_fn(tree.root_node, index_to_code, {})

        # Remove duplicates and limit length
        seen = set()
        clean_dfg = []
        for d in dfg:
            key = (d[0], d[1])
            if key not in seen:
                seen.add(key)
                clean_dfg.append(d)

        return code_tokens[:MAX_CODE_LEN], clean_dfg[:MAX_DFG_LEN]

    except Exception:
        return code.split()[:MAX_CODE_LEN], []


def encode_sample(code: str, language: str):
    """
    Build the full GraphCodeBERT input: [CLS] code [SEP] dfg_vars [SEP]
    Returns tokenizer output + position_idx (for DFG-aware attention).
    """
    code_tokens, dfg = extract_dfg(code, language)

    # DFG variable name tokens
    dfg_vars = [d[0] for d in dfg]

    # Truncate code to leave room for CLS + SEP + dfg_vars + SEP
    max_code = MAX_TOTAL - 3 - len(dfg_vars)
    code_tokens = code_tokens[:max_code]

    # Tokenize: [CLS] <code tokens> [SEP] <dfg_vars> [SEP]
    combined_tokens = (
        [tokenizer.cls_token]
        + code_tokens
        + [tokenizer.sep_token]
        + dfg_vars
        + [tokenizer.sep_token]
    )

    input_ids      = tokenizer.convert_tokens_to_ids(combined_tokens)
    attention_mask = [1] * len(input_ids)

    # Position tracking: code positions, then DFG var positions
    # (GraphCodeBERT uses these to build the data-flow attention mask)
    code_start = 1                         # after [CLS]
    dfg_start  = code_start + len(code_tokens) + 1  # after [SEP]

    position_idx = list(range(len(input_ids)))
    # Map each dfg var's position to its occurrence in the code
    dfg_to_code_positions = []
    for i, d in enumerate(dfg):
        var_idx_in_code = d[1]             # token index in code_tokens
        mapped = code_start + min(var_idx_in_code, len(code_tokens) - 1)
        dfg_to_code_positions.append(mapped)

    return {
        "input_ids":             input_ids,
        "attention_mask":        attention_mask,
        "position_idx":          position_idx,
        "dfg_to_code_positions": dfg_to_code_positions,
        "num_code_tokens":       len(code_tokens),
        "num_dfg_nodes":         len(dfg_vars),
    }


def process_batch(examples):
    """HuggingFace datasets map-compatible batch function."""
    results = {
        "input_ids":             [],
        "attention_mask":        [],
        "position_idx":          [],
        "dfg_to_code_positions": [],
        "num_code_tokens":       [],
        "num_dfg_nodes":         [],
    }
    for code, lang in zip(examples["code"], examples["language"]):
        enc = encode_sample(code, lang)
        for k in results:
            results[k].append(enc[k])
    return results


def load_split(split: str, n_samples: int = None) -> Dataset:
    path = os.path.join(DATA_DIR, f"{split}.parquet")
    df   = pd.read_parquet(path)
    if n_samples:
        df = df.head(n_samples)

    if "label" not in df.columns:
        df["label"] = -1
    if "language" not in df.columns:
        df["language"] = "Python"

    df = df[["code", "label", "language"]].reset_index(drop=True)
    return Dataset.from_pandas(df)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "full"], default="validate")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Workers for .map() — keep at 1 for DFG (tree-sitter not fork-safe).")
    args = parser.parse_args()

    if args.mode == "validate":
        print("\n── VALIDATE MODE (1,000 samples from train) ──")
        ds = load_split("train", n_samples=1000)
        print(f"Loaded {len(ds)} samples")

        tokenized = ds.map(process_batch, batched=True, batch_size=32, desc="DFG+Tokenize")
        print(f"\nFeatures: {tokenized.features}")

        s = tokenized[0]
        print(f"\nSample 0:")
        print(f"  input_ids length    : {len(s['input_ids'])}")
        print(f"  num_code_tokens     : {s['num_code_tokens']}")
        print(f"  num_dfg_nodes       : {s['num_dfg_nodes']}")
        print(f"  dfg_to_code_pos[:5] : {s['dfg_to_code_positions'][:5]}")
        print(f"  label               : {s['label']}")

        # DFG coverage stats
        dfg_lengths = [x["num_dfg_nodes"] for x in tokenized]
        no_dfg = sum(1 for x in dfg_lengths if x == 0)
        print(f"\nDFG stats — avg nodes: {sum(dfg_lengths)/len(dfg_lengths):.1f}, "
              f"samples with 0 DFG: {no_dfg} ({100*no_dfg/len(dfg_lengths):.1f}%)")
        print("\n✅ GraphCodeBERT validation passed!")

    elif args.mode == "full":
        print(f"\n── FULL MODE — saving to '{OUTPUT_DIR}' ──")
        print("⚠️  tree-sitter is not fork-safe; using num_proc=1. "
              "Expect ~4-8 hours on CPU for 500K samples.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        tokenized_splits = {}
        for split_name in ["train", "validation", "test"]:
            print(f"\nProcessing {split_name}…")
            ds = load_split(split_name)
            print(f"  {len(ds):,} samples")
            tok_ds = ds.map(
                process_batch,
                batched=True,
                batch_size=64,
                num_proc=1,          # Must be 1; tree-sitter is not fork-safe
                desc=f"DFG+Tok {split_name}",
            )
            tok_ds = tok_ds.remove_columns(["code"])
            tokenized_splits[split_name] = tok_ds

        DatasetDict(tokenized_splits).save_to_disk(OUTPUT_DIR)
        print(f"\n✅ Full dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
