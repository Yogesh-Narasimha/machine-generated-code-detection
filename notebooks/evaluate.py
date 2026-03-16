import os
import argparse
import torch
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from train import GraphCodeBERTCollator # Import custom collator if needed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["unixcoder", "graphcodebert"])
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # 1. Configuration setup
    if args.model == "unixcoder":
        model_name = "microsoft/unixcoder-base"
        model_dir = "saved_models/unixcoder"
        data_dir = "processed_data/unixcoder"
        submission_file = "unixcoder_submission.csv"
        collator = None
    else:
        model_name = "microsoft/graphcodebert-base"
        model_dir = "saved_models/graphcodebert"
        data_dir = "processed_data/graphcodebert"
        submission_file = "graphcodebert_submission.csv"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        collator = GraphCodeBERTCollator(tokenizer)

    print(f"\n[1] Loading trained model from {model_dir}...")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found. Please train {args.model} first!")
    
    # Load the finetuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    print(f"\n[2] Loading preprocessed test dataset from {data_dir}...")
    ds = load_from_disk(data_dir)
    test_ds = ds["test"]
    print(f"Loaded {len(test_ds)} test samples.")

    # 3. Initialize Trainer just for prediction (no training args required for simple inference)
    # Using Trainer.predict() handles batching and device placement automatically
    print("\n[3] Running inference on the test set...")
    trainer = Trainer(
        model=model,
        data_collator=collator,
    )

    outputs = trainer.predict(test_ds)
    logits = outputs.predictions
    # Binary classification: index 0 (human) or 1 (AI)
    predictions = np.argmax(logits, axis=-1)

    print("\n[4] Generating submission CSV...")
    # The original test.parquet had an 'id' or 'ID' column. We map it back.
    # Usually datasets preserve all original un-removed columns, or we can just load the raw test parquet.
    # We will load the original test.parquet to guarantee we have the exact IDs in order.
    try:
        raw_test_df = pd.read_parquet("Task_A/test.parquet")
        if "id" in raw_test_df.columns:
            ids = raw_test_df["id"]
        elif "ID" in raw_test_df.columns:
            ids = raw_test_df["ID"]
        else:
            ids = range(len(predictions))
            print("Warning: Could not find id column in Task_A/test.parquet")
    except Exception as e:
        print("Warning: Could not load original test.parquet to get IDs, using sequential index.")
        ids = range(len(predictions))

    submission = pd.DataFrame({
        "id": ids,
        "label": predictions
    })

    submission.to_csv(submission_file, index=False)
    print(f"✅ Done! Saved {len(predictions)} predictions to '{submission_file}'.")

if __name__ == "__main__":
    main()
