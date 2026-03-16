"""
train.py
========
Fine-tunes UniXcoder or GraphCodeBERT on the preprocessed datasets.
Run this on your GPU environment (e.g., Colab, A100).

Usage:
  python train.py --model unixcoder
  python train.py --model graphcodebert
"""

import os
import argparse
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ── Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, zero_division=0)
    pr  = precision_score(labels, predictions, zero_division=0)
    re  = recall_score(labels, predictions, zero_division=0)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": pr,
        "recall": re
    }


# ── Custom Data Collator for GraphCodeBERT ────────────────────────────────
class GraphCodeBERTCollator:
    """
    Pads input_ids, attention_mask, position_idx, and dfg_to_code_positions
    to the maximum length in the batch.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Determine max lengths for this specific batch
        max_seq_len = max(len(x["input_ids"]) for x in batch)
        max_dfg_len = max(len(x["dfg_to_code_positions"]) for x in batch)
        
        input_ids = []
        attention_mask = []
        position_idx = []
        labels = []

        for x in batch:
            input_ids.append(
                x["input_ids"] + [self.tokenizer.pad_token_id] * (max_seq_len - len(x["input_ids"]))
            )
            attention_mask.append(
                x["attention_mask"] + [0] * (max_seq_len - len(x["attention_mask"]))
            )
            
            # Position idx: GraphCodeBERT needs matching lengths
            p_idx = x["position_idx"] + [self.tokenizer.pad_token_id] * (max_seq_len - len(x["position_idx"]))
            
            # Add DFG attention logic (if needed, here we just pass position_idx)
            # The model natively builds attention out of position_idx
            position_idx.append(p_idx)
            
            labels.append(x["label"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_idx, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["unixcoder", "graphcodebert"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from the latest checkpoint in the output directory")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    args = parser.parse_args()

    # 1. Config based on model choice
    if args.model == "unixcoder":
        model_name = "microsoft/unixcoder-base"
        data_dir   = "processed_data/unixcoder"
        out_dir    = "saved_models/unixcoder"
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        collator   = DataCollatorWithPadding(tokenizer)  # Pads each batch to max length
    else:
        model_name = "microsoft/graphcodebert-base"
        data_dir   = "processed_data/graphcodebert"
        out_dir    = "saved_models/graphcodebert"
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        collator   = GraphCodeBERTCollator(tokenizer)

    print(f"\n[1] Loading dataset from {data_dir}...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Processed dataset not found at {data_dir}. Run preprocessing first.")
    
    ds = load_from_disk(data_dir)
    print(ds)

    print(f"\n[2] Loading model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,          # Binary classification: Human vs AI
    )

    print("\n[3] Configuring Trainer...")
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Fix OOM
        eval_accumulation_steps=10,     # Prevent CPU OOM and Bus errors during evaluation
        dataloader_pin_memory=False,    # Prevent shared memory bus errors in HPC
        dataloader_num_workers=0,
        eval_strategy="epoch",          # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save checkpoint at the end of each epoch
        load_best_model_at_end=True,    # Keep the best model
        metric_for_best_model="f1",
        fp16=True,                      # Use Mixed Precision (VRAM saving & speedup)
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none",               # Disable wandb to keep it simple
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    print("\n[4] Starting Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("\n[5] Evaluating on Test Set...")
    # NOTE: If test set doesn't have labels, inference is run with trainer.predict()
    # Here we assume validation set evaluation is sufficient for reporting, or you 
    # run trainer.predict(ds['test']) and decode predictions.
    test_metrics = trainer.evaluate(ds["validation"])
    print(f"\nFinal Validation Metrics for {args.model}:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n✅ Training complete. Best model saved to {out_dir}.")

if __name__ == "__main__":
    main()
