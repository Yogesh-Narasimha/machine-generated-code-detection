"""
src/feature_extraction/baseline_task_a.py
==========================================
TF-IDF Baseline for Machine-Generated Code Detection (Task A).

What this module does:
    Implements three TF-IDF baseline variants with Logistic Regression to
    establish lower-bound performance for machine-generated code detection.
    All variants are evaluated both in-distribution (validation set) and
    out-of-distribution (OOD test_sample with unseen languages + label shift).

    Variants:
        1. Word unigram TF-IDF (vanilla baseline)
           - Standard tokenisation, 150k features, no class weighting.
           - Establishes the simplest possible baseline.

        2. Word unigram TF-IDF + balanced training (improved baseline)
           - Adds bigrams, sublinear_tf, min_df=5, class_weight="balanced".
           - Addresses label imbalance; slightly better in-distribution.

        3. Character n-gram TF-IDF (3–5 chars)
           - Language-agnostic surface features; captures punctuation style.
           - Attempts better OOD transfer by avoiding vocabulary mismatch.

        4. Character TF-IDF on language-balanced training subset
           - Samples equal counts from Python / C++ / Java.
           - Tests whether per-language balancing helps OOD robustness.

    Key finding:
        All TF-IDF variants achieve high in-distribution accuracy (~90–93%)
        but collapse on OOD test_sample (~28–38%). Vocabulary mismatch between
        training languages (Python/Java/C++) and OOD languages drives the
        failure. Balanced-language training did NOT improve OOD performance.

    Why this matters:
        These baselines motivate moving to structure-based features (AST
        scalars, AST paths) that are less language-specific.

Usage:
    # Full pipeline — all variants
    python -m src.feature_extraction.baseline_task_a --mode all

    # Only vanilla baseline
    python -m src.feature_extraction.baseline_task_a --mode vanilla

    # Only character TF-IDF variants
    python -m src.feature_extraction.baseline_task_a --mode char

Arguments:
    --data_dir      Path to parquet files. Default: data/task_a
    --output_dir    Root output directory. Default: results_output
                    Results saved to: results_output/baseline/
    --mode          vanilla | improved | char | balanced | all. Default: all
"""

import os
import argparse
import logging
import warnings

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and OOD test splits from parquet files.

    Returns:
        (train_df, val_df, eval_df)
    """
    train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    val_df   = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    eval_df  = pd.read_parquet(os.path.join(data_dir, "test_sample.parquet"))

    log.info("Train size: %d", len(train_df))
    log.info("Validation size: %d", len(val_df))
    log.info("Test (OOD) size: %d", len(eval_df))

    return train_df, val_df, eval_df


# ---------------------------------------------------------------------------
# 2. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    df: pd.DataFrame,
    split_name: str,
    output_dir: str,
) -> dict:
    """
    Evaluate a fitted model on any split and print / save a report.

    Args:
        model:       Fitted LogisticRegression.
        vectorizer:  Fitted TfidfVectorizer (already applied to df['code']).
                     Pass None if X is already transformed.
        df:          DataFrame with 'code' and 'label' columns.
        split_name:  Label for logging / filename (e.g. "Vanilla_Val").
        output_dir:  Directory to save the .txt report.

    Returns:
        Dict of metric_name -> float.
    """
    X = vectorizer.transform(df["code"])
    y = df["label"]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy":        accuracy_score(y, preds),
        "binary_f1":       f1_score(y, preds, zero_division=0),
        "macro_f1":        f1_score(y, preds, average="macro",     zero_division=0),
        "macro_precision": precision_score(y, preds, average="macro", zero_division=0),
        "macro_recall":    recall_score(y, preds, average="macro",    zero_division=0),
        "roc_auc":         roc_auc_score(y, probs),
        "pr_auc":          average_precision_score(y, probs),
        "mcc":             matthews_corrcoef(y, preds),
        "fnr":             1 - recall_score(y, preds, pos_label=1, zero_division=0),
    }

    report = classification_report(y, preds, target_names=["Human", "Machine"])

    print(f"\n{'='*55}")
    print(f"  {split_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<20} : {v:.4f}")
    print(f"\n{report}")
    print("Confusion Matrix:")
    cm = confusion_matrix(y, preds)
    print(cm)

    # Save confusion matrix plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Human", "Machine"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(split_name)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{split_name.replace(' ','_')}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved confusion matrix: %s", cm_path)

    os.makedirs(output_dir, exist_ok=True)
    safe_name   = split_name.replace(" ", "_")
    report_path = os.path.join(output_dir, f"{safe_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{split_name}\n{'='*55}\n")
        for k, v in metrics.items():
            f.write(f"{k:<20}: {v:.4f}\n")
        f.write(f"\n{report}")
    log.info("Saved report: %s", report_path)

    return metrics


def per_language_breakdown(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    df: pd.DataFrame,
    split_name: str,
) -> None:
    """Print accuracy and F1 for each language in the given split."""
    X = vectorizer.transform(df["code"])
    df = df.copy()
    df["pred"] = model.predict(X)

    print(f"\nPer-language breakdown — {split_name}:")
    for lang in sorted(df["language"].unique()):
        sub = df[df["language"] == lang]
        acc = accuracy_score(sub["label"], sub["pred"])
        f1  = f1_score(sub["label"], sub["pred"], zero_division=0)
        print(f"  {lang:<10} — Accuracy: {acc:.4f}, F1: {f1:.4f}")


# ---------------------------------------------------------------------------
# 3. Variant 1 — Vanilla word unigram TF-IDF
# ---------------------------------------------------------------------------

def run_vanilla(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    eval_df:  pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Baseline 1: Word unigram TF-IDF + Logistic Regression (no class weighting).

    Configuration:
        - max_features=150000  (limited for 8 GB RAM)
        - ngram_range=(1,1)    (unigrams only)
        - analyzer="word"
        - lowercase=False      (case matters in code, e.g. True vs true)
    """
    log.info("\n=== Variant 1: Vanilla Word Unigram TF-IDF ===")

    vectorizer = TfidfVectorizer(
        max_features=150000,      # limit features for 8 GB RAM
        ngram_range=(1, 1),       # unigram only
        analyzer="word",
        token_pattern=r"\b\w+\b", # keep code tokens
        lowercase=False,          # case matters in code
    )

    log.info("Fitting TF-IDF on training data...")
    X_train = vectorizer.fit_transform(train_df["code"])
    log.info("TF-IDF train shape: %s", X_train.shape)

    y_train = train_df["label"]

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    log.info("Training Logistic Regression...")
    model.fit(X_train, y_train)

    evaluate(model, vectorizer, val_df,  "Vanilla_Val",      output_dir)
    per_language_breakdown(model, vectorizer, val_df,  "Validation")

    log.info("OOD evaluation — test_sample has different languages + label shift.")
    evaluate(model, vectorizer, eval_df, "Vanilla_OOD_Test", output_dir)
    per_language_breakdown(model, vectorizer, eval_df, "OOD Test")


# ---------------------------------------------------------------------------
# 4. Variant 2 — Improved word TF-IDF + balanced class weights
# ---------------------------------------------------------------------------

def run_improved(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    eval_df:  pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Baseline 2: Improved word TF-IDF with bigrams, sublinear TF, class balancing.

    Changes over vanilla:
        - ngram_range=(1,2)       adds bigrams for richer context
        - min_df=5                drops very rare tokens (noise reduction)
        - sublinear_tf=True       log-scales TF counts for better generalisation
        - class_weight="balanced" compensates for 52/48 label imbalance
        - max_features=120000     slightly reduced to accommodate bigrams
    """
    log.info("\n=== Variant 2: Improved Word TF-IDF + Balanced LR ===")

    vectorizer = TfidfVectorizer(
        max_features=120000,       # reduced to accommodate bigrams
        ngram_range=(1, 2),        # unigrams + bigrams
        analyzer="word",
        token_pattern=r"\b\w+\b",
        lowercase=False,
        min_df=5,                  # remove rare tokens
        sublinear_tf=True,         # better generalisation
    )

    log.info("Fitting improved TF-IDF...")
    X_train = vectorizer.fit_transform(train_df["code"])
    log.info("Train shape: %s", X_train.shape)

    y_train = train_df["label"]

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # handle label shift
        n_jobs=-1,
    )
    log.info("Training model...")
    model.fit(X_train, y_train)

    evaluate(model, vectorizer, val_df,  "Improved_Val",      output_dir)
    evaluate(model, vectorizer, eval_df, "Improved_OOD_Test", output_dir)
    per_language_breakdown(model, vectorizer, eval_df, "Improved OOD")


# ---------------------------------------------------------------------------
# 5. Variant 3 — Character n-gram TF-IDF (3–5 chars)
# ---------------------------------------------------------------------------

def run_char(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    eval_df:  pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Baseline 3: Character n-gram TF-IDF (3–5 chars) + balanced Logistic Regression.

    Rationale:
        Character n-grams capture sub-token patterns (indentation style,
        operator spacing, bracket patterns) that are less language-specific
        than full vocabulary tokens. Hypothesis: better OOD transfer because
        style patterns are shared across languages.

    Configuration:
        - analyzer="char"
        - ngram_range=(3, 5)   — 1–2 too noisy, 6+ too sparse
        - max_features=100000  — safe for 8 GB
        - sublinear_tf=True
        - min_df=5
    """
    log.info("\n=== Variant 3: Character TF-IDF (3–5 grams) ===")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=100000,    # safe for 8 GB
        min_df=5,
        sublinear_tf=True,
    )

    log.info("Fitting Character TF-IDF...")
    X_train = vectorizer.fit_transform(train_df["code"])
    log.info("Train shape: %s", X_train.shape)

    y_train = train_df["label"]

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    log.info("Training model...")
    model.fit(X_train, y_train)

    evaluate(model, vectorizer, val_df,  "Char_Val",      output_dir)
    evaluate(model, vectorizer, eval_df, "Char_OOD_Test", output_dir)
    per_language_breakdown(model, vectorizer, eval_df, "Char OOD")


# ---------------------------------------------------------------------------
# 6. Variant 4 — Character TF-IDF on language-balanced training subset
# ---------------------------------------------------------------------------

def run_balanced(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    eval_df:  pd.DataFrame,
    output_dir: str,
    samples_per_language: int = 15000,
) -> None:
    """
    Baseline 4: Character TF-IDF on a language-balanced training subset.

    Motivation:
        Training data is Python-heavy. Equal sampling (15k per language)
        prevents the model from overfitting to Python surface patterns,
        potentially improving OOD transfer to other languages.

    Conclusion (from notebook experiments):
        Balanced training did NOT improve OOD performance. The fundamental
        issue is vocabulary mismatch — unseen languages introduce tokens
        the TF-IDF vocabulary has never seen, regardless of how training
        data is balanced.

    Args:
        samples_per_language: Rows sampled per language. Default: 15000.
    """
    log.info("\n=== Variant 4: Language-Balanced Character TF-IDF ===")

    # ── Build balanced training subset ────────────────────────────────────────
    languages = ["Python", "C++", "Java"]
    balanced_samples = []

    for lang in languages:
        subset = train_df[train_df["language"] == lang]
        n      = min(samples_per_language, len(subset))
        balanced_samples.append(subset.sample(n=n, random_state=42))

    balanced_train_df = pd.concat(balanced_samples)

    log.info("Balanced training size: %d", len(balanced_train_df))
    log.info("Language distribution:\n%s", balanced_train_df["language"].value_counts().to_string())
    log.info(
        "Label distribution:\n%s",
        balanced_train_df["label"].value_counts(normalize=True).to_string(),
    )

    # ── Fit TF-IDF and train ──────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=100000,
        min_df=5,
        sublinear_tf=True,
    )

    log.info("Fitting balanced Character TF-IDF...")
    X_train_bal = vectorizer.fit_transform(balanced_train_df["code"])
    y_train_bal = balanced_train_df["label"]

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    log.info("Training balanced model...")
    model.fit(X_train_bal, y_train_bal)

    evaluate(model, vectorizer, val_df,  "Balanced_Val",      output_dir)
    evaluate(model, vectorizer, eval_df, "Balanced_OOD_Test", output_dir)

    log.info("Finding: Balanced training did NOT improve OOD performance.")
    log.info(
        "Root cause: unseen OOD languages have out-of-vocabulary tokens "
        "regardless of training language balance."
    )


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """
    End-to-end baseline pipeline.

    Output structure:
        results_output/
        └── baseline/
            ├── Vanilla_Val_report.txt
            ├── Vanilla_OOD_Test_report.txt
            ├── Improved_Val_report.txt
            ├── Improved_OOD_Test_report.txt
            ├── Char_Val_report.txt
            ├── Char_OOD_Test_report.txt
            ├── Balanced_Val_report.txt
            └── Balanced_OOD_Test_report.txt
    """
    out_dir = os.path.join(args.output_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    log.info("All outputs will be saved to: %s", out_dir)

    train_df, val_df, eval_df = load_data(args.data_dir)

    if args.mode in ("vanilla", "all"):
        run_vanilla(train_df, val_df, eval_df, out_dir)

    if args.mode in ("improved", "all"):
        run_improved(train_df, val_df, eval_df, out_dir)

    if args.mode in ("char", "all"):
        run_char(train_df, val_df, eval_df, out_dir)

    if args.mode in ("balanced", "all"):
        run_balanced(train_df, val_df, eval_df, out_dir)

    log.info("\nAll outputs saved to: %s", out_dir)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TF-IDF baseline variants for machine-generated code detection."
    )
    p.add_argument(
        "--data_dir",
        default="data/task_a",
        help="Directory with train/val/test_sample parquet files",
    )
    p.add_argument(
        "--output_dir",
        default="results_output",
        help="Root output dir. Script saves to output_dir/baseline/",
    )
    p.add_argument(
        "--mode",
        choices=["vanilla", "improved", "char", "balanced", "all"],
        default="all",
        help="Which variant(s) to run",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
