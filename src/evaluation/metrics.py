"""
src/evaluation/metrics.py
==========================
Central metrics module — imported by all other scripts.

Computes a consistent set of metrics across all experiments:
  accuracy, binary_f1, macro_f1, macro_precision, macro_recall,
  roc_auc, pr_auc, mcc, fnr, per_language_f1

Why PR-AUC matters for this task:
  test_sample is 22% Machine / 78% Human.
  ROC-AUC is 0.5 for a random classifier regardless of imbalance.
  PR-AUC is equal to the class prevalence (0.22) for a random classifier.
  PR-AUC is therefore a more honest metric on the OOD imbalanced set.

Usage:
  from src.evaluation.metrics import compute_metrics, per_language_breakdown
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
)

log = logging.getLogger(__name__)

SEEN_LANGS   = {"Python", "Java", "C++"}
UNSEEN_LANGS = {"Go", "PHP", "C#", "C", "JavaScript"}


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_probs: np.ndarray) -> dict:
    """
    Compute the full metric set used consistently across all scripts.

    Metrics:
      accuracy        Overall correct fraction. Misleading when imbalanced.
      binary_f1       F1 for the Machine class (positive class = 1).
      macro_f1        Mean of Human-F1 and Machine-F1. Primary SemEval metric.
      macro_precision Mean per-class precision.
      macro_recall    Mean per-class recall.
      roc_auc         Area under ROC curve. Threshold-independent.
                      Baseline (random) = 0.50 regardless of imbalance.
      pr_auc          Area under Precision-Recall curve.
                      Baseline (random) = class prevalence (~0.22 on OOD).
                      More honest than ROC-AUC when Machine is the minority.
      mcc             Matthews Correlation Coefficient [-1, +1].
                      Near-zero = predictions no better than random.
                      Better than accuracy for imbalanced evaluation.
      fnr             False Negative Rate = missed Machine / all Machine.
                      How much LLM code slips through undetected.

    Args:
        y_true:  True labels array.
        y_pred:  Predicted labels array.
        y_probs: Probability scores for the positive class (Machine).

    Returns:
        Dict of metric_name -> float.
    """
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "binary_f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro",     zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "roc_auc":         float(roc_auc_score(y_true, y_probs)),
        "pr_auc":          float(average_precision_score(y_true, y_probs)),
        "mcc":             float(matthews_corrcoef(y_true, y_pred)),
        "fnr":             float(1 - recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def evaluate_and_save(model, X, y: np.ndarray,
                      name: str, out_dir: str,
                      test_df: pd.DataFrame = None) -> dict:
    """
    Evaluate a model, print results, save a .txt report, and optionally
    compute per-language breakdown.

    Args:
        model:    Trained sklearn/catboost classifier.
        X:        Feature matrix (sparse or dense).
        y:        True labels.
        name:     Report name (used for filename and display).
        out_dir:  Directory to save reports. Reports go to out_dir/reports/.
        test_df:  If provided, also compute per-language breakdown.
                  Must have a 'language' column aligned with X rows.

    Returns:
        Dict of metric_name -> float.
    """
    preds = model.predict(X)
    probs = (model.predict_proba(X)[:, 1]
             if hasattr(model, "predict_proba")
             else model.decision_function(X))

    m      = compute_metrics(y, preds, probs)
    report = classification_report(y, preds, target_names=["Human", "Machine"])

    # Print
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    for k, v in m.items():
        print(f"  {k:<20} : {v:.4f}")
    print(f"\n{report}")

    # Save text report
    rdir = os.path.join(out_dir, "reports")
    os.makedirs(rdir, exist_ok=True)
    safe = name.replace(" ", "_").replace("(","").replace(")","").replace("/","_")
    with open(os.path.join(rdir, f"{safe}_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"{name}\n{'='*55}\n")
        for k, v in m.items():
            f.write(f"{k:<20}: {v:.4f}\n")
        f.write(f"\n{report}")
    log.info("Saved report: %s/%s_report.txt", rdir, safe)

    # Per-language breakdown
    if test_df is not None:
        lang_df = per_language_breakdown(preds, probs, y, test_df, name, out_dir)
        m["per_language"] = lang_df.to_dict(orient="records")

    return m


# ---------------------------------------------------------------------------
# Per-language breakdown
# ---------------------------------------------------------------------------

def per_language_breakdown(preds: np.ndarray, probs: np.ndarray,
                            y: np.ndarray, df: pd.DataFrame,
                            model_name: str, out_dir: str) -> pd.DataFrame:
    """
    Compute Macro F1, FNR, and PR-AUC per language on the OOD test set.

    Saves results to out_dir/reports/per_language_{model_name}.csv.

    Args:
        preds:      Predicted labels aligned with df rows.
        probs:      Probability scores for Machine class.
        y:          True labels aligned with df rows.
        df:         DataFrame with 'language' column.
        model_name: Name for display and filename.
        out_dir:    Root output dir for this script.

    Returns:
        DataFrame with per-language metrics.
    """
    rows = []
    for lang in sorted(df["language"].unique()):
        mask   = df["language"].values == lang
        y_l    = y[mask]
        p_l    = preds[mask]
        pr_l   = probs[mask]
        n      = len(y_l)
        n_mach = y_l.sum()
        setting = "Seen" if lang in SEEN_LANGS else "Unseen"

        if len(np.unique(y_l)) < 2:
            log.debug("Skipping %s — only one class present (%d samples)", lang, n)
            continue

        rows.append({
            "language":    lang,
            "setting":     setting,
            "n_samples":   n,
            "pct_machine": float(y_l.mean()),
            "macro_f1":    float(f1_score(y_l, p_l, average="macro",  zero_division=0)),
            "pr_auc":      float(average_precision_score(y_l, pr_l)),
            "fnr":         float(1 - recall_score(y_l, p_l, pos_label=1, zero_division=0)),
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        log.warning("per_language_breakdown: no languages with two classes present.")
        return df_out

    # Print
    print(f"\nPer-language breakdown — {model_name}")
    print(f"{'Language':<14} {'Setting':<8} {'n':>5} {'%Mach':>7} "
          f"{'MacroF1':>9} {'PR-AUC':>8} {'FNR':>7}")
    print("-" * 62)
    for _, row in df_out.iterrows():
        seen_mark = "" if row["setting"] == "Seen" else " *"
        print(f"{row['language']:<14} {row['setting']:<8} "
              f"{row['n_samples']:>5} {row['pct_machine']:>7.1%} "
              f"{row['macro_f1']:>9.4f} {row['pr_auc']:>8.4f} "
              f"{row['fnr']:>7.4f}{seen_mark}")
    print("* = unseen language (OOD)")

    # Save CSV
    rdir = os.path.join(out_dir, "reports")
    os.makedirs(rdir, exist_ok=True)
    safe = model_name.replace(" ", "_").replace("(","").replace(")","")
    path = os.path.join(rdir, f"per_language_{safe}.csv")
    df_out.to_csv(path, index=False)
    log.info("Saved per-language breakdown: %s", path)

    return df_out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(model, X, y: np.ndarray,
                       name: str, out_dir: str) -> None:
    """
    Plot both ROC and Precision-Recall curves side by side.

    The PR curve is more informative on imbalanced OOD data because:
    - Random ROC-AUC = 0.50  (same regardless of imbalance)
    - Random PR-AUC  = class_prevalence (~0.22 on test_sample)
    """
    probs = (model.predict_proba(X)[:, 1]
             if hasattr(model, "predict_proba")
             else model.decision_function(X))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ROC and PR Curves — {name}", fontsize=11, fontweight="bold")

    # ROC
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val      = roc_auc_score(y, probs)
    axes[0].plot(fpr, tpr, label=f"AUC={auc_val:.3f}", color="#1976D2", linewidth=2)
    axes[0].plot([0,1],[0,1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # PR
    prec, rec, _ = precision_recall_curve(y, probs)
    pr_auc_val   = average_precision_score(y, probs)
    baseline_pr  = y.mean()
    axes[1].plot(rec, prec, label=f"PR-AUC={pr_auc_val:.3f}", color="#E53935", linewidth=2)
    axes[1].axhline(baseline_pr, color="k", linestyle="--", alpha=0.4,
                    label=f"Random baseline ({baseline_pr:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve\n(more informative on imbalanced OOD data)")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    safe  = name.replace(" ","_").replace("(","").replace(")","")
    path  = os.path.join(out_dir, f"roc_pr_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved ROC+PR curves: %s", path)


def save_results_csv(val_results: dict, test_results: dict,
                     out_dir: str, filename: str = "results.csv") -> None:
    """Save combined val + OOD results to a CSV file."""
    rows = (
        [{"split": "Val",      "model": n, **v} for n, v in val_results.items()] +
        [{"split": "OOD Test", "model": n, **v} for n, v in test_results.items()]
    )
    # Drop nested per_language column if present
    clean_rows = []
    for r in rows:
        clean_rows.append({k: v for k, v in r.items() if not isinstance(v, list)})

    df   = pd.DataFrame(clean_rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    log.info("Saved results: %s", path)
    print(f"\nResults summary:\n{df[['split','model','macro_f1','pr_auc','mcc','fnr']].to_string(index=False)}")
