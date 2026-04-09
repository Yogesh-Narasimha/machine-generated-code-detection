"""
src/evaluation/generator_analysis.py
======================================
Per-generator detection rate analysis — answers RQ4.

Which LLM families produce code that is hardest to detect?
This experiment is NOT in the CoDet-M4 paper and is a novel contribution.

Requires:
  - test_sample.parquet (has 'generator' column with labels)
  - CatBoost predictions saved by tfidf_paths.py or shap_analysis.py
    at: data/test_preds_catboost.npy

Output structure:
  results_output/
  └── generator_analysis/
      ├── reports/
      │   └── generator_summary.txt
      ├── gen_detection_rate.png
      ├── gen_language_heatmap.png
      └── generator_results.csv

Usage:
  python -m src.evaluation.generator_analysis
  python -m src.evaluation.generator_analysis --preds_path data/test_preds_catboost.npy
"""

import os
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sklearn.metrics import f1_score, recall_score, average_precision_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_gen_name(name: str) -> str:
    """Shorten 'microsoft/Phi-3-medium' -> 'Phi-3-medium'."""
    name = str(name).strip()
    if "/" in name:
        name = name.split("/")[-1]
    return name


def load_data(data_dir: str, preds_path: str) -> pd.DataFrame:
    """
    Load test_sample with real model predictions attached.

    If predictions file does not exist, exits with a clear message.
    """
    test_df = pd.read_parquet(os.path.join(data_dir, "test_sample.parquet"))
    log.info("Loaded test_sample: %d rows", len(test_df))

    if not os.path.exists(preds_path):
        log.error(
            "Predictions file not found: %s\n"
            "Run tfidf_paths.py first:\n"
            "  python -m src.feature_extraction.tfidf_paths --mode all\n"
            "Then re-run this script.",
            preds_path,
        )
        raise FileNotFoundError(
            f"Predictions file not found: {preds_path}\n"
            "Run: python -m src.feature_extraction.tfidf_paths --mode all"
        )

    preds = np.load(preds_path)
    if len(preds) != len(test_df):
        raise ValueError(
            f"Predictions length {len(preds)} != test_sample length {len(test_df)}"
        )

    test_df["pred"]      = preds
    test_df["correct"]   = (test_df["pred"] == test_df["label"]).astype(int)
    test_df["gen_clean"] = test_df["generator"].apply(clean_gen_name)
    log.info("Overall accuracy: %.3f", test_df["correct"].mean())
    return test_df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def per_generator_stats(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detection rate per generator on machine-generated samples only.

    Detection rate = fraction of Machine samples correctly predicted as Machine.
    = recall for the Machine class per generator.
    """
    machine_df = test_df[test_df["label"] == 1].copy()
    if machine_df.empty:
        log.warning("No machine-generated samples in test_df.")
        return pd.DataFrame()

    stats = (
        machine_df
        .groupby("gen_clean")
        .agg(
            n_samples       = ("label", "count"),
            detection_rate  = ("correct", "mean"),
            languages       = ("language", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values("detection_rate", ascending=True)
    )

    print("\n" + "="*65)
    print("PER-GENERATOR DETECTION RATES (Machine samples only)")
    print("="*65)
    print(f"{'Generator':<35} {'N':>5} {'DetRate':>9} {'Languages'}")
    print("-"*65)
    for _, row in stats.iterrows():
        bar = "█" * int(row["detection_rate"] * 20)
        print(f"{row['gen_clean']:<35} {row['n_samples']:>5} "
              f"{row['detection_rate']:>9.3f}  {bar}")
    return stats


def plot_detection_rates(stats: pd.DataFrame, out_dir: str) -> None:
    """Horizontal bar chart of detection rates per generator."""
    if stats.empty:
        return

    fig, ax = plt.subplots(figsize=(11, max(5, len(stats) * 0.55)))

    colors = ["#E53935" if r < 0.5 else "#43A047"
              for r in stats["detection_rate"]]

    bars = ax.barh(range(len(stats)), stats["detection_rate"],
                   color=colors, edgecolor="white", height=0.6)

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{row['detection_rate']:.2f} (n={row['n_samples']})",
                va="center", fontsize=9)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels(stats["gen_clean"], fontsize=9)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5,
               label="50% detection threshold")
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Detection Rate (recall on Machine samples)")
    ax.set_title(
        "Per-Generator Detection Rates\n"
        "Red = hard to detect (<50%) | Green = easy to detect (>50%)\n"
        "Novel analysis not in CoDet-M4 paper",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gen_detection_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


def plot_generator_language_heatmap(test_df: pd.DataFrame, out_dir: str) -> None:
    """
    Heatmap: rows = generators, columns = languages, values = detection rate.
    Shows whether some generators are easy in one language but hard in another.
    """
    machine_df = test_df[test_df["label"] == 1].copy()
    if machine_df.empty:
        return

    pivot = machine_df.pivot_table(
        index="gen_clean",
        columns="language",
        values="correct",
        aggfunc="mean",
    ).fillna(0)

    # Keep generators with at least 3 samples
    gen_counts = machine_df.groupby("gen_clean").size()
    valid_gens  = gen_counts[gen_counts >= 3].index
    pivot       = pivot.loc[pivot.index.isin(valid_gens)]

    if pivot.empty:
        log.warning("Not enough samples for heatmap.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                    max(5, len(pivot) * 0.6)))

    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Detection Rate"})
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Detection Rate")
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i,j]:.2f}",
                        ha="center", va="center", fontsize=8)

    ax.set_title("Detection Rate: Generator × Language\n"
                 "(1.0 = always detected | 0.0 = always missed)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Generator")
    ax.set_xlabel("Language")
    plt.tight_layout()

    path = os.path.join(out_dir, "gen_language_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


def save_summary(stats: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> None:
    """Save a text summary of the generator analysis findings."""
    if stats.empty:
        return

    hardest  = stats.iloc[0]
    easiest  = stats.iloc[-1]
    above_50 = (stats["detection_rate"] >= 0.5).sum()
    below_50 = (stats["detection_rate"] < 0.5).sum()

    summary = f"""GENERATOR ANALYSIS SUMMARY
{'='*65}
Total machine-generated samples: {(test_df['label']==1).sum()}
Total generators found:          {len(stats)}

Easy to detect (>=50%):  {above_50} generators
Hard to detect (<50%):   {below_50} generators

Hardest to detect:
  Generator     : {hardest['gen_clean']}
  Detection rate: {hardest['detection_rate']:.3f}
  Samples       : {hardest['n_samples']}
  Languages     : {hardest['languages']}

Easiest to detect:
  Generator     : {easiest['gen_clean']}
  Detection rate: {easiest['detection_rate']:.3f}
  Samples       : {easiest['n_samples']}
  Languages     : {easiest['languages']}

INTERPRETATION FOR REPORT (RQ4):
  Per-generator analysis reveals that not all LLMs produce equally
  detectable code. {hardest['gen_clean']} is the hardest to detect
  (rate={hardest['detection_rate']:.2f}), suggesting its output most closely
  mimics human structural patterns. {easiest['gen_clean']} is the easiest
  (rate={easiest['detection_rate']:.2f}), likely producing highly templated
  code with clear LLM stylistic markers (function wrappers, comments).

Full results:
{stats[['gen_clean','n_samples','detection_rate','languages']].to_string(index=False)}
"""

    rdir = os.path.join(out_dir, "reports")
    os.makedirs(rdir, exist_ok=True)
    path = os.path.join(rdir, "generator_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args) -> None:
    """Full generator analysis pipeline."""
    out_dir = os.path.join(args.output_dir, "generator_analysis")
    os.makedirs(out_dir, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    test_df = load_data(args.data_dir, args.preds_path)
    stats   = per_generator_stats(test_df)

    if not stats.empty:
        plot_detection_rates(stats, out_dir)
        plot_generator_language_heatmap(test_df, out_dir)
        save_summary(stats, test_df, out_dir)

        csv_path = os.path.join(out_dir, "generator_results.csv")
        stats.to_csv(csv_path, index=False)
        log.info("Saved: %s", csv_path)

    log.info("Done — outputs in %s", out_dir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-generator detection rate analysis for RQ4.")
    p.add_argument("--data_dir",    default="data/task_a")
    p.add_argument("--output_dir",  default="results_output",
                   help="Root output dir. Saves to output_dir/generator_analysis/")
    p.add_argument("--preds_path",  default="data/test_preds_catboost.npy",
                   help="Path to saved CatBoost predictions on test_sample.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
