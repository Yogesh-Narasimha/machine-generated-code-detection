"""
src/feature_extraction/tfidf_paths.py

AST Path TF-IDF feature extraction for machine-generated code detection.

How it works
  1. Parse each code snippet into an AST using tree-sitter.
  2. Walk the AST and record every parent->child node-type transition.
  3. Join transitions as a space-separated string.
  4. Fit TF-IDF on training strings -> 2000-dim sparse matrix.
  5. Train Random Forest, SVM, CatBoost.
  6. Evaluate on BOTH val (in-distribution) AND test_sample (OOD).

The error-node problem
  tree-sitter inserts 'error' nodes when it cannot parse code.
  Human competitive code (CodeForces) is full of Python 2 syntax,
  compact tuple unpacking, inline loop bodies etc. that trip the parser.
  LLM code is always syntactically clean.

  From your actual run (CatBoost feature importance):
    Rank 1  error -> identifier          24.28  <- noise
    Rank 3  module -> error               8.48  <- noise
  41% of top-20 importance = error nodes.

  This collapses on OOD: PHP/Go parsed by a Python parser produces
  all error nodes regardless of authorship -> model predicts Human.

  Fix: --filter_errors strips error-node paths before TF-IDF.
  Remaining 59% of genuine signals:
    module->function_definition  LLM wraps code in a function
    block->comment               LLM adds inline comments
    assignment->call             LLM prefers x = func() patterns
    print_statement->print       Python 2 only in human code

  Use --mode compare to run both variants and print a comparison table.

Output structure
  results_output/
  +-- ast_paths/
      +-- reports/
      |   +-- Random_Forest_Val_report.txt
      |   +-- Random_Forest_OOD_Test_report.txt
      |   +-- SVM_Val_report.txt
      |   +-- SVM_OOD_Test_report.txt
      |   +-- CatBoost_Val_report.txt
      |   +-- CatBoost_OOD_Test_report.txt
      +-- roc_curves_val.png
      +-- roc_curves_ood.png
      +-- feature_importance_CatBoost.png
      +-- feature_importance_Random_Forest.png
      +-- ast_paths_results.csv

Usage
  python -m src.feature_extraction.tfidf_paths --mode all --filter_errors
  python -m src.feature_extraction.tfidf_paths --mode all
  python -m src.feature_extraction.tfidf_paths --mode compare
  python -m src.feature_extraction.tfidf_paths --mode train --filter_errors
  python -m src.feature_extraction.tfidf_paths --mode explain
"""

import os
import argparse
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
    average_precision_score,
    classification_report, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
_PARSERS: dict = {}

def get_parsers() -> dict:
    """Build and cache tree-sitter parsers once."""
    global _PARSERS
    if not _PARSERS:
        from tree_sitter_languages import get_parser
        for lang, ts in [("Python","python"),("Java","java"),("C++","cpp")]:
            try:
                _PARSERS[lang] = get_parser(ts)
                log.info("Parser loaded: %s", lang)
            except Exception as e:
                log.warning("Parser unavailable for %s: %s", lang, e)
    return _PARSERS


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------
def extract_ast_paths(code: str, language: str,
                      filter_errors: bool = False) -> str:
    """
    Walk the AST and collect parent->child node-type transitions.

    Args:
        code:          Raw source code.
        language:      Python / Java / C++.
        filter_errors: If True, drop any transition where either node
                       type contains 'error'. This removes parse-error
                       noise and forces learning on genuine syntax.
    Returns:
        Space-separated transition string. Empty string on failure.
    """
    parser = get_parsers().get(language)
    if parser is None:
        return ""
    try:
        root = parser.parse(bytes(code, "utf8")).root_node
    except Exception:
        return ""

    paths, stack = [], [(root, None)]
    while stack:
        node, parent = stack.pop()
        if parent is not None:
            if filter_errors and (
                "error" in parent.lower() or "error" in node.type.lower()
            ):
                for c in node.children:
                    stack.append((c, node.type))
                continue
            paths.append(f"{parent}->{node.type}")
        for c in node.children:
            stack.append((c, node.type))
    return " ".join(paths)


def parallel_extract(df: pd.DataFrame, filter_errors: bool,
                     max_workers: int, desc: str) -> list:
    """Extract paths for every row using a thread pool."""
    tasks = list(zip(df["code"], df["language"]))
    fn    = lambda x: extract_ast_paths(x[0], x[1], filter_errors)
    docs  = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for d in tqdm(ex.map(fn, tasks), total=len(tasks), desc=desc):
            docs.append(d)
    return docs


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cache_path(base: str, filter_errors: bool) -> str:
    stem, ext = os.path.splitext(base)
    return stem + ("_filtered" if filter_errors else "") + ext


def load_or_compute(df, base_cache, filter_errors, max_workers, desc):
    """Load .npy cache or compute and save it."""
    path = _cache_path(base_cache, filter_errors)
    if os.path.exists(path):
        log.info("Loading cache: %s", path)
        return np.load(path, allow_pickle=True)
    log.info("Computing %s (filter_errors=%s)...", desc, filter_errors)
    docs = parallel_extract(df, filter_errors, max_workers, desc)
    arr  = np.array(docs, dtype=object)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, arr)
    log.info("Saved: %s", path)
    return arr


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
def build_tfidf(train_docs, val_docs, test_docs,
                max_features=2000, min_df=5):
    """
    Fit TF-IDF on training docs, transform all splits.

    token_pattern=r"[^ ]+" keeps 'parent->child' as one token.
    TF-IDF up-weights rare-but-consistent paths (strong discriminative
    signal) and down-weights universal paths like module->import.
    """
    vec = TfidfVectorizer(max_features=max_features,
                          token_pattern=r"[^ ]+", min_df=min_df)
    Xtr = vec.fit_transform(train_docs)
    Xv  = vec.transform(val_docs)
    Xte = vec.transform(test_docs)
    log.info("TF-IDF vocab=%d | train=%s val=%s test=%s",
             len(vec.vocabulary_), Xtr.shape, Xv.shape, Xte.shape)
    return vec, Xtr, Xv, Xte


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_models(X_train, y_train) -> dict:
    """Train RF, SVM, CatBoost. Return {name: model}."""
    log.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight="balanced", n_jobs=-1, random_state=42
    ).fit(X_train, y_train)

    log.info("Training SVM...")
    svm = LinearSVC(
        class_weight="balanced", max_iter=5000
    ).fit(X_train, y_train)

    log.info("Training CatBoost...")
    cat = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1, verbose=100
    ).fit(X_train, y_train)

    return {"Random Forest": rf, "SVM": svm, "CatBoost": cat}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _get_probs(model, X):
    return (model.predict_proba(X)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X))


def evaluate_model(model, X, y, name: str, out_dir: str) -> dict:
    """
    Compute metrics and save a text report to out_dir/reports/.

    Metrics:
      macro_f1        Primary SemEval metric
      accuracy        Overall correct fraction
      binary_f1       F1 for the Machine class
      macro_precision Mean per-class precision
      macro_recall    Mean per-class recall
      roc_auc         Area under ROC curve (threshold-independent)
      mcc             Matthews Correlation Coefficient
                      Better than accuracy for imbalanced data.
                      test_sample is 22% Machine vs 52% in train.
      fnr             False Negative Rate = missed Machine / all Machine
                      How much LLM code slips through undetected.
    """
    preds = model.predict(X)
    probs = _get_probs(model, X)

    m = dict(
        accuracy        = accuracy_score(y, preds),
        binary_f1       = f1_score(y, preds, zero_division=0),
        macro_f1        = f1_score(y, preds, average="macro",     zero_division=0),
        macro_precision = precision_score(y, preds, average="macro", zero_division=0),
        macro_recall    = recall_score(y, preds, average="macro",    zero_division=0),
        roc_auc         = roc_auc_score(y, probs),
        pr_auc          = average_precision_score(y, probs),
        mcc             = matthews_corrcoef(y, preds),
        fnr             = 1 - recall_score(y, preds, pos_label=1, zero_division=0),
    )
    report = classification_report(y, preds, target_names=["Human","Machine"])

    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    for k, v in m.items():
        print(f"  {k:<20} : {v:.4f}")
    print(f"\n{report}")

    rdir = os.path.join(out_dir, "reports")
    os.makedirs(rdir, exist_ok=True)
    safe = name.replace(" ","_").replace("(","").replace(")","")
    with open(os.path.join(rdir, f"{safe}_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"{name}\n{'='*55}\n")
        for k, v in m.items():
            f.write(f"{k:<20}: {v:.4f}\n")
        f.write(f"\n{report}")
    log.info("Saved: %s/%s_report.txt", rdir, safe)
    return m


def run_evaluation(models, Xv, Xte, yv, yte, out_dir):
    """
    Evaluate every model on BOTH splits.

    val.parquet  100K, same distribution as train.
                 Compare with CoDet-M4 paper Table 2.
    test_sample  1K, 8 languages, 22% Machine.
                 OOD robustness evaluation.
    """
    log.info("\n--- Validation (in-distribution, val.parquet) ---")
    vr = {n: evaluate_model(m, Xv,  yv,  f"{n} (Val)",      out_dir)
          for n, m in models.items()}

    log.info("\n--- OOD test_sample (unseen languages + imbalanced) ---")
    log.info("22%% Machine vs 52%% in train. Low scores are expected.")
    tr = {n: evaluate_model(m, Xte, yte, f"{n} (OOD Test)", out_dir)
          for n, m in models.items()}
    return vr, tr


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc(models, Xv, yv, Xte, yte, out_dir):
    """Two separate ROC plots: val and OOD test."""
    os.makedirs(out_dir, exist_ok=True)
    for label, X, y, fname in [
        ("Validation (In-Distribution)", Xv,  yv,  "val"),
        ("OOD Test (Unseen Languages)",  Xte, yte, "ood"),
    ]:
        plt.figure(figsize=(7, 6))
        for name, model in models.items():
            probs = _get_probs(model, X)
            fpr, tpr, _ = roc_curve(y, probs)
            plt.plot(fpr, tpr,
                     label=f"{name} (AUC={roc_auc_score(y,probs):.3f})")
        plt.plot([0,1],[0,1],"k--",alpha=0.4,label="Random baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - AST Paths\n{label}")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, f"roc_curves_{fname}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", path)


def plot_importance(models, vec, out_dir, top_n=20):
    """
    Plot and print top-N feature importance for RF and CatBoost.

    WITHOUT --filter_errors: error->* features dominate ranks 1 and 3.
    WITH    --filter_errors: only genuine structural signals appear.
    Both are informative for the report.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = vec.get_feature_names_out()

    for mname, model in models.items():
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            imp = model.get_feature_importance()
        else:
            continue

        idx       = np.argsort(imp)[::-1][:top_n]
        top_names = [names[i].replace("->", " -> ") for i in idx]
        top_vals  = imp[idx]

        plt.figure(figsize=(11, 7))
        plt.barh(range(top_n), top_vals[::-1], color="#1976D2", alpha=0.85)
        plt.yticks(range(top_n), top_names[::-1], fontsize=9)
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} AST Path Features - {mname}")
        plt.tight_layout()
        path = os.path.join(
            out_dir, f"feature_importance_{mname.replace(' ','_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved: %s", path)

        print(f"\nTop {top_n} features - {mname}")
        print(f"{'Rank':<5} {'AST Path':<50} {'Importance':>10}")
        print("-" * 67)
        for rank, (n, v) in enumerate(zip(top_names, top_vals), 1):
            flag = "  <- error node (noise)" if "error" in n.lower() else ""
            print(f"{rank:<5} {n:<50} {v:>10.4f}{flag}")


def save_csv(vr, tr, out_dir):
    """Save combined val + OOD results to CSV."""
    rows = (
        [{"Model": f"{n} (Val)",      **v} for n, v in vr.items()] +
        [{"Model": f"{n} (OOD Test)", **v} for n, v in tr.items()]
    )
    df   = pd.DataFrame(rows)
    path = os.path.join(out_dir, "ast_paths_results.csv")
    df.to_csv(path, index=False)
    log.info("Saved: %s", path)
    print(f"\nResults summary:\n{df.to_string(index=False)}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args) -> tuple:
    """Full pipeline. Everything saved to results_output/ast_paths/."""
    out = os.path.join(args.output_dir, "ast_paths")
    os.makedirs(out, exist_ok=True)
    log.info("Output dir   : %s", out)
    log.info("filter_errors: %s | max_features: %d",
             args.filter_errors, args.max_features)

    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val_df   = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir,
                                            "test_sample.parquet"))
    log.info("Train=%d | Val=%d | Test(OOD)=%d",
             len(train_df), len(val_df), len(test_df))
    log.info("Machine%% - Train:%.1f  Test:%.1f",
             100*train_df["label"].mean(), 100*test_df["label"].mean())

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    kw = dict(filter_errors=args.filter_errors, max_workers=args.max_workers)
    td = load_or_compute(train_df,
                         os.path.join(args.cache_dir,"train_docs.npy"),
                         desc="train", **kw)
    vd = load_or_compute(val_df,
                         os.path.join(args.cache_dir,"val_docs.npy"),
                         desc="val",   **kw)
    ed = load_or_compute(test_df,
                         os.path.join(args.cache_dir,"test_docs.npy"),
                         desc="test",  **kw)

    vec, Xtr, Xv, Xte = build_tfidf(td, vd, ed,
                                     max_features=args.max_features)

    models = {}
    vr, tr = {}, {}
    if args.mode in ("train", "all"):
        models  = train_models(Xtr, y_train)
        vr, tr  = run_evaluation(models, Xv, Xte, y_val, y_test, out)
        plot_roc(models, Xv, y_val, Xte, y_test, out)
        save_csv(vr, tr, out)

        # Save CatBoost test predictions for generator_analysis.py
        preds_path = os.path.join(args.cache_dir, "test_preds_catboost.npy")
        np.save(preds_path, models["CatBoost"].predict(Xte))
        log.info("Saved test predictions: %s", preds_path)

    if args.mode in ("explain", "all"):
        plot_importance(models, vec, out)

    log.info("Done - outputs in %s", out)
    return models, vec, vr, tr


def run_compare(args):
    """Run pipeline twice and print a side-by-side comparison table."""
    store = {}
    for flag, label in [(False,"standard (with errors)"),
                        (True, "filtered (no errors)")]:
        log.info("\n" + "="*60)
        log.info("Variant: %s", label)
        log.info("="*60)
        args.filter_errors = flag
        _, _, vr, tr = run_pipeline(args)
        store[label] = {"val": vr, "ood": tr}

    print("\n" + "="*72)
    print("COMPARISON: error nodes kept vs removed")
    print("="*72)
    print(f"{'Model':<25} {'Std Val':>9} {'Flt Val':>9}"
          f" {'Std OOD':>9} {'Flt OOD':>9} {'OOD gain':>9}")
    print("-"*72)
    for m in ["Random Forest", "SVM", "CatBoost"]:
        sv = store["standard (with errors)"]["val" ].get(m,{}).get("macro_f1",0)
        fv = store["filtered (no errors)" ]["val" ].get(m,{}).get("macro_f1",0)
        so = store["standard (with errors)"]["ood" ].get(m,{}).get("macro_f1",0)
        fo = store["filtered (no errors)" ]["ood" ].get(m,{}).get("macro_f1",0)
        print(f"{m:<25} {sv:>9.4f} {fv:>9.4f} {so:>9.4f}"
              f" {fo:>9.4f} {fo-so:>+9.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="AST Path TF-IDF pipeline.")
    p.add_argument("--data_dir",      default="data/task_a")
    p.add_argument("--output_dir",    default="results_output",
                   help="Root dir. Outputs saved to output_dir/ast_paths/")
    p.add_argument("--cache_dir",     default="data")
    p.add_argument("--max_features",  type=int, default=2000)
    p.add_argument("--max_workers",   type=int, default=8)
    p.add_argument("--filter_errors", action="store_true",
                   help="Strip error-node paths. Recommended for OOD.")
    p.add_argument("--mode",
                   choices=["train","explain","all","compare"],
                   default="all")
    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()
    if args.mode == "compare":
        run_compare(args)
    else:
        run_pipeline(args)
