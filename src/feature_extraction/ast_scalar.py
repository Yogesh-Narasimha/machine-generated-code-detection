"""
src/feature_extraction/ast_scalar.py
======================================
AST Scalar Feature Extraction for Machine-Generated Code Detection.

What this module does:
    Extracts 18 numerical (scalar) structural features from each code snippet
    by analysing its Abstract Syntax Tree (AST). These features capture the
    SHAPE of a program — how deep, how branchy, how complex — rather than
    its specific syntax tokens.

    Features extracted:
        1.  max_depth        - Maximum depth of the AST tree
        2.  avg_depth        - Average node depth
        3.  depth_var        - Variance in node depths
        4.  total_nodes      - Total number of AST nodes
        5.  leaf_nodes       - Number of leaf nodes (no children)
        6.  leaf_ratio       - leaf_nodes / total_nodes
        7.  avg_branch       - Average number of children per node
        8.  max_children     - Maximum children any single node has
        9.  branch_variance  - Variance in children counts
        10. cyclomatic       - Cyclomatic complexity proxy (decision nodes + 1)
        11. decision_nodes   - Count of if/for/while nodes
        12. unique_types     - Number of unique AST node types
        13. entropy          - Shannon entropy of node type distribution
        14. norm_entropy     - Normalised entropy (0-1 scale)
        15. branch_entropy   - Entropy of branching factor distribution
        16. function_calls   - Number of function call nodes
        17. unique_calls     - Number of distinct call node types
        18. internal_nodes   - Nodes with at least one child

    Why scalar features?
        Unlike TF-IDF path features (which capture specific transitions),
        scalar features are LANGUAGE-AGNOSTIC in principle — max_depth and
        cyclomatic complexity mean the same thing in Python, Java, or Go.
        However, in practice they provide weaker discrimination than path
        features because LLMs and humans can produce similarly-shaped trees.

    Known limitation — parse error nodes:
        tree-sitter is error-tolerant: when it cannot parse code correctly
        it inserts 'error' nodes. Human competitive code (CodeForces)
        often uses abbreviated syntax (Python 2 print, tuple unpacking)
        that triggers these errors. LLM code is syntactically clean.
        Scalar features like entropy are therefore influenced by error nodes,
        which partially explains why these features work in-distribution
        but degrade on OOD languages (unseen code produces error nodes
        regardless of authorship, confusing the model).

Usage:
    # Full pipeline: extract features, train models, evaluate, visualise
    python -m src.feature_extraction.ast_scalar --mode all

    # Only extract and cache features (slow, ~5 min)
    python -m src.feature_extraction.ast_scalar --mode extract

    # Only train and evaluate (uses cached .npy files)
    python -m src.feature_extraction.ast_scalar --mode train

    # Only visualise (SHAP, feature importance, entropy plots)
    python -m src.feature_extraction.ast_scalar --mode explain

Arguments:
    --data_dir      Path to parquet files. Default: data/task_a
    --output_dir    Root output directory. Default: results_output
                    Plots saved to: results_output/ast_scalar/
    --cache_dir     Directory for .npy feature cache. Default: data
    --max_workers   Parallel threads for extraction. Default: 8
    --mode          extract | train | explain | all. Default: all
"""

import os
import math
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    classification_report,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "max_depth",
    "avg_depth",
    "depth_var",
    "total_nodes",
    "leaf_nodes",
    "leaf_ratio",
    "avg_branch",
    "max_children",
    "branch_variance",
    "cyclomatic",
    "decision_nodes",
    "unique_types",
    "entropy",
    "norm_entropy",
    "branch_entropy",
    "function_calls",
    "unique_calls",
    "internal_nodes",
]

N_FEATURES = len(FEATURE_NAMES)    # 18

SUPPORTED_LANGUAGES = ["Python", "Java", "C++"]


# ---------------------------------------------------------------------------
# 1. Parser initialisation
# ---------------------------------------------------------------------------

def _build_parsers() -> dict:
    """
    Build tree-sitter parsers for seen training languages.
    Uses get_parser() from tree_sitter_languages for conciseness.
    """
    from tree_sitter_languages import get_parser
    parsers = {}
    for lang, name in [("Python", "python"), ("Java", "java"), ("C++", "cpp")]:
        try:
            parsers[lang] = get_parser(name)
        except Exception as exc:
            log.warning("Could not load parser for %s: %s", lang, exc)
    return parsers


# Build once at module level — cached for all calls
_PARSERS: dict = {}


def get_parsers() -> dict:
    """Return cached parsers dict, building on first call."""
    global _PARSERS
    if not _PARSERS:
        _PARSERS = _build_parsers()
    return _PARSERS


# ---------------------------------------------------------------------------
# 2. AST scalar feature extraction
# ---------------------------------------------------------------------------

def extract_ast_features(code: str, language: str) -> list[float]:
    """
    Extract 18 structural scalar features from a single code snippet.

    The function builds an AST using tree-sitter and traverses it to
    compute statistics about the tree's shape:
      - Depth statistics:  how deep and variable the tree is
      - Branching:         how many children nodes typically have
      - Complexity:        cyclomatic complexity (decision node count)
      - Diversity:         how many unique node types appear
      - Entropy:           how uniformly node types are distributed
      - Function calls:    count and variety of call expressions

    Note on error nodes:
        tree-sitter inserts 'error' nodes when parsing fails. Human code
        from competitive platforms often triggers these (Python 2 syntax,
        tuple unpacking). LLM code is syntactically clean. Error nodes
        therefore carry implicit authorship signal, which the model learns.
        See module docstring for full discussion.

    Args:
        code:     Raw source code string.
        language: Language name — must be in SUPPORTED_LANGUAGES.
                  Returns zero vector for unsupported languages.

    Returns:
        List of 18 float values (one per feature in FEATURE_NAMES).
        Returns [0]*18 if parsing fails.
    """
    parsers = get_parsers()
    parser  = parsers.get(language)

    if parser is None:
        return [0.0] * N_FEATURES

    try:
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node
    except Exception:
        return [0.0] * N_FEATURES

    # ── Traverse the AST ─────────────────────────────────────────────────────
    stack           = [(root, 0)]
    total_nodes     = 0
    leaf_nodes      = 0
    depths          = []
    children_counts = []
    node_types      = []
    decision_nodes  = 0
    function_calls  = 0
    call_types      = set()

    while stack:
        node, depth = stack.pop()

        total_nodes += 1
        depths.append(depth)
        node_types.append(node.type)

        children = node.children
        children_counts.append(len(children))

        if len(children) == 0:
            leaf_nodes += 1

        if node.type in ("if_statement", "for_statement", "while_statement"):
            decision_nodes += 1

        if "call" in node.type:
            function_calls += 1
            call_types.add(node.type)

        for child in children:
            stack.append((child, depth + 1))

    # ── Depth statistics ─────────────────────────────────────────────────────
    max_depth = max(depths) if depths else 0
    avg_depth = float(np.mean(depths)) if depths else 0.0
    depth_var = float(np.var(depths))  if depths else 0.0

    # ── Branching statistics ──────────────────────────────────────────────────
    leaf_ratio       = leaf_nodes / (total_nodes + 1e-9)
    avg_branch       = float(np.mean(children_counts)) if children_counts else 0.0
    max_children     = max(children_counts) if children_counts else 0
    branch_variance  = float(np.var(children_counts)) if children_counts else 0.0
    internal_nodes   = sum(1 for c in children_counts if c > 0)

    # ── Complexity ────────────────────────────────────────────────────────────
    cyclomatic = decision_nodes + 1

    # ── Node type diversity ───────────────────────────────────────────────────
    unique_types = len(set(node_types))

    # ── Shannon entropy of node type distribution ─────────────────────────────
    # High entropy = many different node types (diverse / human-like)
    # Low entropy  = few types repeated (structured / LLM-like)
    if total_nodes > 0:
        counts = Counter(node_types)
        probs  = [v / total_nodes for v in counts.values()]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
    else:
        entropy = 0.0

    num_types    = len(counts) if total_nodes > 0 else 1
    norm_entropy = entropy / math.log(num_types + 1e-10) if num_types > 1 else 0.0

    # ── Branch factor entropy ─────────────────────────────────────────────────
    if children_counts:
        branch_counts = Counter(children_counts)
        total_b       = len(children_counts)
        branch_probs  = [v / total_b for v in branch_counts.values()]
        branch_entropy = -sum(p * math.log(p + 1e-10) for p in branch_probs)
    else:
        branch_entropy = 0.0

    return [
        float(max_depth),
        avg_depth,
        depth_var,
        float(total_nodes),
        float(leaf_nodes),
        leaf_ratio,
        avg_branch,
        float(max_children),
        branch_variance,
        float(cyclomatic),
        float(decision_nodes),
        float(unique_types),
        entropy,
        norm_entropy,
        branch_entropy,
        float(function_calls),
        float(len(call_types)),
        float(internal_nodes),
    ]


def parallel_extract(
    df: pd.DataFrame,
    max_workers: int = 8,
    desc: str = "Extracting AST scalars",
) -> np.ndarray:
    """
    Extract scalar features for all rows in a DataFrame using parallel threads.

    Args:
        df:          DataFrame with 'code' and 'language' columns.
        max_workers: Number of parallel threads.
        desc:        tqdm label.

    Returns:
        NumPy array of shape (n_samples, 18).
    """
    tasks   = list(zip(df["code"].tolist(), df["language"].tolist()))
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for feat in tqdm(
            executor.map(lambda x: extract_ast_features(x[0], x[1]), tasks),
            total=len(tasks),
            desc=desc,
        ):
            results.append(feat)

    return np.array(results, dtype=float)


# ---------------------------------------------------------------------------
# 3. Feature loading / caching
# ---------------------------------------------------------------------------

def load_or_compute_features(
    df: pd.DataFrame,
    cache_path: str,
    max_workers: int,
    desc: str,
) -> np.ndarray:
    """
    Load precomputed scalar features from .npy cache, or compute and save.

    Args:
        df:          Source DataFrame.
        cache_path:  Path to .npy file.
        max_workers: Parallel threads (used only when computing).
        desc:        Progress bar label.

    Returns:
        Feature array of shape (n_samples, 18).
    """
    if os.path.exists(cache_path):
        log.info("Loading cached features: %s", cache_path)
        return np.load(cache_path)

    log.info("Cache not found. Computing features for %d samples...", len(df))
    features = parallel_extract(df, max_workers=max_workers, desc=desc)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, features)
    log.info("Saved feature cache: %s", cache_path)
    return features


# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------

def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train Random Forest on AST scalar features.

    Uses max_features='sqrt' and min_samples_leaf=5 for regularisation,
    since scalar features are dense (unlike sparse TF-IDF matrices).
    """
    log.info("Training Random Forest on AST scalar features...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_svm(X: np.ndarray, y: np.ndarray) -> LinearSVC:
    """Train LinearSVC on normalised scalar features."""
    log.info("Training LinearSVC on AST scalar features...")
    model = LinearSVC(class_weight="balanced", max_iter=5000)
    model.fit(X, y)
    return model


def train_catboost(X: np.ndarray, y: np.ndarray) -> CatBoostClassifier:
    """
    Train CatBoost on AST scalar features.

    Uses 500 iterations (more than TF-IDF variant) because scalar features
    are dense and require more trees to capture non-linear combinations.
    """
    log.info("Training CatBoost on AST scalar features (500 iterations)...")
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        verbose=100,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    output_dir: str,
) -> dict:
    """
    Evaluate a trained model and save a report to output_dir.

    Always evaluates on the provided (X, y) — caller controls whether
    this is validation or test_sample (OOD).

    Metrics:
        accuracy, binary_f1, macro_f1, macro_precision, macro_recall,
        roc_auc, mcc (Matthews Correlation Coefficient), fnr (False Negative Rate)

    Args:
        model:      Trained classifier.
        X:          Feature matrix.
        y:          True labels.
        name:       Display name for logging and report filename.
        output_dir: Directory to save the .txt report.

    Returns:
        Dict of metric_name -> float.
    """
    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.decision_function(X)

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
    print(f"  {name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<20} : {v:.4f}")
    print(f"\n{report}")

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    safe_name  = name.replace(" ", "_").replace("(", "").replace(")", "")
    report_path = os.path.join(output_dir, f"{safe_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n{'='*55}\n")
        for k, v in metrics.items():
            f.write(f"{k:<20}: {v:.4f}\n")
        f.write(f"\n{report}")
    log.info("Saved report: %s", report_path)

    # Confusion matrix plot (required by report Section 5.2)
    cm   = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Human", "Machine"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{safe_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved confusion matrix: %s", cm_path)

    return metrics


# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models: dict,
    X_val: np.ndarray,  y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    output_dir: str,
) -> None:
    """
    Plot ROC curves on both validation (in-dist) and test_sample (OOD).
    Saves two separate figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, X, y in [
        ("Validation (In-Distribution)", X_val,  y_val),
        ("OOD Test (Unseen Languages)",  X_test, y_test),
    ]:
        plt.figure(figsize=(7, 6))
        for name, model in models.items():
            probs = (
                model.predict_proba(X)[:, 1]
                if hasattr(model, "predict_proba")
                else model.decision_function(X)
            )
            fpr, tpr, _ = roc_curve(y, probs)
            auc = roc_auc_score(y, probs)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — AST Scalar Features\n{split_name}")
        plt.legend()
        plt.tight_layout()

        fname = split_name.split()[0].lower()
        save_path = os.path.join(output_dir, f"roc_curves_ast_scalar_{fname}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved ROC curve: %s", save_path)


def plot_feature_importance(
    model: RandomForestClassifier,
    output_dir: str,
) -> None:
    """
    Plot and print feature importance from Random Forest.

    branch_variance and leaf_ratio tend to be most important —
    LLM code has more balanced trees (lower variance) while human
    code has irregular branching.
    """
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    names_sorted = np.array(FEATURE_NAMES)[sorted_idx]

    plt.figure(figsize=(8, 6))
    plt.barh(names_sorted, importances[sorted_idx], color="#1976D2", alpha=0.85)
    plt.xlabel("Importance Score")
    plt.title("Feature Importance — AST Scalar Features (Random Forest)")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "feature_importance_ast_scalar.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved feature importance: %s", save_path)

    imp_df = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    print(f"\nFeature importance ranking:\n{imp_df.to_string(index=False)}")

    imp_df.to_csv(
        os.path.join(output_dir, "feature_importance_ast_scalar.csv"),
        index=False,
    )


def plot_entropy_distribution(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_dir: str,
) -> None:
    """
    Plot AST entropy distribution split by Human vs Machine.

    Entropy measures how uniformly node types are distributed in the AST.
    LLM code tends to have lower entropy (fewer, more repetitive node types)
    while human code has higher entropy (more diverse structures).
    """
    idx      = FEATURE_NAMES.index("entropy")
    h_vals   = X_train[y_train == 0][:, idx]
    m_vals   = X_train[y_train == 1][:, idx]

    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(h_vals, bins=40, alpha=0.6, density=True, color="#43A047", label="Human")
    plt.hist(m_vals, bins=40, alpha=0.6, density=True, color="#E53935", label="Machine")
    plt.xlabel("AST Entropy")
    plt.ylabel("Density")
    plt.title("AST Node-Type Entropy: Human vs Machine Code")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "entropy_distribution.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved entropy distribution: %s", hist_path)

    # Boxplot
    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [h_vals, m_vals],
        tick_labels=["Human Code", "Machine Code"],
    )
    plt.ylabel("AST Entropy")
    plt.title("Structural Entropy Comparison")
    plt.tight_layout()
    box_path = os.path.join(output_dir, "entropy_boxplot.png")
    plt.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved entropy boxplot: %s", box_path)

    print(f"\nEntropy stats:")
    print(f"  Human   — mean={h_vals.mean():.3f}, std={h_vals.std():.3f}")
    print(f"  Machine — mean={m_vals.mean():.3f}, std={m_vals.std():.3f}")


def plot_feature_correlation(
    X_train: np.ndarray,
    output_dir: str,
) -> None:
    """
    Plot correlation heatmap of all 18 scalar features.
    Helps identify redundant features.
    """
    df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation — AST Scalar Features")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "feature_correlation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved correlation heatmap: %s", save_path)


def plot_shap(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    output_dir: str,
    n_samples: int = 200,
) -> None:
    """
    SHAP summary plot for Random Forest — shows which features push
    predictions toward Machine (positive) or Human (negative).

    Uses a random sample of validation data for speed.
    """
    log.info("Computing SHAP values on %d validation samples...", n_samples)
    idx        = np.random.choice(len(X_val), n_samples, replace=False)
    X_shap     = X_val[idx]
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_shap)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # class 1 = Machine

    shap.summary_plot(
        shap_vals,
        X_shap,
        feature_names=FEATURE_NAMES,
        plot_type="bar",
        show=False,
    )
    save_path = os.path.join(output_dir, "shap_summary_ast_scalar.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved SHAP summary: %s", save_path)


def save_results_table(
    val_results:  dict,
    test_results: dict,
    output_dir:   str,
) -> None:
    """
    Save consolidated results table comparing val and OOD test performance.
    """
    rows = []
    for name in val_results:
        rows.append({"Model": f"{name} (Val)", **val_results[name]})
    for name in test_results:
        rows.append({"Model": f"{name} (OOD Test)", **test_results[name]})

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "ast_scalar_results.csv")
    df.to_csv(save_path, index=False)
    log.info("Saved results table: %s", save_path)
    print(f"\nResults summary:\n{df.to_string(index=False)}")


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """
    End-to-end pipeline for AST scalar feature extraction and evaluation.

    Output structure:
        results_output/
        └── ast_scalar/          ← all outputs from this script
            ├── *_report.txt
            ├── roc_curves_*.png
            ├── feature_importance_*.png
            ├── entropy_*.png
            ├── feature_correlation.png
            ├── shap_summary_*.png
            └── ast_scalar_results.csv
    """
    # All outputs go into results_output/ast_scalar/
    out_dir = os.path.join(args.output_dir, "ast_scalar")
    os.makedirs(out_dir, exist_ok=True)
    log.info("All outputs will be saved to: %s", out_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading datasets from: %s", args.data_dir)
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val_df   = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir, "test_sample.parquet"))

    log.info("Train: %d | Val: %d | Test (OOD): %d", len(train_df), len(val_df), len(test_df))
    log.info(
        "Class balance — Train: %.1f%% Machine | Test (OOD): %.1f%% Machine",
        100 * train_df["label"].mean(),
        100 * test_df["label"].mean(),
    )

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # ── Feature extraction ────────────────────────────────────────────────────
    if args.mode in ("extract", "all", "train", "explain"):
        X_train = load_or_compute_features(
            train_df,
            cache_path=os.path.join(args.cache_dir, "X_train_ast_scalars.npy"),
            max_workers=args.max_workers,
            desc="Extracting train scalar features",
        )
        X_val = load_or_compute_features(
            val_df,
            cache_path=os.path.join(args.cache_dir, "X_val_ast_scalars.npy"),
            max_workers=args.max_workers,
            desc="Extracting val scalar features",
        )
        X_test = load_or_compute_features(
            test_df,
            cache_path=os.path.join(args.cache_dir, "X_test_ast_scalars.npy"),
            max_workers=args.max_workers,
            desc="Extracting test scalar features",
        )
        log.info(
            "Feature matrix shapes — train=%s | val=%s | test=%s",
            X_train.shape, X_val.shape, X_test.shape,
        )

        # Normalise for SVM
        scaler  = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)

    if args.mode == "extract":
        log.info("Feature extraction complete. Exiting.")
        return

    # ── Training ──────────────────────────────────────────────────────────────
    if args.mode in ("train", "all"):
        rf_model  = train_random_forest(X_train, y_train)
        svm_model = train_svm(X_train_scaled, y_train)
        cat_model = train_catboost(X_train, y_train)

        trained_models = {
            "Random Forest": (rf_model,  X_val,        X_test),
            "SVM":           (svm_model, X_val_scaled, X_test_scaled),
            "CatBoost":      (cat_model, X_val,        X_test),
        }

        # ── Validation evaluation (in-distribution) ───────────────────────────
        log.info("\n--- Validation Results (In-Distribution) ---")
        val_results = {}
        for name, (model, Xv, _) in trained_models.items():
            val_results[name] = evaluate_model(
                model, Xv, y_val,
                name=f"{name} (Val)",
                output_dir=out_dir,
            )

        # ── Test evaluation (OOD) — always on test_sample ────────────────────
        log.info("\n--- OOD Test Results (Unseen Languages + Imbalanced) ---")
        log.info("Note: test_sample has 22%% Machine vs 52%% in training.")
        log.info("Low OOD scores are expected — this measures OOD robustness.")
        test_results = {}
        for name, (model, _, Xt) in trained_models.items():
            test_results[name] = evaluate_model(
                model, Xt, y_test,
                name=f"{name} (OOD Test)",
                output_dir=out_dir,
            )

        # ── ROC curves ────────────────────────────────────────────────────────
        plot_roc_curves(
            {n: m for n, (m, _, _) in trained_models.items()},
            X_val=X_val,   y_val=y_val,
            X_test=X_test, y_test=y_test,
            output_dir=out_dir,
        )

        # ── Results table ─────────────────────────────────────────────────────
        save_results_table(val_results, test_results, out_dir)

    # ── Visualisation / interpretation ───────────────────────────────────────
    if args.mode in ("explain", "all"):
        log.info("\n--- Feature Importance & Interpretation ---")
        plot_feature_importance(rf_model, out_dir)
        plot_entropy_distribution(X_train, y_train, out_dir)
        plot_feature_correlation(X_train, out_dir)
        plot_shap(rf_model, X_val, out_dir)

    log.info("\nAll outputs saved to: %s", out_dir)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AST scalar feature extraction for machine-generated code detection."
    )
    p.add_argument("--data_dir",    default="data/task_a",
                   help="Directory with train/val/test_sample parquet files")
    p.add_argument("--output_dir",  default="results_output",
                   help="Root output dir. Script saves to output_dir/ast_scalar/")
    p.add_argument("--cache_dir",   default="data",
                   help="Directory for .npy feature cache files")
    p.add_argument("--max_workers", type=int, default=8,
                   help="Parallel threads for feature extraction")
    p.add_argument("--mode", choices=["extract", "train", "explain", "all"],
                   default="all",
                   help="Pipeline stage to run")
    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
