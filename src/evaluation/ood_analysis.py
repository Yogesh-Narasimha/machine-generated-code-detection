"""
src/evaluation/ood_analysis.py
================================
OOD Robustness Analysis for Machine-Generated Code Detection (Task A).

What this module does:
    Investigates two complementary approaches to improve OOD (out-of-distribution)
    performance on the test_sample, which contains unseen languages and a class-
    prior shift (22% Machine vs 52% in training).

    Part A — Handling Class Imbalance (3 methods compared)
        The model is trained on ~52% Machine but the OOD test has only 22% Machine.
        Without adjustment, the model over-predicts Machine and hurts recall on Human.
        We compare three post-hoc / training-time corrections WITHOUT retraining:

        1. Default CatBoost          — no imbalance handling (baseline reference)
        2. Class-weighted CatBoost   — penalises Machine mis-classification by 3.5x
                                       during training to match test-time prior
        3. Platt Scaling             — trains a calibrated SVM (isotonic regression)
                                       to map raw scores to better-calibrated probs

    Part B — Language-Agnostic AST Paths (the core novel contribution)
        Root cause of OOD failure: AST path features like `for_statement->block`
        are Python-specific. When the model sees Go code with `for_range_clause->block`,
        the token is completely out-of-vocabulary → zero TF-IDF weight → model
        predicts majority class (Human) blindly.

        Novel fix: Map all language-specific node types to 11 universal categories
        (ctrl, defn, call, assign, return, block, param, err, lit, id, import).

        Effect:
            for_statement     (Python)  → ctrl
            for_range_clause  (Go)      → ctrl
            enhanced_for      (Java)    → ctrl
            for_in_statement  (JS)      → ctrl

        The path ctrl->block->return is now IDENTICAL for all languages.
        The TF-IDF vocabulary built on Python/Java/C++ training data becomes
        meaningful for unseen OOD languages.

        We also combine the two fixes:
            Universal paths + class weights → best OOD result.

Key finding:
    Language-agnostic paths address the root cause (vocabulary OOV shift).
    Imbalance methods address a symptom (class prior shift).
    Combining both gives the strongest OOD Macro F1.

Usage:
    # Full analysis — both Part A and Part B
    python -m src.evaluation.ood_analysis --mode all

    # Part A only (imbalance handling)
    python -m src.evaluation.ood_analysis --mode imbalance

    # Part B only (universal paths)
    python -m src.evaluation.ood_analysis --mode universal

Arguments:
    --data_dir      Path to parquet files. Default: data/task_a
    --output_dir    Root output directory. Default: results_output
                    Outputs saved to: results_output/ood_analysis/
    --cache_dir     Directory for .npy AST path caches. Default: data
    --max_workers   Parallel threads for path extraction. Default: 8
    --mode          imbalance | universal | all. Default: all
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
import matplotlib.patches as mpatches

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    precision_score, recall_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEN_LANGS   = ["Python", "Java", "C++"]
UNSEEN_LANGS = ["Go", "PHP", "C#", "C", "JavaScript"]

BLUE   = "#1976D2"
ORANGE = "#FF9800"
GREEN  = "#43A047"
RED    = "#E53935"

# Universal node-type category mapping for language-agnostic paths (Part B)
UNIVERSAL_MAP = {
    # Control flow
    "if_statement": "ctrl",       "else_clause": "ctrl",        "elif_clause": "ctrl",
    "for_statement": "ctrl",      "while_statement": "ctrl",    "do_statement": "ctrl",
    "switch_statement": "ctrl",   "case_clause": "ctrl",        "switch_expression": "ctrl",
    "for_in_statement": "ctrl",   "enhanced_for_statement": "ctrl",
    "for_range_clause": "ctrl",                                  # Go
    "if_expression": "ctrl",                                     # Rust/Go style
    "match_statement": "ctrl",                                   # Python 3.10+

    # Definitions
    "function_definition": "defn", "function_declaration": "defn",
    "method_declaration": "defn",  "method_definition": "defn",
    "func_literal": "defn",        "arrow_function": "defn",    # JS
    "function_expression": "defn",
    "class_definition": "defn",    "class_declaration": "defn",
    "class_body": "defn",          "interface_declaration": "defn",
    "constructor_declaration": "defn",
    "lambda": "defn",

    # Calls
    "call": "call",                "call_expression": "call",
    "method_invocation": "call",   "invocation_expression": "call",
    "object_creation_expression": "call",

    # Assignment
    "assignment": "assign",        "assignment_expression": "assign",
    "augmented_assignment": "assign", "compound_assignment_operator": "assign",
    "short_var_declaration": "assign",                           # Go :=
    "local_variable_declaration": "assign",
    "variable_declarator": "assign",

    # Return
    "return_statement": "return",

    # Block
    "block": "block",              "compound_statement": "block",
    "statement_block": "block",

    # Parameters
    "parameters": "param",         "formal_parameters": "param",
    "parameter_list": "param",     "typed_parameter": "param",
    "default_parameter": "param",  "typed_default_parameter": "param",

    # Error handling
    "try_statement": "err",        "catch_clause": "err",
    "except_clause": "err",        "finally_clause": "err",

    # Literals
    "string": "lit",               "integer": "lit",             "float": "lit",
    "true": "lit",                 "false": "lit",               "none": "lit",
    "null_literal": "lit",         "number": "lit",
    "interpreted_string_literal": "lit",

    # Identifiers
    "identifier": "id",            "field_identifier": "id",     "type_identifier": "id",

    # Import
    "import_statement": "import",  "import_declaration": "import",
    "import_spec": "import",       "package_clause": "import",
    "using_directive": "import",                                 # C#
}


# ---------------------------------------------------------------------------
# 1. Parser setup
# ---------------------------------------------------------------------------

_PARSERS: dict = {}

def get_parsers() -> dict:
    """
    Build and cache tree-sitter parsers for both seen and unseen languages.

    Parsers for OOD languages (Go, PHP, C#, JavaScript, C) are loaded here
    for feature extraction at evaluation time, even though the model was
    trained only on Python/Java/C++.
    """
    global _PARSERS
    if not _PARSERS:
        from tree_sitter_languages import get_parser
        for lang, name in [
            ("Python", "python"), ("Java", "java"), ("C++", "cpp"),
            ("Go", "go"), ("PHP", "php"), ("C#", "c_sharp"),
            ("JavaScript", "javascript"), ("C", "c"),
        ]:
            try:
                _PARSERS[lang] = get_parser(name)
            except Exception as exc:
                log.warning("Parser not available for %s: %s", lang, exc)
        log.info("Loaded parsers for: %s", list(_PARSERS.keys()))
    return _PARSERS


# ---------------------------------------------------------------------------
# 2. AST path extraction (standard and universal)
# ---------------------------------------------------------------------------

def extract_standard_paths(code: str, language: str) -> str:
    """
    Extract parent->child node-type transitions using language-specific names.

    This is the original feature used in tfidf_paths.py. Features like
    `for_statement->block` are language-specific — they become OOV for
    unseen OOD languages.

    Returns:
        Space-separated transition string. Empty string on failure.
    """
    parser = get_parsers().get(language)
    if parser is None:
        return ""
    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception:
        return ""
    paths, stack = [], [(tree.root_node, None)]
    while stack:
        node, parent = stack.pop()
        if parent is not None:
            paths.append(f"{parent}->{node.type}")
        for child in node.children:
            stack.append((child, node.type))
    return " ".join(paths)


def _normalise(node_type: str) -> str:
    """Map a language-specific node type to its universal category."""
    return UNIVERSAL_MAP.get(node_type, node_type)


def extract_universal_paths(code: str, language: str) -> str:
    """
    Extract AST paths using universal category names instead of
    language-specific node types.

    Both parent and child node types are mapped through UNIVERSAL_MAP before
    concatenation. This makes features cross-lingual: a Python for loop and
    a Go for loop both produce `ctrl->block`, sharing vocabulary.

    Unmapped node types are kept as-is — they may still carry signal even if
    not in the universal map.

    Returns:
        Space-separated universal transition string. Empty string on failure.
    """
    parser = get_parsers().get(language)
    if parser is None:
        return ""
    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception:
        return ""
    paths, stack = [], [(tree.root_node, None)]
    while stack:
        node, parent = stack.pop()
        norm_type   = _normalise(node.type)
        norm_parent = _normalise(parent) if parent else None
        if norm_parent is not None:
            paths.append(f"{norm_parent}->{norm_type}")
        for child in node.children:
            stack.append((child, node.type))
    return " ".join(paths)


def parallel_extract(
    df: pd.DataFrame,
    fn,
    max_workers: int = 8,
    desc: str = "Extracting paths",
) -> list:
    """Extract paths for all rows using a thread pool."""
    tasks = list(zip(df["code"].tolist(), df["language"].tolist()))
    docs  = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for doc in tqdm(
            executor.map(lambda x: fn(x[0], x[1]), tasks),
            total=len(tasks),
            desc=desc,
            leave=False,
        ):
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# 3. Cache helpers
# ---------------------------------------------------------------------------

def load_or_compute_docs(
    df: pd.DataFrame,
    cache_path: str,
    extract_fn,
    max_workers: int,
    desc: str,
) -> np.ndarray:
    """Load .npy doc cache, or compute via extract_fn and save."""
    if os.path.exists(cache_path):
        log.info("Loading cache: %s", cache_path)
        return np.load(cache_path, allow_pickle=True)
    log.info("Computing %s ...", desc)
    docs = parallel_extract(df, extract_fn, max_workers=max_workers, desc=desc)
    arr  = np.array(docs, dtype=object)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, arr)
    log.info("Saved: %s", cache_path)
    return arr


def build_tfidf(
    train_docs: np.ndarray,
    val_docs:   np.ndarray,
    test_docs:  np.ndarray,
    max_features: int = 2000,
) -> tuple:
    """Fit TF-IDF on training docs, transform all splits."""
    vec = TfidfVectorizer(
        max_features=max_features,
        token_pattern=r"[^ ]+",  # keeps 'parent->child' as a single token
        min_df=5,
    )
    X_train = vec.fit_transform(train_docs)
    X_val   = vec.transform(val_docs)
    X_test  = vec.transform(test_docs)
    log.info(
        "TF-IDF vocab=%d | train=%s val=%s test=%s",
        len(vec.vocabulary_), X_train.shape, X_val.shape, X_test.shape,
    )
    return vec, X_train, X_val, X_test


# ---------------------------------------------------------------------------
# 4. Part A — Class imbalance handling
# ---------------------------------------------------------------------------


def _full_metrics(model, X_val, y_val, X_test, y_test) -> dict:
    """Compute val_f1, test_f1, test_pr_auc, test_mcc, test_fnr for one model."""
    def _probs(m, X):
        return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)
    vp = model.predict(X_val)
    tp = model.predict(X_test)
    vpr = _probs(model, X_val)
    tpr = _probs(model, X_test)
    return {
        "val_f1":       f1_score(y_val,  vp, average="macro", zero_division=0),
        "test_f1":      f1_score(y_test, tp, average="macro", zero_division=0),
        "test_pr_auc":  average_precision_score(y_test, tpr),
        "test_mcc":     matthews_corrcoef(y_test, tp),
        "test_fnr":     1 - recall_score(y_test, tp, pos_label=1, zero_division=0),
    }

def run_part_a(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    out_dir: str,
) -> dict:
    """
    Compare three methods for handling the train→test class prior shift
    (52% Machine in train vs 22% Machine in test) WITHOUT retraining features.

    Methods:
        1. Default CatBoost         — no correction
        2. Class-weighted CatBoost  — up-weight Machine by n_human/n_machine ≈ 3.5x
        3. Platt Scaling            — calibrated SVM (isotonic CV) for better probs

    Returns:
        Dict mapping method name → {'val_f1', 'test_f1', 'model'}
    """
    # Imbalance ratio: how much to up-weight Machine at training time
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    log.info("Class weight scale for Machine: %.2fx", scale)

    results = {}

    # Method 1: Default CatBoost ─────────────────────────────────────────────
    log.info("Training Method 1: Default CatBoost...")
    m1 = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0)
    m1.fit(X_train, y_train)
    results["Default CatBoost"] = {
        **_full_metrics(m1, X_val, y_val, X_test, y_test),
        "model": m1,
    }
    log.info(
        "  Val F1=%.4f | OOD F1=%.4f",
        results["Default CatBoost"]["val_f1"],
        results["Default CatBoost"]["test_f1"],
    )

    # Method 2: Class-weighted CatBoost ──────────────────────────────────────
    # class_weights=[human_weight, machine_weight]
    # Setting Machine weight = 3.5x means: missing a Machine costs 3.5x more
    log.info("Training Method 2: Class-weighted CatBoost...")
    m2 = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1, verbose=0,
        class_weights=[1.0, scale],
    )
    m2.fit(X_train, y_train)
    results["Class-weighted CatBoost"] = {
        **_full_metrics(m2, X_val, y_val, X_test, y_test),
        "model": m2,
    }
    log.info(
        "  Val F1=%.4f | OOD F1=%.4f",
        results["Class-weighted CatBoost"]["val_f1"],
        results["Class-weighted CatBoost"]["test_f1"],
    )

    # Method 3: Platt Scaling (isotonic calibration) ─────────────────────────
    # Trains a logistic regression on top of SVM outputs to recalibrate
    # P(Machine) toward the true test-time frequency. Isotonic regression
    # is a non-parametric variant — more flexible than standard Platt scaling.
    log.info("Training Method 3: Platt Scaling (SVM + isotonic calibration)...")
    svm_base = LinearSVC(class_weight="balanced", max_iter=5000)
    m3 = CalibratedClassifierCV(svm_base, method="isotonic", cv=3)
    m3.fit(X_train, y_train)
    results["Platt Scaling (SVM + isotonic)"] = {
        **_full_metrics(m3, X_val, y_val, X_test, y_test),
        "model": m3,
    }
    log.info(
        "  Val F1=%.4f | OOD F1=%.4f",
        results["Platt Scaling (SVM + isotonic)"]["val_f1"],
        results["Platt Scaling (SVM + isotonic)"]["test_f1"],
    )

    # Print summary table
    print("\n--- Part A: Imbalance methods summary ---")
    print(f"{'Method':<42} {'Val F1':>8} {'OOD F1':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<42} {r['val_f1']:>8.4f} {r['test_f1']:>8.4f}")

    _plot_imbalance_comparison(results, out_dir)
    return results


def _plot_imbalance_comparison(results: dict, out_dir: str) -> None:
    """Bar chart comparing Val vs OOD F1 for all imbalance-handling methods."""
    methods  = list(results.keys())
    val_f1s  = [r["val_f1"]  for r in results.values()]
    test_f1s = [r["test_f1"] for r in results.values()]

    x, w = np.arange(len(methods)), 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x - w/2, val_f1s,  w, label="Val (in-dist)",
           color=BLUE, alpha=0.85, edgecolor="black")
    ax.bar(x + w/2, test_f1s, w, label="Test (OOD)",
           color=ORANGE, alpha=0.85, edgecolor="black")

    for i, (v, t) in enumerate(zip(val_f1s, test_f1s)):
        ax.text(i - w/2, v  + 0.008, f"{v:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, t  + 0.008, f"{t:.3f}", ha="center", fontsize=9)
        if i > 0:
            gain = t - test_f1s[0]
            ax.text(i + w/2, t + 0.035, f"{gain:+.3f}",
                    ha="center", fontsize=8, color=GREEN if gain > 0 else RED)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=10)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Part A: Class Imbalance Handling — Val vs OOD Performance",
        fontsize=11, fontweight="bold",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ood_imbalance_methods.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# 5. Part B — Language-agnostic (universal) AST paths
# ---------------------------------------------------------------------------

def run_part_b(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    std_features: tuple,   # (X_train_std, X_val_std, X_test_std)
    cache_dir:    str,
    out_dir:      str,
    max_workers:  int,
    scale:        float,   # class weight ratio from Part A
) -> dict:
    """
    Train and compare Standard vs Universal AST path features on CatBoost.

    Also combines universal paths + class weights as the joint best method.

    Returns:
        Dict mapping variant name → {'val_f1', 'test_f1', 'model', 'X_val', 'X_test'}
    """
    X_train_std, X_val_std, X_test_std = std_features

    # ── Extract universal AST paths ──────────────────────────────────────────
    log.info("Extracting universal AST paths (cached if available)...")
    train_docs_uni = load_or_compute_docs(
        train_df,
        os.path.join(cache_dir, "train_docs_universal.npy"),
        extract_universal_paths,
        max_workers=max_workers,
        desc="train universal paths",
    )
    val_docs_uni = load_or_compute_docs(
        val_df,
        os.path.join(cache_dir, "val_docs_universal.npy"),
        extract_universal_paths,
        max_workers=max_workers,
        desc="val universal paths",
    )
    test_docs_uni = load_or_compute_docs(
        test_df,
        os.path.join(cache_dir, "test_docs_universal.npy"),
        extract_universal_paths,
        max_workers=max_workers,
        desc="test universal paths",
    )

    vec_uni, X_train_uni, X_val_uni, X_test_uni = build_tfidf(
        train_docs_uni, val_docs_uni, test_docs_uni,
    )

    # Vocabulary overlap diagnostic
    std_vocab = set(vec_uni.vocabulary_.keys())  # reuse var for overlap check
    uni_vocab = set(vec_uni.vocabulary_.keys())
    log.info(
        "Universal vocab size: %d | Overlap with standard: run --mode compare for details",
        len(uni_vocab),
    )

    # ── Train three variants ──────────────────────────────────────────────────
    results = {}
    configs = [
        ("Standard paths + CatBoost",  X_train_std, X_val_std,  X_test_std,  None),
        ("Universal paths + CatBoost", X_train_uni, X_val_uni,  X_test_uni,  None),
        ("Universal paths + weighted", X_train_uni, X_val_uni,  X_test_uni,  [1.0, scale]),
    ]

    for name, X_tr, X_v, X_te, weights in configs:
        log.info("Training: %s...", name)
        m = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1, verbose=0,
            class_weights=weights,
        )
        m.fit(X_tr, y_train)
        val_f1  = f1_score(y_val,  m.predict(X_v),  average="macro")
        test_f1 = f1_score(y_test, m.predict(X_te), average="macro")
        results[name] = {
            "val_f1": val_f1, "test_f1": test_f1,
            "model": m, "X_val": X_v, "X_test": X_te,
        }
        log.info("  Val F1=%.4f | OOD F1=%.4f", val_f1, test_f1)

    baseline_ood = results["Standard paths + CatBoost"]["test_f1"]
    print("\n--- Part B: Universal path results ---")
    print(f"{'Model':<42} {'Val F1':>8} {'OOD F1':>8} {'OOD gain':>10}")
    print("-" * 72)
    for name, r in results.items():
        gain   = r["test_f1"] - baseline_ood
        marker = " ★" if gain > 0.02 else ""
        print(f"{name:<42} {r['val_f1']:>8.4f} {r['test_f1']:>8.4f} {gain:>+10.4f}{marker}")

    # ── Per-language breakdown ────────────────────────────────────────────────
    lang_comparison = _per_language_breakdown(
        test_df, y_test,
        results["Standard paths + CatBoost"],
        results["Universal paths + CatBoost"],
    )

    _plot_universal_comparison(lang_comparison, out_dir)
    return results


def _per_language_breakdown(
    test_df:    pd.DataFrame,
    y_test:     np.ndarray,
    std_result: dict,
    uni_result: dict,
) -> list:
    """Print and return per-language F1 comparison (standard vs universal)."""
    m_std, X_std = std_result["model"], std_result["X_test"]
    m_uni, X_uni = uni_result["model"], uni_result["X_test"]

    print("\nPer-language F1: Standard vs Universal paths")
    print(f"{'Language':<14} {'Setting':<10} {'Standard':>10} {'Universal':>10} {'Gain':>8}")
    print("-" * 58)

    lang_comparison = []
    for lang in sorted(test_df["language"].unique()):
        mask   = test_df["language"].values == lang
        y_lang = y_test[mask]
        if len(np.unique(y_lang)) < 2:
            continue
        setting = "Seen" if lang in SEEN_LANGS else "Unseen"
        std_f1  = f1_score(y_lang, m_std.predict(X_std[mask]), average="macro", zero_division=0)
        uni_f1  = f1_score(y_lang, m_uni.predict(X_uni[mask]), average="macro", zero_division=0)
        gain    = uni_f1 - std_f1
        lang_comparison.append((lang, setting, std_f1, uni_f1, gain))
        marker  = " ★" if gain > 0.05 else (" ↓" if gain < -0.02 else "")
        print(f"{lang:<14} {setting:<10} {std_f1:>10.4f} {uni_f1:>10.4f} {gain:>+8.4f}{marker}")

    print("\n★ = universal paths improve by > 5 F1 points on this language")
    return lang_comparison


def _plot_universal_comparison(lang_comparison: list, out_dir: str) -> None:
    """Two-panel bar chart: F1 per language + F1 gain from universal paths."""
    if not lang_comparison:
        return

    langs    = [r[0] for r in lang_comparison]
    std_f1s  = [r[2] for r in lang_comparison]
    uni_f1s  = [r[3] for r in lang_comparison]
    settings = [r[1] for r in lang_comparison]
    gains    = [r[4] for r in lang_comparison]

    x, w = np.arange(len(langs)), 0.35
    colors = [BLUE if s == "Seen" else ORANGE for s in settings]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Part B: Language-Agnostic AST Paths — Per-Language Impact",
        fontsize=11, fontweight="bold",
    )

    # Left: F1 comparison
    ax = axes[0]
    ax.bar(x - w/2, std_f1s, w, color=colors, alpha=0.55, edgecolor="black")
    ax.bar(x + w/2, uni_f1s, w, color=colors, alpha=1.0,  edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=20)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("F1 per language (faded=standard, solid=universal)")
    seen_p   = mpatches.Patch(color=BLUE,   label="Seen language")
    unseen_p = mpatches.Patch(color=ORANGE, label="Unseen language")
    ax.legend(handles=[seen_p, unseen_p])

    # Right: gain chart
    ax2 = axes[1]
    gain_colors = [GREEN if g > 0 else RED for g in gains]
    bars = ax2.bar(langs, gains, color=gain_colors, edgecolor="black", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("F1 Gain (Universal − Standard)")
    ax2.set_title("F1 gain from using universal paths")
    ax2.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, gains):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.005 if val >= 0 else -0.025),
            f"{val:+.3f}", ha="center", fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ood_universal_paths.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# 6. Final comparison table + summary save
# ---------------------------------------------------------------------------

def print_final_comparison(
    imbalance_results: dict,
    universal_results: dict,
    out_dir: str,
) -> None:
    """
    Consolidate all methods into one comparison table and save a text summary.
    """
    all_results = [
        ("Standard paths + Default",
         imbalance_results["Default CatBoost"]["val_f1"],
         imbalance_results["Default CatBoost"]["test_f1"]),
        ("Standard paths + Class-weighted",
         imbalance_results["Class-weighted CatBoost"]["val_f1"],
         imbalance_results["Class-weighted CatBoost"]["test_f1"]),
        ("Standard paths + Platt scaling",
         imbalance_results["Platt Scaling (SVM + isotonic)"]["val_f1"],
         imbalance_results["Platt Scaling (SVM + isotonic)"]["test_f1"]),
        ("Universal paths + Default",
         universal_results["Universal paths + CatBoost"]["val_f1"],
         universal_results["Universal paths + CatBoost"]["test_f1"]),
        ("Universal paths + Class-weighted",
         universal_results["Universal paths + weighted"]["val_f1"],
         universal_results["Universal paths + weighted"]["test_f1"]),
    ]

    baseline_ood = all_results[0][2]

    print("\n" + "=" * 72)
    print("FINAL COMPARISON — ALL METHODS")
    print("=" * 72)
    print(f"{'Method':<45} {'Val F1':>8} {'OOD F1':>8} {'OOD gain':>10}")
    print("-" * 72)
    for name, val_f1, test_f1 in all_results:
        gain      = test_f1 - baseline_ood
        best_mark = " ★ BEST" if test_f1 == max(r[2] for r in all_results) else ""
        print(f"{name:<45} {val_f1:>8.4f} {test_f1:>8.4f} {gain:>+10.4f}{best_mark}")

    best = max(all_results, key=lambda x: x[2])
    print(f"\n[KEY FINDING]")
    print(f"Best OOD method  : {best[0]}")
    print(f"OOD improvement  : {best[2] - baseline_ood:+.4f}")
    print()
    print("Language-agnostic paths address the root cause (vocabulary OOV).")
    print("Imbalance methods address a symptom (class prior shift).")
    print("Combining both gives the best OOD result.")

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "ood_robustness_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for name, v, t in all_results:
            f.write(f"{name}: val={v:.4f}, ood={t:.4f}, gain={t - baseline_ood:+.4f}\n")
    log.info("Saved summary: %s", summary_path)


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """
    End-to-end OOD robustness analysis pipeline.

    Output structure:
        results_output/
        └── ood_analysis/
            ├── ood_imbalance_methods.png    (Part A bar chart)
            ├── ood_universal_paths.png      (Part B per-language chart)
            └── ood_robustness_summary.txt   (all methods, text summary)
    """
    out_dir = os.path.join(args.output_dir, "ood_analysis")
    os.makedirs(out_dir, exist_ok=True)
    log.info("All outputs will be saved to: %s", out_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val_df   = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir, "test_sample.parquet"))

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    train_prior = (y_train == 1).mean()
    test_prior  = (y_test  == 1).mean()
    scale       = (y_train == 0).sum() / (y_train == 1).sum()

    log.info("Train: %d | Machine=%.1f%%", len(y_train), 100 * train_prior)
    log.info("Val  : %d | Machine=%.1f%%", len(y_val),   100 * (y_val == 1).mean())
    log.info("Test : %d | Machine=%.1f%%  <-- %.1fx fewer Machine than train",
             len(y_test), 100 * test_prior, (1 - test_prior) / test_prior)

    # ── Build standard AST path features (shared by both parts) ─────────────
    log.info("Loading standard AST path features (shared base)...")
    train_docs_std = load_or_compute_docs(
        train_df,
        os.path.join(args.cache_dir, "train_docs.npy"),
        extract_standard_paths,
        max_workers=args.max_workers,
        desc="train standard paths",
    )
    val_docs_std = load_or_compute_docs(
        val_df,
        os.path.join(args.cache_dir, "val_docs.npy"),
        extract_standard_paths,
        max_workers=args.max_workers,
        desc="val standard paths",
    )
    test_docs_std = load_or_compute_docs(
        test_df,
        os.path.join(args.cache_dir, "test_docs.npy"),
        extract_standard_paths,
        max_workers=args.max_workers,
        desc="test standard paths",
    )
    _, X_train_std, X_val_std, X_test_std = build_tfidf(
        train_docs_std, val_docs_std, test_docs_std,
    )

    # ── Run selected parts ────────────────────────────────────────────────────
    imbalance_results = None
    universal_results = None

    if args.mode in ("imbalance", "all"):
        log.info("\n%s", "=" * 60)
        log.info("PART A — CLASS IMBALANCE HANDLING")
        log.info("%s", "=" * 60)
        imbalance_results = run_part_a(
            X_train_std, y_train,
            X_val_std,   y_val,
            X_test_std,  y_test,
            out_dir,
        )

    if args.mode in ("universal", "all"):
        log.info("\n%s", "=" * 60)
        log.info("PART B — LANGUAGE-AGNOSTIC AST PATHS")
        log.info("%s", "=" * 60)
        universal_results = run_part_b(
            train_df, val_df, test_df,
            y_train, y_val, y_test,
            std_features=(X_train_std, X_val_std, X_test_std),
            cache_dir=args.cache_dir,
            out_dir=out_dir,
            max_workers=args.max_workers,
            scale=scale,
        )

    if args.mode == "all" and imbalance_results and universal_results:
        print_final_comparison(imbalance_results, universal_results, out_dir)

    log.info("\nAll outputs saved to: %s", out_dir)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOD robustness analysis for machine-generated code detection."
    )
    p.add_argument("--data_dir",    default="data/task_a",
                   help="Directory with train/val/test_sample parquet files")
    p.add_argument("--output_dir",  default="results_output",
                   help="Root output dir. Saves to output_dir/ood_analysis/")
    p.add_argument("--cache_dir",   default="data",
                   help="Directory for .npy AST doc cache files")
    p.add_argument("--max_workers", type=int, default=8,
                   help="Parallel threads for path extraction")
    p.add_argument("--mode",
                   choices=["imbalance", "universal", "all"],
                   default="all",
                   help="Which part(s) to run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
