"""
src/parsing/ast_scalar.py
===========================
Regex + Python AST Structural Feature Extraction for Machine-Generated Code Detection.

What this module does:
    This was the earliest structural feature experiment in the project — the
    diagnostic stage that motivated the more advanced tree-sitter approach in
    src/feature_extraction/ast_scalar.py.

    It implements two feature extractors and a diagnostic pipeline:

    Extractor 1 — Regex Structural Features (15 features, language-agnostic)
        Simple regex-based surface features extracted from raw source text.
        No parser required — works on any language. Captures:
            line/character counts, loop/conditional keyword counts, brace/
            semicolon counts, comment markers, blank lines, indentation spread.

        Used to establish: "do even surface structural stats discriminate?"

    Extractor 2 — Python AST Features — Basic (8 features)
        Uses Python's built-in ast module to parse Python code only.
        Captures: total nodes, max depth, node-type entropy, function/loop/
        if/assignment/return counts.

        Used to establish: "do AST structural features outperform char TF-IDF OOD?"

    Extractor 3 — Python AST Features — Enhanced (15 features)
        Extends the basic extractor with richer statistics:
            depth variance, branching factor (avg + max), leaf ratio,
            cyclomatic complexity proxy.

        Used to establish: "does deeper structural characterisation further
        improve OOD robustness?"

    Diagnostic experiments run:
        A. All-language regex features → Logistic Regression
           Establishes structural baseline across all languages.

        B. Python-only char TF-IDF → OOD F1 ~0.49
           Separates language shift from domain shift.

        C. Python-only basic AST features → OOD F1 ~0.53
           Confirms structural features generalise better than lexical.

        D. Python-only enhanced AST features → OOD F1 ~0.585
           Confirms richer structural characterisation further helps OOD.

Key findings:
    - Cross-language shift explains PART of OOD degradation: restricting to
      Python-only improves OOD F1 from ~0.38 → ~0.49.
    - AST structural features (Python-only) improve OOD further: ~0.49 → ~0.53.
    - Enhanced AST features push further: ~0.53 → ~0.585.
    - BUT in-domain performance drops sharply: lexical ~0.97 vs AST ~0.72.
    - Conclusion: structural features generalise better but discriminate
      less sharply in-distribution. Both language shift AND domain/style
      shift contribute to OOD failure.

    This motivated moving to tree-sitter (src/feature_extraction/ast_scalar.py)
    which supports all languages — not just Python — with richer 18-feature
    scalar extraction.

Usage:
    # Full diagnostic pipeline (all 4 experiments)
    python -m src.parsing.ast_scalar --mode all

    # Only regex structural features (fastest, all languages)
    python -m src.parsing.ast_scalar --mode regex

    # Only Python AST experiments (basic + enhanced)
    python -m src.parsing.ast_scalar --mode ast

    # Python-only language-shift diagnostic
    python -m src.parsing.ast_scalar --mode diagnostic

Arguments:
    --data_dir   Path to parquet files. Default: data/task_a
    --output_dir Root output directory. Default: results_output
                 Outputs saved to: results_output/ast_structural/
    --mode       regex | diagnostic | ast | all. Default: all
"""

import os
import re
import ast
import math
import argparse
import logging
import warnings

import numpy as np
import pandas as pd

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Regex structural features (all languages, no parser needed)
# ---------------------------------------------------------------------------

REGEX_FEATURE_NAMES = [
    "num_lines",
    "num_chars",
    "avg_line_length",
    "num_for",
    "num_while",
    "num_if",
    "num_def",
    "num_function",
    "num_open_brace",
    "num_close_brace",
    "num_semicolon",
    "num_hash_comment",
    "num_slash_comment",
    "blank_lines",
    "indent_std",
]


def extract_regex_features(code: str) -> list[float]:
    """
    Extract 15 surface-level structural features using regex + string ops.

    No AST parser required — works on any programming language.
    These features capture the gross shape of code: how long it is,
    how indented, how many control structures, comment style, etc.

    Features:
        num_lines        — total line count
        num_chars        — total character count
        avg_line_length  — mean characters per line
        num_for          — count of 'for' keyword occurrences
        num_while        — count of 'while' keyword occurrences
        num_if           — count of 'if' keyword occurrences
        num_def          — count of 'def' keyword (Python functions)
        num_function     — count of 'function' keyword (JS/PHP/Go)
        num_open_brace   — count of '{' (C-family block openers)
        num_close_brace  — count of '}' (C-family block closers)
        num_semicolon    — count of ';' (statement terminators)
        num_hash_comment — count of '#' (Python/Ruby/Perl comments)
        num_slash_comment— count of '//' (C-family comments)
        blank_lines      — count of empty lines
        indent_std       — standard deviation of indentation depth per line

    Args:
        code: Raw source code string (any language).

    Returns:
        List of 15 float values.
    """
    lines = code.split("\n")
    num_lines = len(lines)

    num_chars        = len(code)
    avg_line_length  = float(np.mean([len(l) for l in lines])) if num_lines > 0 else 0.0
    num_for          = len(re.findall(r"\bfor\b",      code))
    num_while        = len(re.findall(r"\bwhile\b",    code))
    num_if           = len(re.findall(r"\bif\b",       code))
    num_def          = len(re.findall(r"\bdef\b",      code))
    num_function     = len(re.findall(r"\bfunction\b", code))
    num_open_brace   = code.count("{")
    num_close_brace  = code.count("}")
    num_semicolon    = code.count(";")
    num_hash_comment = code.count("#")
    num_slash_comment= code.count("//")
    blank_lines      = sum(1 for l in lines if l.strip() == "")

    indent_levels = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip() != ""]
    indent_std    = float(np.std(indent_levels)) if indent_levels else 0.0

    return [
        float(num_lines),
        float(num_chars),
        avg_line_length,
        float(num_for),
        float(num_while),
        float(num_if),
        float(num_def),
        float(num_function),
        float(num_open_brace),
        float(num_close_brace),
        float(num_semicolon),
        float(num_hash_comment),
        float(num_slash_comment),
        float(blank_lines),
        indent_std,
    ]


# ---------------------------------------------------------------------------
# 2. Python AST features — basic (8 features)
# ---------------------------------------------------------------------------

BASIC_AST_FEATURE_NAMES = [
    "total_nodes",
    "max_depth",
    "entropy",
    "num_functions",
    "num_loops",
    "num_ifs",
    "num_assign",
    "num_return",
]


def extract_basic_ast_features(code: str) -> list[float]:
    """
    Extract 8 structural features using Python's built-in ast module.

    Python-only — returns [0]*8 for non-Python code or parse failures.

    Features:
        total_nodes   — total number of AST nodes
        max_depth     — maximum nesting depth
        entropy       — Shannon entropy of node-type distribution
                        (high → many different node types, diverse structure)
                        (low  → few types repeated, structured/uniform)
        num_functions — count of FunctionDef nodes
        num_loops     — count of For + While nodes
        num_ifs       — count of If nodes
        num_assign    — count of Assign nodes
        num_return    — count of Return nodes

    Args:
        code: Python source code string.

    Returns:
        List of 8 float values.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return [0.0] * 8

    node_types = []
    max_depth  = 0

    def traverse(node, depth: int = 0) -> None:
        nonlocal max_depth
        node_types.append(type(node).__name__)
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            traverse(child, depth + 1)

    traverse(tree)
    total_nodes = len(node_types)

    if total_nodes == 0:
        return [0.0] * 8

    counter = Counter(node_types)
    probs   = [v / total_nodes for v in counter.values()]
    entropy = -sum(p * math.log(p + 1e-10) for p in probs)

    num_functions = sum(1 for n in node_types if n == "FunctionDef")
    num_loops     = sum(1 for n in node_types if n in ("For", "While"))
    num_ifs       = sum(1 for n in node_types if n == "If")
    num_assign    = sum(1 for n in node_types if n == "Assign")
    num_return    = sum(1 for n in node_types if n == "Return")

    return [
        float(total_nodes),
        float(max_depth),
        entropy,
        float(num_functions),
        float(num_loops),
        float(num_ifs),
        float(num_assign),
        float(num_return),
    ]


# ---------------------------------------------------------------------------
# 3. Python AST features — enhanced (15 features)
# ---------------------------------------------------------------------------

ENHANCED_AST_FEATURE_NAMES = [
    "total_nodes",
    "entropy",
    "max_depth",
    "avg_depth",
    "depth_std",
    "avg_branching",
    "max_branching",
    "num_functions",
    "num_loops",
    "num_ifs",
    "num_assign",
    "num_return",
    "num_leaves",
    "leaf_ratio",
    "cyclomatic",
]


def extract_enhanced_ast_features(code: str) -> list[float]:
    """
    Extract 15 enhanced structural features using Python's built-in ast module.

    Extends the basic extractor with depth statistics, branching factor,
    leaf ratio, and cyclomatic complexity proxy.

    Python-only — returns [0]*15 for non-Python code or parse failures.

    Features (additions over basic):
        avg_depth     — mean node depth across the tree
        depth_std     — standard deviation of node depths
        avg_branching — mean number of children per node
        max_branching — maximum children any single node has
        num_leaves    — nodes with zero children
        leaf_ratio    — leaf_nodes / total_nodes
        cyclomatic    — num_loops + num_ifs + 1 (McCabe complexity proxy)

    Args:
        code: Python source code string.

    Returns:
        List of 15 float values.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return [0.0] * 15

    node_types   = []
    depths       = []
    child_counts = []

    def traverse(node, depth: int = 0) -> None:
        node_types.append(type(node).__name__)
        depths.append(depth)
        children = list(ast.iter_child_nodes(node))
        child_counts.append(len(children))
        for child in children:
            traverse(child, depth + 1)

    traverse(tree)
    total_nodes = len(node_types)

    if total_nodes == 0:
        return [0.0] * 15

    counter = Counter(node_types)
    probs   = [v / total_nodes for v in counter.values()]
    entropy = -sum(p * math.log(p + 1e-10) for p in probs)

    max_depth    = float(max(depths))
    avg_depth    = float(np.mean(depths))
    depth_std    = float(np.std(depths))
    avg_branch   = float(np.mean(child_counts))
    max_branch   = float(max(child_counts))

    num_functions = float(counter.get("FunctionDef", 0))
    num_loops     = float(counter.get("For", 0) + counter.get("While", 0))
    num_ifs       = float(counter.get("If", 0))
    num_assign    = float(counter.get("Assign", 0))
    num_return    = float(counter.get("Return", 0))
    num_leaves    = float(sum(1 for c in child_counts if c == 0))
    leaf_ratio    = num_leaves / total_nodes
    cyclomatic    = num_loops + num_ifs + 1.0

    return [
        float(total_nodes),
        entropy,
        max_depth,
        avg_depth,
        depth_std,
        avg_branch,
        max_branch,
        num_functions,
        num_loops,
        num_ifs,
        num_assign,
        num_return,
        num_leaves,
        leaf_ratio,
        cyclomatic,
    ]


# ---------------------------------------------------------------------------
# 4. Training + evaluation helpers
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_eval:  np.ndarray,
    y_eval:  np.ndarray,
    label:   str,
    out_dir: str,
) -> dict:
    """
    Scale → fit Logistic Regression → evaluate on val + OOD test.

    Args:
        label:   Display name for logging and report filename.
        out_dir: Directory to save the .txt report.

    Returns:
        Dict with val_f1 and ood_f1.
    """
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_train)
    X_v     = scaler.transform(X_val)
    X_e     = scaler.transform(X_eval)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    log.info("Training %s...", label)
    model.fit(X_tr, y_train)

    val_preds  = model.predict(X_v)
    eval_preds = model.predict(X_e)

    val_f1  = f1_score(y_val,  val_preds,  zero_division=0)
    ood_f1  = f1_score(y_eval, eval_preds, zero_division=0)
    val_acc = accuracy_score(y_val,  val_preds)
    ood_acc = accuracy_score(y_eval, eval_preds)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Val  — Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
    print(f"  OOD  — Accuracy: {ood_acc:.4f} | F1: {ood_f1:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds,
                                target_names=["Human", "Machine"], zero_division=0))
    print("OOD Classification Report:")
    print(classification_report(y_eval, eval_preds,
                                target_names=["Human", "Machine"], zero_division=0))

    os.makedirs(out_dir, exist_ok=True)
    safe  = label.replace(" ", "_").replace("(", "").replace(")", "")
    with open(os.path.join(out_dir, f"{safe}_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"{label}\n{'='*55}\n")
        f.write(f"Val  Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}\n")
        f.write(f"OOD  Accuracy: {ood_acc:.4f} | F1: {ood_f1:.4f}\n\n")
        f.write("Validation:\n")
        f.write(classification_report(y_val, val_preds,
                                      target_names=["Human", "Machine"], zero_division=0))
        f.write("\nOOD:\n")
        f.write(classification_report(y_eval, eval_preds,
                                      target_names=["Human", "Machine"], zero_division=0))
    log.info("Saved report: %s/%s_report.txt", out_dir, safe)

    return {"val_f1": val_f1, "ood_f1": ood_f1}


# ---------------------------------------------------------------------------
# 5. Experiment A — Regex structural features (all languages)
# ---------------------------------------------------------------------------

def run_regex_experiment(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    eval_df:  pd.DataFrame,
    out_dir:  str,
) -> dict:
    """
    Experiment A: Regex surface features on all languages.

    Establishes the simplest possible structural baseline. No parser, works
    on every language in the dataset.
    """
    log.info("\n=== Experiment A: Regex Structural Features (All Languages) ===")

    log.info("Extracting regex features (train)...")
    X_train = np.array(train_df["code"].apply(extract_regex_features).tolist())
    log.info("Extracting regex features (val)...")
    X_val   = np.array(val_df["code"].apply(extract_regex_features).tolist())
    log.info("Extracting regex features (eval)...")
    X_eval  = np.array(eval_df["code"].apply(extract_regex_features).tolist())

    log.info("Feature shape: %s", X_train.shape)

    return train_and_evaluate(
        X_train, train_df["label"].values,
        X_val,   val_df["label"].values,
        X_eval,  eval_df["label"].values,
        label="Regex_Structural_All_Languages",
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# 6. Experiment B — Python-only language-shift diagnostic (char TF-IDF)
# ---------------------------------------------------------------------------

def run_diagnostic_experiment(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    eval_df:  pd.DataFrame,
    out_dir:  str,
) -> dict:
    """
    Experiment B: Python-only char TF-IDF to isolate language-shift vs domain-shift.

    Hypothesis:
        If OOD F1 improves strongly when restricting to Python-only, then
        language shift is the dominant problem. If it remains poor, domain/
        style shift is also a major factor.

    Result: OOD F1 improves from ~0.38 → ~0.49 (Python-only).
    Conclusion: language shift is real but not sufficient to explain failure.
    """
    log.info("\n=== Experiment B: Python-Only Language Shift Diagnostic ===")

    train_py = train_df[train_df["language"] == "Python"]
    val_py   = val_df[val_df["language"]   == "Python"]
    eval_py  = eval_df[eval_df["language"] == "Python"]

    log.info(
        "Python subsets — Train: %d | Val: %d | Eval: %d",
        len(train_py), len(val_py), len(eval_py),
    )

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=100000,
        min_df=5,
        sublinear_tf=True,
    )
    log.info("Fitting Python-only char TF-IDF...")
    X_train = vec.fit_transform(train_py["code"])
    X_val   = vec.transform(val_py["code"])
    X_eval  = vec.transform(eval_py["code"])

    y_train = train_py["label"].values
    y_val   = val_py["label"].values
    y_eval  = eval_py["label"].values

    scaler  = StandardScaler(with_mean=False)   # sparse-safe
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_eval  = scaler.transform(X_eval)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    log.info("Training Python-only char TF-IDF model...")
    model.fit(X_train, y_train)

    val_f1  = f1_score(y_val,  model.predict(X_val),  zero_division=0)
    ood_f1  = f1_score(y_eval, model.predict(X_eval), zero_division=0)

    print(f"\nPython-only Char TF-IDF — Val F1: {val_f1:.4f} | OOD F1: {ood_f1:.4f}")
    print("Interpretation: OOD F1 ~0.49 vs ~0.38 full-language → language shift is real")
    print("but domain/style shift persists within Python (still far below val ~0.97)")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "diagnostic_python_char_report.txt"), "w") as f:
        f.write(f"Python-only Char TF-IDF\nVal F1: {val_f1:.4f}\nOOD F1: {ood_f1:.4f}\n")
    log.info("Saved: %s/diagnostic_python_char_report.txt", out_dir)

    return {"val_f1": val_f1, "ood_f1": ood_f1}


# ---------------------------------------------------------------------------
# 7. Experiments C & D — Python AST features (basic + enhanced)
# ---------------------------------------------------------------------------

def run_ast_experiments(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    eval_df:  pd.DataFrame,
    out_dir:  str,
) -> dict:
    """
    Experiments C & D: Python-only AST structural features.

    Uses Python's built-in ast module (not tree-sitter).
    Restricted to Python-only rows to match the diagnostic baseline.

    Experiment C — Basic AST (8 features):
        Confirms structural features outperform char TF-IDF on OOD.
        Expected: OOD F1 ~0.53 vs ~0.49 for char TF-IDF.

    Experiment D — Enhanced AST (15 features):
        Adds depth/branching statistics and cyclomatic complexity proxy.
        Expected: OOD F1 ~0.585, confirming deeper structural features
        generalise better under domain shift.
    """
    log.info("\n=== Experiments C & D: Python AST Features ===")

    train_py = train_df[train_df["language"] == "Python"]
    val_py   = val_df[val_df["language"]   == "Python"]
    eval_py  = eval_df[eval_df["language"] == "Python"]

    y_train = train_py["label"].values
    y_val   = val_py["label"].values
    y_eval  = eval_py["label"].values

    results = {}

    # ── Experiment C: Basic AST (8 features) ─────────────────────────────────
    log.info("Extracting basic AST features (train)...")
    X_tr_b = np.array(train_py["code"].apply(extract_basic_ast_features).tolist())
    log.info("Extracting basic AST features (val)...")
    X_v_b  = np.array(val_py["code"].apply(extract_basic_ast_features).tolist())
    log.info("Extracting basic AST features (eval)...")
    X_e_b  = np.array(eval_py["code"].apply(extract_basic_ast_features).tolist())

    results["basic_ast"] = train_and_evaluate(
        X_tr_b, y_train, X_v_b, y_val, X_e_b, y_eval,
        label="Python_Basic_AST_8_features",
        out_dir=out_dir,
    )

    # ── Experiment D: Enhanced AST (15 features) ──────────────────────────────
    log.info("Extracting enhanced AST features (train)...")
    X_tr_e = np.array(train_py["code"].apply(extract_enhanced_ast_features).tolist())
    log.info("Extracting enhanced AST features (val)...")
    X_v_e  = np.array(val_py["code"].apply(extract_enhanced_ast_features).tolist())
    log.info("Extracting enhanced AST features (eval)...")
    X_e_e  = np.array(eval_py["code"].apply(extract_enhanced_ast_features).tolist())

    results["enhanced_ast"] = train_and_evaluate(
        X_tr_e, y_train, X_v_e, y_val, X_e_e, y_eval,
        label="Python_Enhanced_AST_15_features",
        out_dir=out_dir,
    )

    return results


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """
    End-to-end diagnostic structural feature pipeline.

    Run order and dependency:
        all → A (regex) → B (diagnostic) → C+D (AST)
        Each experiment is independent; B is recommended before C/D to
        understand the Python-only baseline being compared against.

    Output structure:
        results_output/
        └── ast_structural/
            ├── Regex_Structural_All_Languages_report.txt
            ├── diagnostic_python_char_report.txt
            ├── Python_Basic_AST_8_features_report.txt
            └── Python_Enhanced_AST_15_features_report.txt
    """
    out_dir = os.path.join(args.output_dir, "ast_structural")
    os.makedirs(out_dir, exist_ok=True)
    log.info("All outputs will be saved to: %s", out_dir)

    log.info("Loading datasets from: %s", args.data_dir)
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val_df   = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    eval_df  = pd.read_parquet(os.path.join(args.data_dir, "test_sample.parquet"))

    log.info("Train: %d | Val: %d | Eval: %d",
             len(train_df), len(val_df), len(eval_df))

    all_results = {}

    if args.mode in ("regex", "all"):
        all_results["regex"] = run_regex_experiment(
            train_df, val_df, eval_df, out_dir,
        )

    if args.mode in ("diagnostic", "all"):
        all_results["diagnostic"] = run_diagnostic_experiment(
            train_df, val_df, eval_df, out_dir,
        )

    if args.mode in ("ast", "all"):
        all_results.update(run_ast_experiments(
            train_df, val_df, eval_df, out_dir,
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY — OOD F1 progression")
        print("=" * 60)
        print(f"{'Experiment':<45} {'Val F1':>8} {'OOD F1':>8}")
        print("-" * 60)
        labels = {
            "diagnostic":   "Python-only Char TF-IDF (lang-shift baseline)",
            "basic_ast":    "Python Basic AST (8 features)",
            "enhanced_ast": "Python Enhanced AST (15 features)",
            "regex":        "Regex Structural All Languages",
        }
        for key, res in all_results.items():
            name = labels.get(key, key)
            print(f"{name:<45} {res['val_f1']:>8.4f} {res['ood_f1']:>8.4f}")

    log.info("\nAll outputs saved to: %s", out_dir)


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regex + Python AST structural feature diagnostic pipeline."
    )
    p.add_argument("--data_dir",   default="data/task_a",
                   help="Directory with train/val/test_sample parquet files")
    p.add_argument("--output_dir", default="results_output",
                   help="Root output dir. Saves to output_dir/ast_structural/")
    p.add_argument("--mode",
                   choices=["regex", "diagnostic", "ast", "all"],
                   default="all",
                   help="Which experiment(s) to run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
