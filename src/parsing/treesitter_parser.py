"""
src/parsing/treesitter_parser.py
==================================
Tree-sitter parser initialisation and code parsing utilities.

This module is the single place in the project that manages tree-sitter
parsers. All other modules (tfidf_paths.py, ast_features.py, ood_robustness.py)
import from here instead of initialising parsers themselves.

Supported languages
-------------------
Seen (training languages):
    Python, Java, C++

Unseen / OOD (test-only languages):
    Go, PHP, C#, JavaScript, C

Usage
-----
    from src.parsing.treesitter_parser import get_parsers, parse_code

    # Get parsers for all supported languages
    parsers = get_parsers()

    # Parse a single snippet
    tree = parse_code("def foo(): return 1", "Python", parsers)
    if tree is not None:
        root = tree.root_node

    # Get only the seen (training) languages
    seen_parsers = get_parsers(languages_only=["Python", "Java", "C++"])
"""

from __future__ import annotations

import logging
from tree_sitter_languages import get_parser

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language map
# ---------------------------------------------------------------------------

# Maps the language name used in our dataset to the tree-sitter language name.
# Seen = languages present in training data.
# Unseen = OOD languages present only in test_sample.

SEEN_LANGUAGES: dict[str, str] = {
    "Python": "python",
    "Java":   "java",
    "C++":    "cpp",
}

UNSEEN_LANGUAGES: dict[str, str] = {
    "Go":         "go",
    "PHP":        "php",
    "C#":         "c_sharp",
    "JavaScript": "javascript",
    "C":          "c",
}

ALL_LANGUAGES: dict[str, str] = {**SEEN_LANGUAGES, **UNSEEN_LANGUAGES}


# ---------------------------------------------------------------------------
# Parser cache — built once at module level
# ---------------------------------------------------------------------------
# Parsers are cached here so every call to get_parsers() returns the same
# objects. Building a parser is not expensive, but doing it 500K times
# (once per sample) would add unnecessary overhead.

_PARSER_CACHE: dict[str, object] = {}


def _load_parser(lang_name: str, ts_name: str) -> object | None:
    """
    Load a single tree-sitter parser, with caching and error handling.

    Args:
        lang_name: Human-readable language name (e.g. "Python").
        ts_name:   tree-sitter language identifier (e.g. "python").

    Returns:
        A tree-sitter Parser object, or None if loading fails.
    """
    if lang_name in _PARSER_CACHE:
        return _PARSER_CACHE[lang_name]

    try:
        # get_parser() is equivalent to:
        #   p = Parser()
        #   p.set_language(get_language(ts_name))
        #   return p
        # We use get_parser() as the canonical approach throughout this project.
        parser = get_parser(ts_name)
        _PARSER_CACHE[lang_name] = parser
        return parser
    except Exception as exc:
        log.warning("Could not load tree-sitter parser for %s (%s): %s", lang_name, ts_name, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_parsers(languages_only: list[str] | None = None) -> dict[str, object]:
    """
    Return a dict of language_name -> tree-sitter Parser for all supported languages.

    Parsers are cached after the first call so this is safe to call repeatedly.

    Args:
        languages_only: If provided, only return parsers for these language names.
                        If None, returns parsers for ALL supported languages
                        (both seen and unseen).

    Returns:
        Dict mapping language name (e.g. "Python") to its parser object.
        Languages that fail to load are omitted with a warning — they do not
        raise an exception so the rest of the pipeline can continue.

    Examples:
        # All languages (seen + OOD)
        parsers = get_parsers()

        # Only training languages
        parsers = get_parsers(languages_only=["Python", "Java", "C++"])
    """
    target = ALL_LANGUAGES if languages_only is None else {
        lang: ALL_LANGUAGES[lang]
        for lang in languages_only
        if lang in ALL_LANGUAGES
    }

    result = {}
    for lang_name, ts_name in target.items():
        parser = _load_parser(lang_name, ts_name)
        if parser is not None:
            result[lang_name] = parser

    log.debug("Loaded parsers for: %s", list(result.keys()))
    return result


def parse_code(code: str, language: str, parsers: dict) -> object | None:
    """
    Parse source code into an AST using the appropriate tree-sitter parser.

    Args:
        code:     Raw source code string.
        language: Language name (e.g. "Python", "Go"). Must be a key in parsers.
        parsers:  Dict of language_name -> parser (from get_parsers()).

    Returns:
        A tree-sitter Tree object if parsing succeeds.
        None if the language is unsupported or parsing fails.

    Notes:
        - tree-sitter parsers are error-tolerant: they always produce a tree,
          even for syntactically invalid code. The tree may contain ERROR nodes.
        - Call tree.root_node on the returned object to start traversal.

    Example:
        parsers = get_parsers()
        tree = parse_code("def foo(): return 1", "Python", parsers)
        if tree:
            root = tree.root_node   # start AST traversal here
    """
    parser = parsers.get(language)

    if parser is None:
        return None

    try:
        tree = parser.parse(bytes(code, "utf8"))
        return tree
    except Exception as exc:
        log.debug("Parsing failed for language=%s: %s", language, exc)
        return None


def get_supported_languages() -> dict[str, list[str]]:
    """
    Return a summary of all supported languages split by seen/unseen status.

    Returns:
        Dict with keys 'seen' and 'unseen', each containing a list of
        language names.

    Example:
        info = get_supported_languages()
        print(info['seen'])    # ['Python', 'Java', 'C++']
        print(info['unseen'])  # ['Go', 'PHP', 'C#', 'JavaScript', 'C']
    """
    return {
        "seen":   list(SEEN_LANGUAGES.keys()),
        "unseen": list(UNSEEN_LANGUAGES.keys()),
    }


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Loading all parsers...")
    parsers = get_parsers()
    print(f"Successfully loaded {len(parsers)} parsers: {list(parsers.keys())}")
    print()

    # Test parse on a small Python snippet
    test_code = "def add(a, b):\n    return a + b"
    tree = parse_code(test_code, "Python", parsers)
    if tree:
        print(f"Python parse OK — root node type: {tree.root_node.type}")
        print(f"Children: {[c.type for c in tree.root_node.children]}")
    else:
        print("Python parse FAILED")

    # Test unsupported language gracefully
    result = parse_code("package main", "Rust", parsers)
    print(f"\nUnsupported language (Rust) returns: {result}  (expected None)")
