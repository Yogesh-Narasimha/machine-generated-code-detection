from tree_sitter import Parser
from tree_sitter_languages import get_language


def get_language_parser(language_name):

    language_map = {
        "Python": "python",
        "C++": "cpp",
        "Java": "java"
    }

    lang = language_map.get(language_name)

    if lang is None:
        return None

    parser = Parser()
    parser.set_language(get_language(lang))

    return parser


def parse_code(code, language):

    parser = get_language_parser(language)

    if parser is None:
        return None

    tree = parser.parse(bytes(code, "utf8"))

    return tree