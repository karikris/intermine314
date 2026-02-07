# Configuration file for the Sphinx documentation builder.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

project = "intermine314"
copyright = "2026, Monash University, Plant Energy and Biotechnology Lab"
author = "Kris Kari; Dr. Maria Ermakova; Plant Energy and Biotechnology Lab, Monash University"
version = "0.1"
release = "0.1.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

html_theme = "alabaster"
html_static_path = ["_static"]
htmlhelp_basename = "intermine314doc"

autodoc_member_order = "bysource"
autodoc_mock_imports = ["polars", "duckdb", "matplotlib", "numpy", "lxml"]

latex_documents = [
    (master_doc, "intermine314.tex", "intermine314 Documentation", author, "manual"),
]

man_pages = [(master_doc, "intermine314", "intermine314 Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "intermine314",
        "intermine314 Documentation",
        author,
        "intermine314",
        "Python 3.14+ InterMine client with Polars, Parquet, and DuckDB workflows.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]

intersphinx_mapping = {"python": ("https://docs.python.org/3.14", None)}

todo_include_todos = False
