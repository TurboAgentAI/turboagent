"""Sphinx configuration for TurboAgent documentation."""

project = "TurboAgent"
copyright = "2026, TurboAgent Contributors"
author = "TurboAgent Contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "TurboAgent"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
