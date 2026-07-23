# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Macht den src-Ordner für Sphinx und autodoc auffindbar
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GroupedTransform"
copyright = "2025, Felix"
author = "Felix"
release = "29.07.2025"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Unterstützt Google/NumPy-Style Docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",  
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# GroupedTransform.py importiert torch und pykeops, die aber nur in den
# Methodenkoerpern benutzt werden. Beide sind sehr gross und werden fuer die
# Docstrings nicht gebraucht, deshalb werden sie hier nur simuliert.
autodoc_mock_imports = ["torch", "pykeops"]

templates_path = []
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 5,
}
html_css_files = [
    'custom.css',
]
