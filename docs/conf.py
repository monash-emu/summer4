"""Sphinx configuration for the summer4 documentation."""

from __future__ import annotations

import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
os.environ["PYTHONPATH"] = _project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

project = "summer4"
copyright = "2026, Monash EMU"
author = "Monash EMU"
release = "0.2.0a1"
version = "0.2.0a1"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".jupyter_cache"]

master_doc = "index"
language = "en"

# MyST
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "linkify",
]
myst_heading_anchors = 3

# Notebook execution: docs are runnable, so a broken cell fails the build.
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True
nb_execution_cache_path = os.path.join(os.path.dirname(__file__), "_build", ".jupyter_cache")

# Autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
# PEP-695 type params (e.g. Groups[V]) are not visible to the typehints
# resolver yet; the annotations themselves are correct for mypy.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

pygments_style = "friendly"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Plotly figures carry their own CDN loader: notebooks set the
# "notebook_connected" renderer, so each output embeds a script tag for the
# plotly.js matching the installed plotly. A global html_js_files pin would
# load a second, different major version alongside it. See dev/plotting.

html_theme_options = {
    "navigation_depth": 3,
    "show_toc_level": 2,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/monash-emu/summer4",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {"text": "summer4"},
}

html_context = {"default_mode": "light"}
