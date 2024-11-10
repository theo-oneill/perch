# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'perch'
copyright = "2024, O'Neill"
author = "O'Neill"

release = '0.1'
version = '0.1.0'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import mock

MOCK_MODULES = ['numpy', 'matplotlib', 'matplotlib.pyplot', 'cc3d', 'jax', 'jax.numpy', 'tqdm']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints'
]
autosummary_generate = True
autosummary_imported_members = True

autodoc_default_options = {
    'members': True,           # Document class members
    'undoc-members': True,     # Include even undocumented members
    'inherited-members': True, # Include inherited members
    'show-inheritance': True,  # Show class inheritance
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
