# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(''))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../cvt'))

print(sys.path)
print(sys.version)
import cvt

# -- Project information -----------------------------------------------------

project = 'cvlab_toolbox'
copyright = '2019, Atom Scott'
author = 'Atom Scott'

# The full version, including alpha/beta/rc tags
release = '0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'recommonmark',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxcontrib.bibtex',
    'sphinx_gallery.gen_gallery'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If you have your own conf.py file, it overrides Read the Doc's default
# conf.py. By default, Sphinx expects the master doc to be contents.
# Read the Docs will set master doc to index instead.
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'prev_next_buttons_location': 'bottom',

    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_favicon = '_static/logo.png'
html_logo = '_static/logo.png'


# The following configuration declares the location of the 
# ‘examples’ directory ('example_dirs') to be ../examples 
# and the ‘output’ directory ('gallery_dirs') to be auto_examples:
sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'examples_scripts',  # path to where to save gallery generated output
}

# generate autosummary even if no references
autosummary_generate = True