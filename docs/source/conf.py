# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("/Users/johnlevy/opt/anaconda3/envs/climate_risk"))
sys.path.insert(
    0,
    os.path.abspath("/Users/johnlevy/opt/anaconda3/envs/climate_risk/lib/python3.12"),
)
sys.path.insert(
    0, os.path.abspath("/Users/johnlevy/opt/anaconda3/envs/climate_risk/lib/python3.12")
)
sys.path.insert(
    0,
    os.path.abspath(
        "/Users/johnlevy/opt/anaconda3/envs/climate_risk/lib/python3.12/lib-dynload"
    ),
)
sys.path.insert(
    0,
    os.path.abspath(
        "/Users/johnlevy/opt/anaconda3/envs/climate_risk/lib/python3.12/site-packages"
    ),
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Maritime Trade Risk"
copyright = "2024, John Levy"
author = "John Levy"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
]
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
