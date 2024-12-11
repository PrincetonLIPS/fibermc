import sys
from pathlib import Path

PROJECT_DIRECTORY: Path = Path(__file__).parent.parent
SOURCE_DIRECTORY: Path = PROJECT_DIRECTORY / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

project = "Fibermc"
author = "Nick Richardson"
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"
html_logo = ""
html_favicon = ""
