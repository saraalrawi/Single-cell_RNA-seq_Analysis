"""
Single-Cell RNA-seq Analysis Package

A comprehensive toolkit for single-cell RNA sequencing data analysis,
including preprocessing, modeling, and visualization capabilities.
"""

__version__ = "0.1.0"
__author__ = "Single-Cell Analysis Team"
__email__ = "contact@scrna-analysis.com"

from .data import SingleCellDataset, SingleCellPreprocessor
from .models import scVAE, SingleCellAnalyzer
from .visualization import Visualizer

__all__ = [
    "SingleCellDataset",
    "SingleCellPreprocessor",
    "scVAE",
    "SingleCellAnalyzer",
    "Visualizer"
]