# Single-Cell RNA-seq Analysis - Installation Guide

This guide will help you install all the required dependencies to fix the import errors in your single-cell RNA-seq analysis project.

## Overview

Your [`dataset/datasetprocessing.py`](dataset/datasetprocessing.py) file requires several specialized libraries for single-cell analysis, machine learning, and data visualization. This guide provides multiple installation methods to get your environment working.

## Required Dependencies

Based on your code analysis, here are the required packages:

### Core Scientific Computing
- `numpy` - Numerical computing foundation
- `pandas` - Data manipulation and analysis
- `scipy` - Scientific computing utilities

### Single-Cell Analysis
- `scanpy` - Single-cell analysis in Python (main package)
- `anndata` - Annotated data structures (included with scanpy)

### Machine Learning & Deep Learning
- `scikit-learn` - Machine learning utilities
- `torch` - PyTorch deep learning framework
- `pytorch-lightning` - High-level PyTorch wrapper

### Visualization
- `matplotlib` - Basic plotting library
- `seaborn` - Statistical data visualization

## Installation Methods

### Method 1: Conda Installation (Recommended)

Conda is recommended for scientific computing as it handles complex dependencies better.

```bash
# Create a new conda environment
conda create -n scrna-analysis python=3.9

# Activate the environment
conda activate scrna-analysis

# Install core scientific packages
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn

# Install single-cell analysis packages
conda install -c bioconda scanpy

# Install PyTorch (CPU version)
conda install pytorch pytorch-lightning -c pytorch

# For GPU support (if you have CUDA-compatible GPU)
# conda install pytorch pytorch-lightning pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Method 2: Pip Installation

If you prefer pip or don't have conda:

```bash
# Create virtual environment
python -m venv scrna-env

# Activate virtual environment
# On macOS/Linux:
source scrna-env/bin/activate
# On Windows:
# scrna-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install scanpy
pip install torch pytorch-lightning
```

### Method 3: Requirements File Content

Create a `requirements.txt` file with the following content:

```txt
# Core Scientific Computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Single-Cell Analysis
scanpy>=1.9.0
anndata>=0.8.0

# Deep Learning
torch>=1.12.0
pytorch-lightning>=1.7.0

# Additional dependencies that may be needed
leidenalg>=0.8.0
python-igraph>=0.9.0
```

Then install with:
```bash
pip install -r requirements.txt
```

## Verification Steps

After installation, verify everything works:

### 1. Test Basic Imports

Create a test script or run in Python:

```python
# Test all imports from your datasetprocessing.py
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("All imports successful!")
print(f"Scanpy version: {sc.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pl.__version__}")
```

### 2. Test Your Classes

```python
# Test your custom classes
from dataset.datasetprocessing import SingleCellDataset, SingleCellPreprocessor

# Create dummy data to test
import numpy as np
dummy_expression = np.random.randn(100, 50)
dummy_labels = np.random.randint(0, 3, 100)

# Test dataset class
dataset = SingleCellDataset(dummy_expression, dummy_labels)
print(f"Dataset length: {len(dataset)}")

# Test preprocessor
preprocessor = SingleCellPreprocessor()
print("Classes created successfully!")
```

## Troubleshooting Common Issues

### Issue 1: Scanpy Installation Problems
```bash
# If scanpy fails to install via pip, try:
pip install scanpy[leiden]
# or use conda:
conda install -c bioconda scanpy
```

### Issue 2: PyTorch Installation Issues
```bash
# For CPU-only PyTorch:
pip install torch pytorch-lightning --index-url https://download.pytorch.org/whl/cpu

# For CUDA support, check: https://pytorch.org/get-started/locally/
```

### Issue 3: Memory Issues with Large Datasets
- Consider installing `sparse` for memory-efficient operations:
```bash
pip install sparse
```

### Issue 4: Jupyter Notebook Support
If you plan to use Jupyter notebooks:
```bash
pip install jupyter ipykernel
# Register your environment as a kernel
python -m ipykernel install --user --name=scrna-analysis
```

## Environment Management

### Conda Environment Commands
```bash
# List environments
conda env list

# Activate environment
conda activate scrna-analysis

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n scrna-analysis
```

### Virtual Environment Commands
```bash
# Activate (macOS/Linux)
source scrna-env/bin/activate

# Activate (Windows)
scrna-env\Scripts\activate

# Deactivate
deactivate

# Remove environment
rm -rf scrna-env
```

## Next Steps

1. Choose your preferred installation method
2. Install all dependencies
3. Run the verification steps
4. Test your `dataset/datasetprocessing.py` file
5. If you encounter any issues, refer to the troubleshooting section

## Additional Resources

- [Scanpy Documentation](https://scanpy.readthedocs.io/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Single-Cell Analysis Best Practices](https://www.sc-best-practices.org/)

## System Requirements

- Python 3.8 or higher
- At least 8GB RAM (16GB+ recommended for large datasets)
- 2GB+ free disk space for all packages
- Optional: CUDA-compatible GPU for accelerated computing