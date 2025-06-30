#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly
for the single-cell RNA-seq analysis project.
"""

import sys
from typing import List, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        __import__(module_name)
        return True, f"✓ {package_name or module_name} imported successfully"
    except ImportError as e:
        return False, f"✗ {package_name or module_name} failed to import: {e}"

def test_all_imports() -> List[Tuple[bool, str]]:
    """Test all required imports from datasetprocessing.py"""
    
    imports_to_test = [
        # Core scientific computing
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        
        # Single-cell analysis
        ("scanpy", "scanpy"),
        ("anndata", "anndata"),
        
        # Deep learning
        ("torch", "PyTorch"),
        ("torch.nn", "PyTorch Neural Networks"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("torch.utils.data", "PyTorch Data Utils"),
        
        # Machine learning
        ("sklearn", "scikit-learn"),
        ("sklearn.model_selection", "sklearn.model_selection"),
        ("sklearn.preprocessing", "sklearn.preprocessing"),
        ("sklearn.metrics", "sklearn.metrics"),
        
        # Visualization
        ("matplotlib", "matplotlib"),
        ("matplotlib.pyplot", "matplotlib.pyplot"),
        ("seaborn", "seaborn"),
    ]
    
    results = []
    for module, display_name in imports_to_test:
        success, message = test_import(module, display_name)
        results.append((success, message))
        print(message)
    
    return results

def test_versions():
    """Print versions of key packages."""
    print("\n" + "="*50)
    print("PACKAGE VERSIONS")
    print("="*50)
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")
    
    try:
        import pandas as pd
        print(f"Pandas: {pd.__version__}")
    except ImportError:
        print("Pandas: Not installed")
    
    try:
        import scanpy as sc
        print(f"Scanpy: {sc.__version__}")
    except ImportError:
        print("Scanpy: Not installed")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import pytorch_lightning as pl
        print(f"PyTorch Lightning: {pl.__version__}")
    except ImportError:
        print("PyTorch Lightning: Not installed")
    
    try:
        import sklearn
        print(f"Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("Scikit-learn: Not installed")

def test_custom_classes():
    """Test if custom classes from datasetprocessing.py can be imported."""
    print("\n" + "="*50)
    print("TESTING CUSTOM CLASSES")
    print("="*50)
    
    try:
        from dataset.datasetprocessing import SingleCellDataset, SingleCellPreprocessor
        print("✓ Custom classes imported successfully")
        
        # Test creating instances with dummy data
        import numpy as np
        dummy_expression = np.random.randn(10, 5)
        dummy_labels = np.random.randint(0, 3, 10)
        
        # Test SingleCellDataset
        dataset = SingleCellDataset(dummy_expression, dummy_labels)
        print(f"✓ SingleCellDataset created successfully (length: {len(dataset)})")
        
        # Test SingleCellPreprocessor
        preprocessor = SingleCellPreprocessor()
        print("✓ SingleCellPreprocessor created successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import custom classes: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing custom classes: {e}")
        return False

def main():
    """Main test function."""
    print("="*60)
    print("SINGLE-CELL RNA-SEQ ANALYSIS - DEPENDENCY TEST")
    print("="*60)
    print(f"Python version: {sys.version}")
    print("="*60)
    
    # Test all imports
    results = test_all_imports()
    
    # Count successes and failures
    successes = sum(1 for success, _ in results if success)
    failures = len(results) - successes
    
    # Test versions
    test_versions()
    
    # Test custom classes
    custom_classes_ok = test_custom_classes()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Successful imports: {successes}")
    print(f"✗ Failed imports: {failures}")
    print(f"Custom classes: {'✓ OK' if custom_classes_ok else '✗ FAILED'}")
    
    if failures == 0 and custom_classes_ok:
        print("\n🎉 All dependencies are installed correctly!")
        print("Your single-cell RNA-seq analysis environment is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {failures} dependencies failed to import.")
        print("Please check the installation guide and install missing packages.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)