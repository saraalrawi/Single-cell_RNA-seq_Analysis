#!/usr/bin/env python3
"""
Test script to verify all dependencies and new package structure are working correctly
for the single-cell RNA-seq analysis project.
"""

import sys
import os
from typing import List, Tuple

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        __import__(module_name)
        return True, f"✓ {package_name or module_name} imported successfully"
    except ImportError as e:
        return False, f"✗ {package_name or module_name} failed to import: {e}"

def test_core_dependencies() -> List[Tuple[bool, str]]:
    """Test all core dependencies."""
    
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

def test_package_structure() -> List[Tuple[bool, str]]:
    """Test the new package structure."""
    
    package_imports = [
        # Main package
        ("src", "Main package"),
        
        # Data module
        ("src.data", "Data module"),
        ("src.data.dataset", "Dataset module"),
        ("src.data.preprocessing", "Preprocessing module"),
        
        # Models module
        ("src.models", "Models module"),
        ("src.models.scvae", "scVAE module"),
        ("src.models.analyzer", "Analyzer module"),
        
        # Visualization module
        ("src.visualization", "Visualization module"),
        ("src.visualization.plots", "Plots module"),
    ]
    
    results = []
    print("\n" + "="*50)
    print("TESTING PACKAGE STRUCTURE")
    print("="*50)
    
    for module, display_name in package_imports:
        success, message = test_import(module, display_name)
        results.append((success, message))
        print(message)
    
    return results

def test_class_imports() -> List[Tuple[bool, str]]:
    """Test importing specific classes from the new structure."""
    
    class_imports = [
        # Data classes
        ("src.data", "Data classes import"),
        
        # Model classes
        ("src.models", "Model classes import"),
        
        # Visualization classes
        ("src.visualization", "Visualization classes import"),
    ]
    
    results = []
    print("\n" + "="*50)
    print("TESTING CLASS IMPORTS")
    print("="*50)
    
    for module, display_name in class_imports:
        success, message = test_import(module, display_name)
        results.append((success, message))
        print(message)
    
    return results

def test_class_instantiation():
    """Test creating instances of key classes."""
    print("\n" + "="*50)
    print("TESTING CLASS INSTANTIATION")
    print("="*50)
    
    try:
        from src.data import SingleCellDataset, SingleCellPreprocessor
        from src.models import scVAE, SingleCellAnalyzer
        from src.visualization import Visualizer
        
        # Test data classes
        import numpy as np
        dummy_expression = np.random.randn(10, 5)
        dummy_labels = np.random.randint(0, 3, 10)
        
        dataset = SingleCellDataset(dummy_expression, dummy_labels)
        print(f"✓ SingleCellDataset created successfully (length: {len(dataset)})")
        
        preprocessor = SingleCellPreprocessor()
        print("✓ SingleCellPreprocessor created successfully")
        
        # Test model classes
        model = scVAE(n_input=100, n_hidden=64, n_latent=10, n_classes=3)
        print("✓ scVAE model created successfully")
        
        analyzer = SingleCellAnalyzer(preprocessor)
        print("✓ SingleCellAnalyzer created successfully")
        
        # Test visualization
        visualizer = Visualizer()
        print("✓ Visualizer created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating class instances: {e}")
        return False

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

def main():
    """Main test function."""
    print("="*60)
    print("SINGLE-CELL RNA-SEQ ANALYSIS - COMPREHENSIVE TEST")
    print("="*60)
    print(f"Python version: {sys.version}")
    print("="*60)
    
    # Test core dependencies
    core_results = test_core_dependencies()
    
    # Test package structure
    structure_results = test_package_structure()
    
    # Test class imports
    class_results = test_class_imports()
    
    # Test class instantiation
    instantiation_ok = test_class_instantiation()
    
    # Test versions
    test_versions()
    
    # Count successes and failures
    all_results = core_results + structure_results + class_results
    successes = sum(1 for success, _ in all_results if success)
    failures = len(all_results) - successes
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Successful imports: {successes}")
    print(f"✗ Failed imports: {failures}")
    print(f"Class instantiation: {'✓ OK' if instantiation_ok else '✗ FAILED'}")
    
    if failures == 0 and instantiation_ok:
        print("\n🎉 All tests passed! Your restructured single-cell RNA-seq analysis environment is ready!")
        print("✅ Package structure follows Python best practices")
        print("✅ All dependencies are properly installed")
        print("✅ All classes can be imported and instantiated")
        return 0
    else:
        print(f"\n⚠️  {failures} imports failed or class instantiation failed.")
        print("Please check the installation and package structure.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)