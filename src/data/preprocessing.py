"""
Data preprocessing utilities for single-cell RNA-seq analysis.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Dict, Tuple, Union
import logging
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Configure scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')


class SingleCellPreprocessor:
    """
    Comprehensive preprocessing pipeline for single-cell RNA-seq data.
    
    This class provides a complete preprocessing workflow including quality control,
    normalization, feature selection, and data scaling following best practices
    for single-cell RNA-seq analysis.
    
    Args:
        min_genes: Minimum number of genes expressed per cell
        min_cells: Minimum number of cells expressing each gene
        max_genes: Maximum number of genes per cell (filter outliers)
        max_mt_percent: Maximum mitochondrial gene percentage
        target_sum: Target sum for normalization
        n_top_genes: Number of highly variable genes to select
        
    Example:
        >>> preprocessor = SingleCellPreprocessor()
        >>> adata_processed = preprocessor.preprocess_adata(adata)
        >>> expression_matrix = adata_processed.X.toarray()
    """
    
    def __init__(
        self,
        min_genes: int = 200,
        min_cells: int = 3,
        max_genes: int = 5000,
        max_mt_percent: float = 20.0,
        target_sum: float = 1e4,
        n_top_genes: Optional[int] = 2000
    ):
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.max_genes = max_genes
        self.max_mt_percent = max_mt_percent
        self.target_sum = target_sum
        self.n_top_genes = n_top_genes
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.highly_variable_genes = None
        self.qc_metrics = {}
        
        logger.info(f"Initialized SingleCellPreprocessor with parameters:")
        logger.info(f"  min_genes={min_genes}, min_cells={min_cells}")
        logger.info(f"  max_genes={max_genes}, max_mt_percent={max_mt_percent}")
        logger.info(f"  target_sum={target_sum}, n_top_genes={n_top_genes}")
    
    def preprocess_adata(
        self, 
        adata, 
        cell_type_col: str = 'cell_type',
        copy: bool = True
    ):
        """
        Complete preprocessing pipeline for AnnData object.
        
        Args:
            adata: AnnData object containing raw count data
            cell_type_col: Column name for cell type annotations
            copy: Whether to work on a copy of the data
            
        Returns:
            Preprocessed AnnData object
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Work on copy to avoid modifying original
        if copy:
            adata = adata.copy()
        
        # Store original data
        adata.raw = adata
        
        # Step 1: Basic filtering
        adata = self._basic_filtering(adata)
        
        # Step 2: Quality control
        adata = self._quality_control(adata)
        
        # Step 3: Normalization
        adata = self._normalization(adata)
        
        # Step 4: Feature selection
        adata = self._feature_selection(adata)
        
        # Step 5: Data scaling
        adata = self._data_scaling(adata)
        
        # Step 6: Encode cell type labels
        if cell_type_col in adata.obs.columns:
            adata = self._encode_cell_types(adata, cell_type_col)
        
        logger.info("Preprocessing pipeline completed successfully!")
        return adata
    
    def _basic_filtering(self, adata):
        """Apply basic cell and gene filtering."""
        logger.info(f"Initial shape: {adata.shape}")
        
        # Filter cells with too few genes
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        logger.info(f"After filtering cells (min_genes={self.min_genes}): {adata.shape}")
        
        # Filter genes expressed in too few cells
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        logger.info(f"After filtering genes (min_cells={self.min_cells}): {adata.shape}")
        
        return adata
    
    def _quality_control(self, adata):
        """Calculate and apply quality control metrics."""
        logger.info("Calculating quality control metrics...")
        
        # Identify mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=['mt'],
            percent_top=None,
            log1p=False,
            inplace=True
        )
        
        # Store QC metrics for later analysis
        self.qc_metrics = {
            'n_genes_by_counts': adata.obs['n_genes_by_counts'].describe(),
            'total_counts': adata.obs['total_counts'].describe()
        }
        
        # Add mitochondrial percentage if mitochondrial genes exist
        if 'pct_counts_mt' in adata.obs.columns:
            self.qc_metrics['pct_counts_mt'] = adata.obs['pct_counts_mt'].describe()
        else:
            logger.warning("No mitochondrial genes found in the dataset")
        
        # Filter cells based on QC metrics
        n_cells_before = adata.n_obs
        
        # Remove cells with too many genes (likely doublets)
        adata = adata[adata.obs.n_genes_by_counts < self.max_genes, :]
        
        # Remove cells with high mitochondrial content (if available)
        if 'pct_counts_mt' in adata.obs.columns:
            adata = adata[adata.obs.pct_counts_mt < self.max_mt_percent, :]
        
        n_cells_after = adata.n_obs
        logger.info(f"After QC filtering: {adata.shape}")
        logger.info(f"Removed {n_cells_before - n_cells_after} cells ({100*(n_cells_before - n_cells_after)/n_cells_before:.1f}%)")
        
        return adata
    
    def _normalization(self, adata):
        """Normalize and log-transform the data."""
        logger.info("Normalizing and log-transforming data...")
        
        # Normalize to target sum
        sc.pp.normalize_total(adata, target_sum=self.target_sum)
        
        # Log transform
        sc.pp.log1p(adata)
        
        return adata
    
    def _feature_selection(self, adata):
        """Select highly variable genes."""
        logger.info("Selecting highly variable genes...")
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(
            adata, 
            min_mean=0.0125, 
            max_mean=3, 
            min_disp=0.5,
            n_top_genes=self.n_top_genes
        )
        
        # Store highly variable genes info
        self.highly_variable_genes = adata.var['highly_variable'].copy()
        
        # Keep only highly variable genes
        adata = adata[:, adata.var.highly_variable]
        logger.info(f"After HVG selection: {adata.shape}")
        
        return adata
    
    def _data_scaling(self, adata):
        """Scale data to unit variance."""
        logger.info("Scaling data...")
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
        
        return adata
    
    def _encode_cell_types(self, adata, cell_type_col: str):
        """Encode cell type labels."""
        logger.info(f"Encoding cell type labels from column '{cell_type_col}'...")
        
        # Encode cell type labels
        adata.obs['cell_type_encoded'] = self.label_encoder.fit_transform(
            adata.obs[cell_type_col]
        )
        
        # Log cell type information
        cell_type_counts = adata.obs[cell_type_col].value_counts()
        logger.info(f"Found {len(cell_type_counts)} cell types:")
        for cell_type, count in cell_type_counts.items():
            logger.info(f"  {cell_type}: {count} cells")
        
        return adata
    
    def get_label_mapping(self) -> Dict[int, str]:
        """
        Get mapping from encoded labels to original cell types.
        
        Returns:
            Dictionary mapping encoded integers to cell type names
        """
        if not hasattr(self.label_encoder, 'classes_'):
            raise ValueError("Label encoder has not been fitted yet. Run preprocess_adata first.")
        
        return dict(enumerate(self.label_encoder.classes_))
    
    def get_qc_summary(self) -> Dict:
        """
        Get summary of quality control metrics.
        
        Returns:
            Dictionary containing QC metric summaries
        """
        return self.qc_metrics.copy()
    
    def save_preprocessing_info(self, filepath: str):
        """
        Save preprocessing parameters and results to file.
        
        Args:
            filepath: Path to save the preprocessing information
        """
        import json
        
        info = {
            'parameters': {
                'min_genes': self.min_genes,
                'min_cells': self.min_cells,
                'max_genes': self.max_genes,
                'max_mt_percent': self.max_mt_percent,
                'target_sum': self.target_sum,
                'n_top_genes': self.n_top_genes
            },
            'qc_metrics': self.qc_metrics,
            'label_mapping': self.get_label_mapping() if hasattr(self.label_encoder, 'classes_') else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        logger.info(f"Preprocessing information saved to {filepath}")


def load_data(filepath: str, **kwargs):
    """
    Load single-cell data from various formats.
    
    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments passed to scanpy read functions
        
    Returns:
        AnnData object containing the loaded data
    """
    if filepath.endswith('.h5ad'):
        return sc.read_h5ad(filepath, **kwargs)
    elif filepath.endswith('.h5'):
        return sc.read_10x_h5(filepath, **kwargs)
    elif filepath.endswith('.mtx') or filepath.endswith('.mtx.gz'):
        return sc.read_10x_mtx(filepath, **kwargs)
    elif filepath.endswith('.csv'):
        return sc.read_csv(filepath, **kwargs)
    elif filepath.endswith('.xlsx'):
        return sc.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")