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
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Configure scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

class SingleCellDataset(Dataset):
    """PyTorch dataset for single-cell RNA-seq data"""
    
    def __init__(self, expression_data: np.ndarray, cell_types: np.ndarray):
        self.expression_data = torch.FloatTensor(expression_data)
        self.cell_types = torch.LongTensor(cell_types)
        
    def __len__(self):
        return len(self.expression_data)
    
    def __getitem__(self, idx):
        return self.expression_data[idx], self.cell_types[idx]

class SingleCellPreprocessor:
    """Preprocessing pipeline for single-cell RNA-seq data"""
    
    def __init__(self, min_genes: int = 200, min_cells: int = 3, 
                 max_genes: int = 5000, target_sum: int = 1e4):
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.max_genes = max_genes
        self.target_sum = target_sum
        self.label_encoder = LabelEncoder()
        self.highly_variable_genes = None
        
    def preprocess_adata(self, adata, cell_type_col: str = 'cell_type'):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Make a copy to avoid modifying original
        adata = adata.copy()
        
        # Basic filtering
        print(f"Initial shape: {adata.shape}")
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        print(f"After basic filtering: {adata.shape}")
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        # Filter cells based on QC metrics
        adata = adata[adata.obs.n_genes_by_counts < self.max_genes, :]
        adata = adata[adata.obs.pct_counts_mt < 20, :]  # Remove high mitochondrial content
        print(f"After QC filtering: {adata.shape}")
        
        # Normalize and log transform
        sc.pp.normalize_total(adata, target_sum=self.target_sum)
        sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        self.highly_variable_genes = adata.var['highly_variable']
        
        # Keep only highly variable genes
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        print(f"After HVG selection: {adata.shape}")
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
        
        # Encode cell type labels
        if cell_type_col in adata.obs.columns:
            adata.obs['cell_type_encoded'] = self.label_encoder.fit_transform(adata.obs[cell_type_col])
        
        return adata
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get mapping from encoded labels to original cell types"""
        return dict(enumerate(self.label_encoder.classes_))