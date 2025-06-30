"""
PyTorch dataset classes for single-cell RNA-seq data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SingleCellDataset(Dataset):
    """
    PyTorch dataset for single-cell RNA-seq data.
    
    This dataset class handles single-cell expression data and corresponding
    cell type labels for use with PyTorch DataLoaders and training pipelines.
    
    Args:
        expression_data: Gene expression matrix of shape (n_cells, n_genes)
        cell_types: Cell type labels of shape (n_cells,)
        transform: Optional transform to be applied to the data
        
    Example:
        >>> expression = np.random.randn(1000, 2000)  # 1000 cells, 2000 genes
        >>> labels = np.random.randint(0, 5, 1000)    # 5 cell types
        >>> dataset = SingleCellDataset(expression, labels)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self, 
        expression_data: Union[np.ndarray, torch.Tensor], 
        cell_types: Union[np.ndarray, torch.Tensor],
        transform: Optional[callable] = None
    ):
        # Convert to tensors if numpy arrays
        if isinstance(expression_data, np.ndarray):
            self.expression_data = torch.FloatTensor(expression_data)
        else:
            self.expression_data = expression_data.float()
            
        if isinstance(cell_types, np.ndarray):
            self.cell_types = torch.LongTensor(cell_types)
        else:
            self.cell_types = cell_types.long()
            
        self.transform = transform
        
        # Validate dimensions
        if len(self.expression_data) != len(self.cell_types):
            raise ValueError(
                f"Expression data and cell types must have same length. "
                f"Got {len(self.expression_data)} and {len(self.cell_types)}"
            )
            
        logger.info(
            f"Created SingleCellDataset with {len(self)} cells and "
            f"{self.expression_data.shape[1]} genes"
        )
        
    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return len(self.expression_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (expression_vector, cell_type_label)
        """
        expression = self.expression_data[idx]
        cell_type = self.cell_types[idx]
        
        if self.transform:
            expression = self.transform(expression)
            
        return expression, cell_type
    
    @property
    def n_genes(self) -> int:
        """Number of genes in the dataset."""
        return self.expression_data.shape[1]
    
    @property
    def n_cell_types(self) -> int:
        """Number of unique cell types in the dataset."""
        return len(torch.unique(self.cell_types))
    
    def get_cell_type_counts(self) -> dict:
        """
        Get counts of each cell type in the dataset.
        
        Returns:
            Dictionary mapping cell type indices to counts
        """
        unique, counts = torch.unique(self.cell_types, return_counts=True)
        return {int(cell_type): int(count) for cell_type, count in zip(unique, counts)}
    
    def split(self, train_ratio: float = 0.8, random_state: int = 42) -> Tuple['SingleCellDataset', 'SingleCellDataset']:
        """
        Split the dataset into training and validation sets.
        
        Args:
            train_ratio: Proportion of data to use for training
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(len(self))
        train_idx, val_idx = train_test_split(
            indices, 
            train_size=train_ratio, 
            random_state=random_state,
            stratify=self.cell_types.numpy()
        )
        
        train_dataset = SingleCellDataset(
            self.expression_data[train_idx],
            self.cell_types[train_idx],
            transform=self.transform
        )
        
        val_dataset = SingleCellDataset(
            self.expression_data[val_idx],
            self.cell_types[val_idx],
            transform=self.transform
        )
        
        return train_dataset, val_dataset