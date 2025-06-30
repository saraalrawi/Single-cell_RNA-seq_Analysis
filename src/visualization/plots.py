"""
Visualization utilities for single-cell RNA-seq analysis results.

This module provides comprehensive plotting functions for visualizing
single-cell data, model results, and analysis outcomes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
from typing import Optional, Dict, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization toolkit for single-cell RNA-seq analysis.
    
    This class provides methods for visualizing various aspects of single-cell
    data analysis including quality control metrics, dimensionality reduction,
    model results, and comparative analyses.
    
    Example:
        >>> visualizer = Visualizer()
        >>> visualizer.plot_qc_metrics(adata)
        >>> visualizer.plot_latent_space(latent_data, labels)
        >>> visualizer.plot_training_history(history)
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        
        # Configure scanpy plotting
        sc.settings.set_figure_params(dpi=dpi, facecolor='white')
        
        logger.info("Initialized Visualizer")
    
    def plot_qc_metrics(
        self,
        adata,
        metrics: List[str] = None,
        save: Optional[str] = None
    ):
        """
        Plot quality control metrics for single-cell data.
        
        Args:
            adata: AnnData object with QC metrics
            metrics: List of metrics to plot
            save: Path to save the plot
        """
        if metrics is None:
            metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
        
        # Check which metrics are available
        available_metrics = [m for m in metrics if m in adata.obs.columns]
        
        if not available_metrics:
            logger.warning("No QC metrics found in adata.obs")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            # Violin plot
            sc.pl.violin(
                adata, metric,
                jitter=0.4,
                multi_panel=True,
                ax=axes[i]
            )
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_latent_space(
        self,
        latent_data: np.ndarray,
        labels: np.ndarray,
        label_mapping: Optional[Dict[int, str]] = None,
        method: str = 'umap',
        title: str = 'Latent Space Visualization',
        save: Optional[str] = None
    ):
        """
        Visualize latent space using dimensionality reduction.
        
        Args:
            latent_data: Latent representations
            labels: Cell type labels
            label_mapping: Mapping from label indices to names
            method: Dimensionality reduction method ('umap' or 'tsne')
            title: Plot title
            save: Path to save the plot
        """
        # Create AnnData object for scanpy visualization
        latent_adata = ad.AnnData(X=latent_data)
        
        # Add cell type labels
        if label_mapping:
            cell_types = [label_mapping.get(label, f'Type_{label}') for label in labels]
        else:
            cell_types = [f'Type_{label}' for label in labels]
        
        latent_adata.obs['cell_type'] = cell_types
        
        # Compute neighbors
        sc.pp.neighbors(latent_adata, n_neighbors=15, n_pcs=latent_data.shape[1])
        
        # Compute dimensionality reduction
        if method.lower() == 'umap':
            sc.tl.umap(latent_adata)
            sc.pl.umap(
                latent_adata,
                color='cell_type',
                title=title,
                frameon=False,
                save=f'_{save}' if save else None
            )
        elif method.lower() == 'tsne':
            sc.tl.tsne(latent_adata)
            sc.pl.tsne(
                latent_adata,
                color='cell_type',
                title=title,
                frameon=False,
                save=f'_{save}' if save else None
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save: Optional[str] = None
    ):
        """
        Plot training history from model training.
        
        Args:
            history: Dictionary containing training metrics
            save: Path to save the plot
        """
        # Determine number of subplots needed
        metrics = list(history.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            logger.warning("No training history provided")
            return
        
        # Create subplots
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].plot(history[metric], label=metric)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_gene_expression(
        self,
        adata,
        genes: List[str],
        groupby: str = 'cell_type',
        plot_type: str = 'violin',
        save: Optional[str] = None
    ):
        """
        Plot gene expression across cell types.
        
        Args:
            adata: AnnData object
            genes: List of genes to plot
            groupby: Column to group by
            plot_type: Type of plot ('violin', 'box', 'dot')
            save: Path to save the plot
        """
        # Check if genes exist
        available_genes = [g for g in genes if g in adata.var_names]
        
        if not available_genes:
            logger.warning(f"None of the specified genes found in data: {genes}")
            return
        
        if len(available_genes) < len(genes):
            missing = set(genes) - set(available_genes)
            logger.warning(f"Some genes not found: {missing}")
        
        # Plot based on type
        if plot_type == 'violin':
            sc.pl.violin(
                adata,
                available_genes,
                groupby=groupby,
                multi_panel=True,
                save=f'_{save}' if save else None
            )
        elif plot_type == 'box':
            # Create box plots manually
            n_genes = len(available_genes)
            fig, axes = plt.subplots(1, n_genes, figsize=(5*n_genes, 6))
            
            if n_genes == 1:
                axes = [axes]
            
            for i, gene in enumerate(available_genes):
                gene_data = []
                group_labels = []
                
                for group in adata.obs[groupby].unique():
                    mask = adata.obs[groupby] == group
                    gene_expr = adata[mask, gene].X.toarray().flatten()
                    gene_data.append(gene_expr)
                    group_labels.append(group)
                
                axes[i].boxplot(gene_data, labels=group_labels)
                axes[i].set_title(gene)
                axes[i].set_ylabel('Expression')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
            
            plt.show()
            
        elif plot_type == 'dot':
            sc.pl.dotplot(
                adata,
                available_genes,
                groupby=groupby,
                save=f'_{save}' if save else None
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def plot_reconstruction_quality(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        n_samples: int = 1000,
        save: Optional[str] = None
    ):
        """
        Plot reconstruction quality comparison.
        
        Args:
            original: Original data
            reconstructed: Reconstructed data
            n_samples: Number of samples to plot
            save: Path to save the plot
        """
        # Sample random subset for visualization
        n_samples = min(n_samples, original.shape[0])
        indices = np.random.choice(original.shape[0], n_samples, replace=False)
        
        orig_sample = original[indices]
        recon_sample = reconstructed[indices]
        
        # Calculate correlation
        correlations = [
            np.corrcoef(orig_sample[i], recon_sample[i])[0, 1]
            for i in range(n_samples)
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot of original vs reconstructed
        axes[0, 0].scatter(orig_sample.flatten(), recon_sample.flatten(), alpha=0.5)
        axes[0, 0].plot([orig_sample.min(), orig_sample.max()], 
                       [orig_sample.min(), orig_sample.max()], 'r--')
        axes[0, 0].set_xlabel('Original Expression')
        axes[0, 0].set_ylabel('Reconstructed Expression')
        axes[0, 0].set_title('Original vs Reconstructed')
        
        # Correlation distribution
        axes[0, 1].hist(correlations, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Correlation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Reconstruction Correlation\nMean: {np.mean(correlations):.3f}')
        
        # Example reconstructions
        for i, idx in enumerate([0, 1]):
            if idx < len(orig_sample):
                x = np.arange(len(orig_sample[idx]))
                axes[1, i].plot(x, orig_sample[idx], label='Original', alpha=0.7)
                axes[1, i].plot(x, recon_sample[idx], label='Reconstructed', alpha=0.7)
                axes[1, i].set_xlabel('Gene Index')
                axes[1, i].set_ylabel('Expression')
                axes[1, i].set_title(f'Sample {idx + 1} Reconstruction')
                axes[1, i].legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_cell_type_distribution(
        self,
        labels: np.ndarray,
        label_mapping: Optional[Dict[int, str]] = None,
        title: str = 'Cell Type Distribution',
        save: Optional[str] = None
    ):
        """
        Plot distribution of cell types.
        
        Args:
            labels: Cell type labels
            label_mapping: Mapping from indices to names
            title: Plot title
            save: Path to save the plot
        """
        # Count cell types
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Map labels to names
        if label_mapping:
            label_names = [label_mapping.get(label, f'Type_{label}') for label in unique_labels]
        else:
            label_names = [f'Type_{label}' for label in unique_labels]
        
        # Create bar plot
        plt.figure(figsize=self.figsize)
        bars = plt.bar(label_names, counts)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.xlabel('Cell Type')
        plt.ylabel('Number of Cells')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_latent_interpolation(
        self,
        interpolated_data: np.ndarray,
        title: str = 'Latent Space Interpolation',
        save: Optional[str] = None
    ):
        """
        Plot latent space interpolation results.
        
        Args:
            interpolated_data: Interpolated samples
            title: Plot title
            save: Path to save the plot
        """
        n_steps = interpolated_data.shape[0]
        
        plt.figure(figsize=(15, 4))
        
        # Show first few genes across interpolation
        n_genes_to_show = min(20, interpolated_data.shape[1])
        
        for i in range(n_genes_to_show):
            plt.plot(range(n_steps), interpolated_data[:, i], alpha=0.7, linewidth=1)
        
        plt.xlabel('Interpolation Step')
        plt.ylabel('Gene Expression')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_comparison(
        self,
        data_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
        label_mapping: Optional[Dict[int, str]] = None,
        method: str = 'umap',
        title: str = 'Method Comparison',
        save: Optional[str] = None
    ):
        """
        Compare multiple dimensionality reduction methods.
        
        Args:
            data_dict: Dictionary of method names to data arrays
            labels: Cell type labels
            label_mapping: Mapping from indices to names
            method: Dimensionality reduction method for visualization
            title: Plot title
            save: Path to save the plot
        """
        n_methods = len(data_dict)
        
        if n_methods == 0:
            logger.warning("No data provided for comparison")
            return
        
        # Create subplots
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        
        if n_methods == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (method_name, data) in enumerate(data_dict.items()):
            if i < len(axes):
                # Create temporary AnnData for visualization
                temp_adata = ad.AnnData(X=data)
                
                # Add labels
                if label_mapping:
                    cell_types = [label_mapping.get(label, f'Type_{label}') for label in labels]
                else:
                    cell_types = [f'Type_{label}' for label in labels]
                
                temp_adata.obs['cell_type'] = cell_types
                
                # Compute neighbors and UMAP/t-SNE
                sc.pp.neighbors(temp_adata, n_neighbors=15, n_pcs=min(50, data.shape[1]))
                
                if method.lower() == 'umap':
                    sc.tl.umap(temp_adata)
                    embedding = temp_adata.obsm['X_umap']
                else:
                    sc.tl.tsne(temp_adata)
                    embedding = temp_adata.obsm['X_tsne']
                
                # Plot
                unique_types = temp_adata.obs['cell_type'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
                
                for j, cell_type in enumerate(unique_types):
                    mask = temp_adata.obs['cell_type'] == cell_type
                    axes[i].scatter(
                        embedding[mask, 0], 
                        embedding[mask, 1],
                        c=[colors[j]], 
                        label=cell_type, 
                        alpha=0.7,
                        s=20
                    )
                
                axes[i].set_title(f'{method_name}')
                axes[i].set_xlabel(f'{method.upper()} 1')
                axes[i].set_ylabel(f'{method.upper()} 2')
                
                # Add legend only to first subplot
                if i == 0:
                    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()