#!/usr/bin/env python3
"""
Visualize PBMC clusters and cell types in low-dimensional space.

This script creates comprehensive visualizations of PBMC data including:
- UMAP/t-SNE plots with cell type annotations
- Cluster analysis
- Gene expression patterns
"""

import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import PBMCDataLoader

def main():
    """Create comprehensive PBMC visualizations."""
    print("="*70)
    print("PBMC CLUSTER VISUALIZATION")
    print("="*70)
    
    # Configure scanpy
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=150, facecolor='white', figsize=(8, 6))
    
    # Create output directory
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load PBMC data
    print("\n📊 Loading PBMC data...")
    pbmc_loader = PBMCDataLoader(data_dir="./data")
    
    # Check if processed data exists
    processed_file = Path("./results/pbmc_processed_with_analysis.h5ad")
    if processed_file.exists():
        print(f"Loading processed data from {processed_file}")
        adata = sc.read_h5ad(processed_file)
    else:
        print("Loading and processing PBMC data...")
        adata = pbmc_loader.load_raw_data()
        
        # Basic preprocessing
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        # Filter cells
        adata = adata[adata.obs.n_genes_by_counts < 2500, :]
        adata = adata[adata.obs.pct_counts_mt < 5, :] if 'pct_counts_mt' in adata.obs else adata
        
        # Normalize and log transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
        
        # PCA
        sc.tl.pca(adata, svd_solver='arpack')
        
        # Compute neighbors
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        
        # UMAP
        sc.tl.umap(adata)
        
        # Leiden clustering
        sc.tl.leiden(adata, resolution=0.5)
        
        # Add cell type annotations
        adata = pbmc_loader.add_manual_annotations(adata)
    
    print(f"\nData shape: {adata.shape}")
    print(f"Cell types: {adata.obs['cell_type'].value_counts()}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # 1. UMAP plot colored by cell types
    print("Plotting UMAP with cell types...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(
        adata, 
        color='cell_type', 
        legend_loc='on data',
        title='PBMC Cell Types (UMAP)',
        frameon=False,
        ax=ax,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'pbmc_umap_cell_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. UMAP plot colored by leiden clusters
    print("Plotting UMAP with Leiden clusters...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(
        adata, 
        color='leiden', 
        legend_loc='on data',
        title='PBMC Leiden Clusters (UMAP)',
        frameon=False,
        ax=ax,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'pbmc_umap_leiden.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. t-SNE visualization
    if 'X_tsne' not in adata.obsm:
        print("Computing t-SNE...")
        sc.tl.tsne(adata)
    
    print("Plotting t-SNE with cell types...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.tsne(
        adata, 
        color='cell_type', 
        legend_loc='on data',
        title='PBMC Cell Types (t-SNE)',
        frameon=False,
        ax=ax,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'pbmc_tsne_cell_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cell type composition plot
    print("Creating cell type composition plot...")
    cell_type_counts = adata.obs['cell_type'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_type_counts)))
    
    # Bar plot
    cell_type_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_xlabel('Cell Type')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Cell Type Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Pie chart
    cell_type_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=colors)
    ax2.set_ylabel('')
    ax2.set_title('Cell Type Proportions')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pbmc_cell_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Marker gene expression
    print("Plotting marker gene expression...")
    marker_genes = {
        'T cells': ['CD3D', 'CD3E', 'CD4', 'CD8A'],
        'B cells': ['CD19', 'CD79A', 'MS4A1'],  # MS4A1 is CD20
        'NK cells': ['GNLY', 'NKG7', 'KLRB1'],
        'Monocytes': ['CD14', 'LYZ', 'CST3'],
        'Dendritic': ['FCER1A', 'CST3', 'IL3RA']
    }
    
    # Flatten marker genes and filter available ones
    all_markers = []
    for markers in marker_genes.values():
        all_markers.extend(markers)
    all_markers = list(set(all_markers))
    available_markers = [g for g in all_markers if g in adata.raw.var_names]
    
    if available_markers:
        print(f"Found {len(available_markers)} marker genes")
        
        # Dot plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sc.pl.dotplot(
            adata, 
            available_markers, 
            groupby='cell_type',
            dendrogram=True,
            ax=ax,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / 'pbmc_marker_genes_dotplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature plots for key markers
        key_markers = ['CD3D', 'CD14', 'CD19', 'GNLY']
        available_key_markers = [m for m in key_markers if m in adata.raw.var_names]
        
        if available_key_markers:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, marker in enumerate(available_key_markers[:4]):
                sc.pl.umap(
                    adata,
                    color=marker,
                    use_raw=True,
                    ax=axes[i],
                    show=False,
                    frameon=False,
                    title=f'{marker} Expression'
                )
            
            plt.tight_layout()
            plt.savefig(output_dir / 'pbmc_key_markers_umap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. QC metrics visualization
    print("Creating QC metrics plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Number of genes per cell
    axes[0, 0].hist(adata.obs['n_genes_by_counts'], bins=50, alpha=0.7)
    axes[0, 0].set_xlabel('Number of Genes')
    axes[0, 0].set_ylabel('Number of Cells')
    axes[0, 0].set_title('Genes per Cell Distribution')
    
    # Total counts per cell
    axes[0, 1].hist(adata.obs['total_counts'], bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Total Counts')
    axes[0, 1].set_ylabel('Number of Cells')
    axes[0, 1].set_title('Total Counts Distribution')
    
    # Mitochondrial percentage (if available)
    if 'pct_counts_mt' in adata.obs:
        axes[1, 0].hist(adata.obs['pct_counts_mt'], bins=50, alpha=0.7)
        axes[1, 0].set_xlabel('Mitochondrial Gene Percentage')
        axes[1, 0].set_ylabel('Number of Cells')
        axes[1, 0].set_title('Mitochondrial Gene % Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'No MT data available', ha='center', va='center')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
    
    # Scatter plot: genes vs counts
    axes[1, 1].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], 
                      alpha=0.3, s=1)
    axes[1, 1].set_xlabel('Total Counts')
    axes[1, 1].set_ylabel('Number of Genes')
    axes[1, 1].set_title('Genes vs Counts Correlation')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pbmc_qc_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualizations saved to {output_dir}/")
    print("\nGenerated plots:")
    for plot_file in sorted(output_dir.glob('pbmc_*.png')):
        print(f"  - {plot_file.name}")
    
    return adata

if __name__ == "__main__":
    adata = main()
    print("\n🎉 Visualization complete!")