#!/usr/bin/env python3
"""
Complete PBMC Analysis Pipeline

This example demonstrates the full single-cell RNA-seq analysis pipeline
using real PBMC data with the restructured package architecture.

Features:
- PBMC data loading and preprocessing
- Quality control and filtering
- Dimensionality reduction and clustering
- Deep learning with scVAE
- Pharma-relevant analysis
- Comprehensive visualization
"""

import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import (
    SingleCellDataset, 
    SingleCellPreprocessor, 
    PBMCDataLoader,
    print_pharma_relevance
)
from src.models import scVAE, SingleCellAnalyzer
from src.visualization import Visualizer

def main():
    """Complete PBMC analysis pipeline."""
    print("="*70)
    print("PBMC SINGLE-CELL RNA-SEQ ANALYSIS - COMPLETE PIPELINE")
    print("="*70)
    
    # Configure scanpy
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    # Step 1: Load PBMC Data
    print("\n🔬 STEP 1: Loading PBMC Data")
    print("-" * 40)
    
    pbmc_loader = PBMCDataLoader(data_dir="./data")
    
    # Load processed data (faster) or raw data (more control)
    use_raw_data = True  # Set to False to use pre-processed data
    
    if use_raw_data:
        print("Loading raw PBMC data for full preprocessing control...")
        adata = pbmc_loader.load_raw_data()
    else:
        print("Loading pre-processed PBMC data...")
        adata = pbmc_loader.load_processed_data()
    
    print(f"Initial data shape: {adata.shape}")
    
    # Step 2: Data Preprocessing
    print("\n🧹 STEP 2: Data Preprocessing")
    print("-" * 40)
    
    # Initialize preprocessor with PBMC-optimized parameters
    preprocessor = SingleCellPreprocessor(
        min_genes=200,      # Filter cells with < 200 genes
        min_cells=3,        # Filter genes in < 3 cells
        max_genes=5000,     # Filter cells with > 5000 genes (doublets)
        max_mt_percent=20,  # Filter cells with > 20% mitochondrial genes
        target_sum=1e4,     # Normalize to 10,000 counts per cell
        n_top_genes=2000    # Select top 2000 highly variable genes
    )
    
    # Initialize analyzer
    analyzer = SingleCellAnalyzer(preprocessor)
    
    # Preprocess data
    adata_processed = analyzer.load_and_preprocess_data(
        adata, 
        cell_type_col='cell_type' if 'cell_type' in adata.obs.columns else None
    )
    
    # Add cell type annotations if not present
    if 'cell_type' not in adata_processed.obs.columns:
        print("Adding cell type annotations...")
        adata_processed = pbmc_loader.add_manual_annotations(adata_processed)
    
    # Add pharma-relevant context
    print("Adding pharma-relevant annotations...")
    adata_processed = pbmc_loader.add_pharma_context(adata_processed)
    
    print(f"Processed data shape: {adata_processed.shape}")
    print(f"Cell types identified: {adata_processed.obs['cell_type'].nunique()}")
    
    # Step 3: Prepare Datasets for Deep Learning
    print("\n🤖 STEP 3: Preparing Deep Learning Datasets")
    print("-" * 40)
    
    train_dataset, val_dataset = analyzer.prepare_datasets(
        test_size=0.2,
        random_state=42,
        stratify=True
    )
    
    print(f"Training set: {len(train_dataset)} cells")
    print(f"Validation set: {len(val_dataset)} cells")
    print(f"Number of genes: {train_dataset.n_genes}")
    print(f"Number of cell types: {train_dataset.n_cell_types}")
    
    # Step 4: Train scVAE Model
    print("\n🧠 STEP 4: Training scVAE Model")
    print("-" * 40)
    
    # Train the model
    model = analyzer.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_latent=10,        # 10-dimensional latent space
        n_hidden=128,       # 128 hidden units
        n_layers=2,         # 2 hidden layers
        learning_rate=1e-3, # Learning rate
        batch_size=256,     # Batch size
        max_epochs=50,      # Maximum epochs
        patience=10         # Early stopping patience
    )
    
    print("Model training completed!")
    
    # Step 5: Model Evaluation
    print("\n📊 STEP 5: Model Evaluation")
    print("-" * 40)
    
    results = analyzer.evaluate_model(val_dataset, batch_size=256)
    
    print(f"Validation accuracy: {results.get('accuracy', 'N/A'):.4f}")
    print(f"Average loss: {results['avg_loss']:.4f}")
    
    # Step 6: Visualization and Analysis
    print("\n📈 STEP 6: Visualization and Analysis")
    print("-" * 40)
    
    visualizer = Visualizer(figsize=(12, 8))
    
    # Get label mapping
    label_mapping = preprocessor.get_label_mapping()
    
    # Visualize latent space
    print("Generating latent space visualization...")
    visualizer.plot_latent_space(
        latent_data=results['latent_representations'],
        labels=results['labels'],
        label_mapping=label_mapping,
        method='umap',
        title='scVAE Latent Space - PBMC Cell Types',
        save='pbmc_latent_space.png'
    )
    
    # Plot cell type distribution
    print("Plotting cell type distribution...")
    visualizer.plot_cell_type_distribution(
        labels=results['labels'],
        label_mapping=label_mapping,
        title='PBMC Cell Type Distribution',
        save='pbmc_cell_distribution.png'
    )
    
    # Plot reconstruction quality
    if len(results['reconstructions']) > 0:
        print("Analyzing reconstruction quality...")
        # Get original data for comparison
        val_loader = analyzer.trainer.val_dataloaders
        original_data = []
        for batch in val_loader:
            x, _ = batch
            original_data.extend(x.numpy())
        original_data = np.array(original_data)
        
        visualizer.plot_reconstruction_quality(
            original=original_data,
            reconstructed=results['reconstructions'],
            n_samples=500,
            save='pbmc_reconstruction_quality.png'
        )
    
    # Step 7: Pharma-Relevant Analysis
    print("\n💊 STEP 7: Pharma-Relevant Analysis")
    print("-" * 40)
    
    # Analyze drug target expression
    print("Analyzing drug target expression...")
    
    # Get cells with high target scores
    target_columns = [col for col in adata_processed.obs.columns if '_target_score' in col]
    if target_columns:
        print("\nDrug target expression analysis:")
        for col in target_columns:
            mean_score = adata_processed.obs[col].mean()
            std_score = adata_processed.obs[col].std()
            print(f"  {col}: {mean_score:.3f} ± {std_score:.3f}")
    
    # Analyze clinical pathway scores
    pathway_columns = [col for col in adata_processed.obs.columns if col.endswith('_score') and 'target' not in col]
    if pathway_columns:
        print("\nClinical pathway analysis:")
        for col in pathway_columns:
            mean_score = adata_processed.obs[col].mean()
            std_score = adata_processed.obs[col].std()
            print(f"  {col}: {mean_score:.3f} ± {std_score:.3f}")
    
    # Step 8: Save Results
    print("\n💾 STEP 8: Saving Results")
    print("-" * 40)
    
    # Save processed data
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    
    adata_processed.write_h5ad(output_dir / "pbmc_processed_with_analysis.h5ad")
    print(f"Saved processed data to: {output_dir / 'pbmc_processed_with_analysis.h5ad'}")
    
    # Save model
    analyzer.save_model(output_dir / "pbmc_scvae_model.pth")
    print(f"Saved model to: {output_dir / 'pbmc_scvae_model.pth'}")
    
    # Save analysis results
    results_df = pd.DataFrame({
        'cell_id': range(len(results['labels'])),
        'true_label': results['labels'],
        'predicted_label': results.get('predictions', results['labels']),
        'cell_type': [label_mapping.get(label, f'Type_{label}') for label in results['labels']]
    })
    
    # Add latent representations
    latent_df = pd.DataFrame(
        results['latent_representations'], 
        columns=[f'latent_{i}' for i in range(results['latent_representations'].shape[1])]
    )
    results_df = pd.concat([results_df, latent_df], axis=1)
    
    results_df.to_csv(output_dir / "pbmc_analysis_results.csv", index=False)
    print(f"Saved analysis results to: {output_dir / 'pbmc_analysis_results.csv'}")
    
    # Step 9: Summary and Pharma Relevance
    print("\n🎯 STEP 9: Analysis Summary")
    print("-" * 40)
    
    print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📊 Dataset: {adata_processed.shape[0]} cells, {adata_processed.shape[1]} genes")
    print(f"🧬 Cell types: {adata_processed.obs['cell_type'].nunique()}")
    print(f"🎯 Model accuracy: {results.get('accuracy', 'N/A'):.4f}")
    print(f"📁 Results saved to: {output_dir}")
    
    # Print pharma relevance
    print_pharma_relevance()
    
    # Model summary
    model_summary = analyzer.get_model_summary()
    print(f"\n🤖 MODEL SUMMARY:")
    print(f"  Architecture: {model_summary['architecture']}")
    print(f"  Total parameters: {model_summary['total_parameters']:,}")
    print(f"  Trainable parameters: {model_summary['trainable_parameters']:,}")
    
    print(f"\n🚀 Ready for downstream analysis and drug discovery applications!")
    
    return adata_processed, analyzer, results

if __name__ == "__main__":
    # Import required modules
    from pathlib import Path
    
    try:
        # Run the complete analysis
        adata, analyzer, results = main()
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Use the returned objects for further analysis:")
        print(f"  - adata: Processed AnnData object")
        print(f"  - analyzer: Trained SingleCellAnalyzer")
        print(f"  - results: Analysis results dictionary")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)