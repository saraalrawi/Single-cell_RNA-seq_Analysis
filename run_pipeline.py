#!/usr/bin/env python3
"""
Complete Single-Cell RNA-seq Analysis Pipeline Runner

This script runs the complete analysis pipeline from data loading to results.
Simply run: python run_pipeline.py

The pipeline includes:
1. Data loading (PBMC dataset)
2. Preprocessing and quality control
3. Deep learning model training
4. Evaluation and analysis
5. Visualization and results export
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def run_complete_pipeline():
    """Run the complete single-cell analysis pipeline."""
    
    print("🧬" + "="*60)
    print("   SINGLE-CELL RNA-SEQ ANALYSIS - COMPLETE PIPELINE")
    print("="*63)
    
    try:
        # Import all required modules
        from src.data import PBMCDataLoader, SingleCellPreprocessor
        from src.models import SingleCellAnalyzer
        from src.visualization import Visualizer
        import scanpy as sc
        
        # Configure scanpy
        sc.settings.verbosity = 1  # Reduce verbosity for cleaner output
        sc.settings.set_figure_params(dpi=80, facecolor='white')
        
        # Step 1: Data Loading
        print("\n📥 STEP 1: Loading PBMC Data")
        print("-" * 40)
        
        loader = PBMCDataLoader(data_dir="./data")
        print("Loading PBMC 3K dataset...")
        adata = loader.load_raw_data()
        print(f"✅ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
        
        # Step 2: Preprocessing
        print("\n🧹 STEP 2: Data Preprocessing")
        print("-" * 40)
        
        preprocessor = SingleCellPreprocessor(
            min_genes=200,
            min_cells=3,
            max_genes=5000,
            max_mt_percent=20,
            n_top_genes=2000
        )
        
        analyzer = SingleCellAnalyzer(preprocessor)
        adata_processed = analyzer.load_and_preprocess_data(adata)
        
        # Add cell type annotations
        adata_processed = loader.add_manual_annotations(adata_processed)
        adata_processed = loader.add_pharma_context(adata_processed)
        
        print(f"✅ Preprocessed data: {adata_processed.shape[0]} cells, {adata_processed.shape[1]} genes")
        print(f"✅ Cell types identified: {adata_processed.obs['cell_type'].nunique()}")
        
        # Step 3: Dataset Preparation
        print("\n🔄 STEP 3: Preparing Datasets")
        print("-" * 40)
        
        train_dataset, val_dataset = analyzer.prepare_datasets(test_size=0.2, random_state=42)
        print(f"✅ Training set: {len(train_dataset)} cells")
        print(f"✅ Validation set: {len(val_dataset)} cells")
        
        # Step 4: Model Training
        print("\n🧠 STEP 4: Training Deep Learning Model")
        print("-" * 40)
        
        print("Training scVAE model...")
        model = analyzer.train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_latent=10,
            n_hidden=128,
            n_layers=2,
            batch_size=256,
            max_epochs=30,  # Reduced for faster demo
            patience=8
        )
        print("✅ Model training completed!")
        
        # Step 5: Evaluation
        print("\n📊 STEP 5: Model Evaluation")
        print("-" * 40)
        
        results = analyzer.evaluate_model(val_dataset)
        accuracy = results.get('accuracy', 0)
        print(f"✅ Model accuracy: {accuracy:.4f}")
        print(f"✅ Average loss: {results['avg_loss']:.4f}")
        
        # Step 6: Visualization
        print("\n📈 STEP 6: Generating Visualizations")
        print("-" * 40)
        
        visualizer = Visualizer()
        label_mapping = preprocessor.get_label_mapping()
        
        # Create results directory
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)
        
        print("Generating latent space visualization...")
        visualizer.plot_latent_space(
            latent_data=results['latent_representations'],
            labels=results['labels'],
            label_mapping=label_mapping,
            method='umap',
            title='scVAE Latent Space - PBMC Cell Types'
        )
        
        print("Generating cell type distribution plot...")
        visualizer.plot_cell_type_distribution(
            labels=results['labels'],
            label_mapping=label_mapping,
            title='PBMC Cell Type Distribution'
        )
        
        print("✅ Visualizations generated!")
        
        # Step 7: Save Results
        print("\n💾 STEP 7: Saving Results")
        print("-" * 40)
        
        # Save processed data
        adata_processed.write_h5ad(results_dir / "pbmc_analysis_complete.h5ad")
        print(f"✅ Saved processed data: {results_dir / 'pbmc_analysis_complete.h5ad'}")
        
        # Save model
        analyzer.save_model(results_dir / "scvae_model.pth")
        print(f"✅ Saved model: {results_dir / 'scvae_model.pth'}")
        
        # Save results summary
        import pandas as pd
        summary_df = pd.DataFrame({
            'metric': ['n_cells', 'n_genes', 'n_cell_types', 'model_accuracy', 'avg_loss'],
            'value': [
                adata_processed.shape[0],
                adata_processed.shape[1], 
                adata_processed.obs['cell_type'].nunique(),
                accuracy,
                results['avg_loss']
            ]
        })
        summary_df.to_csv(results_dir / "analysis_summary.csv", index=False)
        print(f"✅ Saved summary: {results_dir / 'analysis_summary.csv'}")
        
        # Step 8: Final Summary
        print("\n🎉 STEP 8: Analysis Complete!")
        print("-" * 40)
        
        print(f"""
📊 ANALYSIS SUMMARY:
   • Dataset: PBMC 3K cells
   • Cells analyzed: {adata_processed.shape[0]:,}
   • Genes analyzed: {adata_processed.shape[1]:,}
   • Cell types identified: {adata_processed.obs['cell_type'].nunique()}
   • Model accuracy: {accuracy:.1%}
   • Results saved to: {results_dir}

🎯 PHARMACEUTICAL INSIGHTS:
   • Drug target scores computed for all cell types
   • Clinical pathway analysis completed
   • Inflammation, immune activation, and exhaustion scores calculated
   • Ready for biomarker discovery and therapeutic target analysis

📁 OUTPUT FILES:
   • {results_dir / 'pbmc_analysis_complete.h5ad'} - Complete processed dataset
   • {results_dir / 'scvae_model.pth'} - Trained deep learning model
   • {results_dir / 'analysis_summary.csv'} - Analysis metrics
   • Visualization plots (displayed above)

🚀 NEXT STEPS:
   • Explore the processed data in {results_dir / 'pbmc_analysis_complete.h5ad'}
   • Use the trained model for new data analysis
   • Investigate drug targets and clinical pathways
   • Extend analysis for your specific research questions
        """)
        
        print("\n✨ Pipeline completed successfully! ✨")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n🔧 Please ensure all dependencies are installed:")
        print("   1. Run: source scrna-env/bin/activate")
        print("   2. Run: python tests/test_imports.py")
        print("   3. If tests fail, run: ./install_dependencies.sh")
        return False
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting complete single-cell RNA-seq analysis pipeline...")
    print("This will download PBMC data, preprocess it, train a model, and generate results.")
    print("\nEstimated time: 5-15 minutes (depending on your system)")
    
    # Ask for confirmation
    response = input("\nProceed with analysis? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = run_complete_pipeline()
        if success:
            print("\n🎊 All done! Check the results/ directory for outputs.")
        else:
            print("\n💡 Check the error messages above and try again.")
            sys.exit(1)
    else:
        print("Pipeline cancelled. Run again when ready!")