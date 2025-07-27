"""
Single-cell analysis workflow orchestrator.

This module provides a high-level interface for conducting complete single-cell
RNA-seq analysis workflows, including data preprocessing, model training, and
evaluation.
"""

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
import logging

from ..data import SingleCellDataset, SingleCellPreprocessor
from .scvae import scVAE

logger = logging.getLogger(__name__)


class SingleCellAnalyzer:
    """
    Main orchestrator class for single-cell analysis workflows.
    
    This class provides a high-level interface for conducting complete single-cell
    RNA-seq analysis, including data preprocessing, model training, evaluation,
    and visualization.
    
    Args:
        preprocessor: SingleCellPreprocessor instance for data preprocessing
        
    Example:
        >>> preprocessor = SingleCellPreprocessor()
        >>> analyzer = SingleCellAnalyzer(preprocessor)
        >>> analyzer.load_and_preprocess_data(adata)
        >>> train_ds, val_ds = analyzer.prepare_datasets()
        >>> model = analyzer.train_model(train_ds, val_ds)
        >>> results = analyzer.evaluate_model(val_ds)
    """
    
    def __init__(self, preprocessor: Optional[SingleCellPreprocessor] = None):
        self.preprocessor = preprocessor or SingleCellPreprocessor()
        self.model = None
        self.adata = None
        self.trainer = None
        self.training_history = {}
        
        logger.info("Initialized SingleCellAnalyzer")
    
    def load_and_preprocess_data(self, adata, cell_type_col: str = 'cell_type', copy: bool = True):
        """
        Load and preprocess single-cell data.
        
        Args:
            adata: AnnData object containing raw single-cell data
            cell_type_col: Column name for cell type annotations
            copy: Whether to work on a copy of the data
            
        Returns:
            Preprocessed AnnData object
        """
        logger.info("Loading and preprocessing data...")
        self.adata = self.preprocessor.preprocess_adata(adata, cell_type_col, copy=copy)
        
        logger.info(f"Preprocessed data shape: {self.adata.shape}")
        if cell_type_col and cell_type_col in self.adata.obs.columns:
            logger.info(f"Number of cell types: {len(self.adata.obs[cell_type_col].unique())}")
        else:
            logger.info("No cell type annotations available")
        
        return self.adata
    
    def prepare_datasets(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[SingleCellDataset, SingleCellDataset]:
        """
        Prepare training and validation datasets.
        
        Args:
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split by cell types
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.adata is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data first.")
        
        logger.info("Preparing datasets...")
        
        # Get expression data and labels
        X = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        y = self.adata.obs['cell_type_encoded'].values
        
        # Split data
        stratify_labels = y if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_labels
        )
        
        # Create datasets
        train_dataset = SingleCellDataset(X_train, y_train)
        val_dataset = SingleCellDataset(X_val, y_val)
        
        logger.info(f"Training set: {len(train_dataset)} cells")
        logger.info(f"Validation set: {len(val_dataset)} cells")
        
        # Log cell type distribution
        train_counts = train_dataset.get_cell_type_counts()
        val_counts = val_dataset.get_cell_type_counts()
        logger.info(f"Training cell type distribution: {train_counts}")
        logger.info(f"Validation cell type distribution: {val_counts}")
        
        return train_dataset, val_dataset
    
    def train_model(
        self,
        train_dataset: SingleCellDataset,
        val_dataset: SingleCellDataset,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 15,
        **model_kwargs
    ) -> scVAE:
        """
        Train the scVAE model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            n_latent: Latent space dimensionality
            n_hidden: Hidden layer size
            n_layers: Number of hidden layers
            learning_rate: Learning rate
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience
            **model_kwargs: Additional model parameters
            
        Returns:
            Trained scVAE model
        """
        logger.info("Starting model training...")
        
        # Get model dimensions
        n_input = train_dataset.n_genes
        n_classes = train_dataset.n_cell_types
        
        logger.info(f"Model architecture: {n_input} → {n_hidden} → {n_latent}")
        logger.info(f"Classification: {n_classes} cell types")
        
        # Create model
        self.model = scVAE(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_classes=n_classes,
            learning_rate=learning_rate,
            **model_kwargs
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Setup callbacks
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor='val_total_loss',
                patience=patience,
                mode='min',
                verbose=True
            ),
            pl.callbacks.ModelCheckpoint(
                monitor='val_total_loss',
                mode='min',
                save_top_k=1,
                verbose=True
            )
        ]
        
        # Setup trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
            check_val_every_n_epoch=1
        )
        
        # Train model
        logger.info(f"Training for up to {max_epochs} epochs...")
        self.trainer.fit(self.model, train_loader, val_loader)
        
        # Load best model
        if self.trainer.checkpoint_callback.best_model_path:
            self.model = scVAE.load_from_checkpoint(
                self.trainer.checkpoint_callback.best_model_path
            )
            logger.info("Loaded best model from checkpoint")
        
        logger.info("Model training completed!")
        return self.model
    
    def evaluate_model(
        self,
        val_dataset: SingleCellDataset,
        batch_size: int = 512
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            val_dataset: Validation dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        logger.info("Evaluating model...")
        
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_latent = []
        all_reconstructions = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                # Move data to the same device as the model
                device = next(self.model.parameters()).device
                x = x.to(device)
                y = y.to(device)
                outputs = self.model(x)
                
                # Compute loss
                loss_dict = self.model.loss_function(x, outputs, y)
                total_loss += loss_dict['total_loss'].item()
                
                # Store predictions and representations
                if 'cell_type_logits' in outputs:
                    preds = torch.argmax(outputs['cell_type_logits'], dim=1)
                    all_preds.extend(preds.cpu().numpy())
                
                all_labels.extend(y.cpu().numpy())
                all_latent.extend(outputs['z'].cpu().numpy())
                all_reconstructions.extend(outputs['x_recon'].cpu().numpy())
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_latent = np.array(all_latent)
        all_reconstructions = np.array(all_reconstructions)
        avg_loss = total_loss / len(val_loader)
        
        # Prepare results
        results = {
            'labels': all_labels,
            'latent_representations': all_latent,
            'reconstructions': all_reconstructions,
            'avg_loss': avg_loss
        }
        
        # Classification metrics
        if all_preds:
            all_preds = np.array(all_preds)
            results['predictions'] = all_preds
            
            # Calculate accuracy
            accuracy = accuracy_score(all_labels, all_preds)
            results['accuracy'] = accuracy
            
            # Get label mapping
            label_mapping = self.preprocessor.get_label_mapping()
            target_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
            
            # Classification report
            class_report = classification_report(
                all_labels, all_preds, 
                target_names=target_names,
                output_dict=True
            )
            results['classification_report'] = class_report
            
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            results['confusion_matrix'] = cm
            
            logger.info(f"Classification accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=target_names))
            
            # Plot confusion matrix
            self._plot_confusion_matrix(cm, target_names)
        
        logger.info(f"Average validation loss: {avg_loss:.4f}")
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, target_names: list):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            xticklabels=target_names,
            yticklabels=target_names, 
            cmap='Blues'
        )
        plt.title('Cell Type Prediction Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def get_latent_representations(
        self,
        dataset: SingleCellDataset,
        batch_size: int = 512
    ) -> np.ndarray:
        """
        Get latent representations for a dataset.
        
        Args:
            dataset: Dataset to encode
            batch_size: Batch size for processing
            
        Returns:
            Latent representations as numpy array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        latent_reps = []
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                x = x.to(device)
                mu, _ = self.model.encode(x)
                latent_reps.extend(mu.cpu().numpy())
        
        return np.array(latent_reps)
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_params: Dict[str, Any]):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            model_params: Model parameters for initialization
        """
        self.model = scVAE(**model_params)
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of the trained model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "No model trained"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": "scVAE",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": {
                "n_input": self.model.n_input,
                "n_hidden": self.model.n_hidden,
                "n_latent": self.model.n_latent,
                "n_layers": self.model.n_layers,
                "n_classes": self.model.n_classes
            },
            "hyperparameters": {
                "learning_rate": self.model.learning_rate,
                "dropout_rate": self.model.dropout_rate,
                "beta": self.model.beta
            }
        }