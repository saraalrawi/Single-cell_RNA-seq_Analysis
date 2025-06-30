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
import anndata as ad
import warnings
warnings.filterwarnings('ignore')

# Configure scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')
class scVAE(pl.LightningModule):
    """Variational Autoencoder for single-cell RNA-seq with cell type classification"""
    
    def __init__(self, n_input: int, n_hidden: int = 128, n_latent: int = 10, 
                 n_layers: int = 1, n_classes: int = None, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        
        # Encoder layers
        encoder_layers = []
        layer_input = n_input
        for i in range(n_layers):
            encoder_layers.extend([
                nn.Linear(layer_input, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            layer_input = n_hidden
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_var = nn.Linear(n_hidden, n_latent)
        
        # Decoder layers
        decoder_layers = []
        layer_input = n_latent
        for i in range(n_layers):
            decoder_layers.extend([
                nn.Linear(layer_input, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            layer_input = n_hidden
        
        decoder_layers.append(nn.Linear(n_hidden, n_input))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Classification head (if provided)
        if n_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(n_latent, n_hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(n_hidden // 2, n_classes)
            )
        
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        
        outputs = {'x_recon': x_recon, 'mu': mu, 'log_var': log_var, 'z': z}
        
        if self.n_classes is not None:
            cell_type_logits = self.classifier(z)
            outputs['cell_type_logits'] = cell_type_logits
            
        return outputs
    
    def loss_function(self, x, outputs, cell_types=None, beta=1.0):
        """VAE loss with optional classification loss"""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(outputs['x_recon'], x, reduction='mean')
        
        # KLD loss
        kld_loss = -0.5 * torch.sum(1 + outputs['log_var'] - outputs['mu'].pow(2) - outputs['log_var'].exp())
        kld_loss = kld_loss / x.size(0)  # Normalize by batch size
        
        # Total VAE loss
        vae_loss = recon_loss + beta * kld_loss
        
        loss_dict = {
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'vae_loss': vae_loss
        }
        
        # Classification loss (if applicable)
        if self.n_classes is not None and cell_types is not None:
            class_loss = nn.functional.cross_entropy(outputs['cell_type_logits'], cell_types)
            total_loss = vae_loss + class_loss
            loss_dict.update({
                'class_loss': class_loss,
                'total_loss': total_loss
            })
        else:
            loss_dict['total_loss'] = vae_loss
            
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        x, cell_types = batch
        outputs = self(x)
        
        # Compute losses
        loss_dict = self.loss_function(x, outputs, cell_types)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, prog_bar=True)
            
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        x, cell_types = batch
        outputs = self(x)
        
        # Compute losses
        loss_dict = self.loss_function(x, outputs, cell_types)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, prog_bar=True)
        
        # Calculate accuracy if classification is enabled
        if self.n_classes is not None:
            preds = torch.argmax(outputs['cell_type_logits'], dim=1)
            acc = (preds == cell_types).float().mean()
            self.log('val_acc', acc, prog_bar=True)
            
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_total_loss'
        }
    def visualize_latent_space(self, results, method='umap'):
        """Visualize latent space using UMAP or t-SNE"""
        latent_data = results['latent_representations']
        labels = results['labels']
        
        # Add latent representations to adata for visualization
        if hasattr(self, 'adata') and self.adata is not None:
            latent_adata = ad.AnnData(X=latent_data)
            
            # Add cell type labels
            label_mapping = self.preprocessor.get_label_mapping()
            cell_types = [label_mapping[label] for label in labels]
            latent_adata.obs['cell_type'] = cell_types
            
            # Compute UMAP
            sc.pp.neighbors(latent_adata, n_neighbors=15, n_pcs=latent_data.shape[1])
            sc.tl.umap(latent_adata)
            
            # Plot
            sc.pl.umap(latent_adata, color='cell_type', legend_loc='on data', 
                      title='Latent Space Visualization (VAE)', frameon=False, save='.pdf')