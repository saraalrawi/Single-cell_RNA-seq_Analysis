"""
Single-cell Variational Autoencoder (scVAE) implementation.

This module contains a PyTorch Lightning implementation of a Variational Autoencoder
specifically designed for single-cell RNA-seq data analysis with optional cell type
classification capabilities.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class scVAE(pl.LightningModule):
    """
    Variational Autoencoder for single-cell RNA-seq data with optional cell type classification.
    
    This model combines dimensionality reduction through variational autoencoders with
    optional supervised cell type classification. It's designed to learn meaningful
    latent representations of single-cell gene expression data.
    
    Args:
        n_input: Number of input features (genes)
        n_hidden: Number of hidden units in encoder/decoder layers
        n_latent: Dimensionality of the latent space
        n_layers: Number of hidden layers in encoder/decoder
        n_classes: Number of cell types for classification (optional)
        learning_rate: Learning rate for optimization
        dropout_rate: Dropout rate for regularization
        beta: Weight for KL divergence term (β-VAE)
        
    Example:
        >>> model = scVAE(n_input=2000, n_hidden=128, n_latent=10, n_classes=5)
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(model, train_dataloader, val_dataloader)
    """
    
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_classes: Optional[int] = None,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        beta: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.beta = beta
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Latent space parameters
        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_var = nn.Linear(n_hidden, n_latent)
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Optional classification head
        if n_classes is not None:
            self.classifier = self._build_classifier()
        
        logger.info(f"Initialized scVAE with {n_input} inputs, {n_latent} latent dims")
        if n_classes:
            logger.info(f"Classification enabled for {n_classes} cell types")
    
    def _build_encoder(self) -> nn.Module:
        """Build the encoder network."""
        layers = []
        layer_input = self.n_input
        
        for i in range(self.n_layers):
            layers.extend([
                nn.Linear(layer_input, self.n_hidden),
                nn.BatchNorm1d(self.n_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            layer_input = self.n_hidden
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build the decoder network."""
        layers = []
        layer_input = self.n_latent
        
        for i in range(self.n_layers):
            layers.extend([
                nn.Linear(layer_input, self.n_hidden),
                nn.BatchNorm1d(self.n_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            layer_input = self.n_hidden
        
        # Output layer (no activation for reconstruction)
        layers.append(nn.Linear(self.n_hidden, self.n_input))
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self) -> nn.Module:
        """Build the classification head."""
        return nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate * 2),  # Higher dropout for classifier
            nn.Linear(self.n_hidden // 2, self.n_classes)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters.
        
        Args:
            x: Input tensor of shape (batch_size, n_input)
            
        Returns:
            Tuple of (mu, log_var) tensors
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decode(z)
        
        # Prepare outputs
        outputs = {
            'x_recon': x_recon,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
        
        # Optional classification
        if self.n_classes is not None:
            cell_type_logits = self.classifier(z)
            outputs['cell_type_logits'] = cell_type_logits
        
        return outputs
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        cell_types: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss with optional classification loss.
        
        Args:
            x: Original input
            outputs: Model outputs from forward pass
            cell_types: Cell type labels (optional)
            
        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(
            outputs['x_recon'], x, reduction='mean'
        )
        
        # KL divergence loss
        kld_loss = -0.5 * torch.sum(
            1 + outputs['log_var'] - outputs['mu'].pow(2) - outputs['log_var'].exp()
        )
        kld_loss = kld_loss / x.size(0)  # Normalize by batch size
        
        # Total VAE loss
        vae_loss = recon_loss + self.beta * kld_loss
        
        loss_dict = {
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'vae_loss': vae_loss
        }
        
        # Optional classification loss
        if self.n_classes is not None and cell_types is not None:
            class_loss = nn.functional.cross_entropy(
                outputs['cell_type_logits'], cell_types
            )
            total_loss = vae_loss + class_loss
            loss_dict.update({
                'class_loss': class_loss,
                'total_loss': total_loss
            })
        else:
            loss_dict['total_loss'] = vae_loss
        
        return loss_dict
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning."""
        x, cell_types = batch
        outputs = self(x)
        
        # Compute losses
        loss_dict = self.loss_function(x, outputs, cell_types)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for PyTorch Lightning."""
        x, cell_types = batch
        outputs = self(x)
        
        # Compute losses
        loss_dict = self.loss_function(x, outputs, cell_types)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy if classification is enabled
        if self.n_classes is not None and 'cell_type_logits' in outputs:
            preds = torch.argmax(outputs['cell_type_logits'], dim=1)
            acc = (preds == cell_types).float().mean()
            self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss',
                'frequency': 1
            }
        }
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation for input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation (mean of posterior)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
    def generate_samples(self, n_samples: int = 100) -> torch.Tensor:
        """
        Generate samples from the learned latent distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        self.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(n_samples, self.n_latent)
            if next(self.parameters()).is_cuda:
                z = z.cuda()
            
            # Decode samples
            samples = self.decode(z)
        
        return samples
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two samples in latent space.
        
        Args:
            x1: First sample
            x2: Second sample
            n_steps: Number of interpolation steps
            
        Returns:
            Interpolated samples
        """
        self.eval()
        with torch.no_grad():
            # Encode both samples
            mu1, _ = self.encode(x1.unsqueeze(0))
            mu2, _ = self.encode(x2.unsqueeze(0))
            
            # Create interpolation weights
            alphas = torch.linspace(0, 1, n_steps).unsqueeze(1)
            if next(self.parameters()).is_cuda:
                alphas = alphas.cuda()
            
            # Interpolate in latent space
            z_interp = alphas * mu2 + (1 - alphas) * mu1
            
            # Decode interpolated latent vectors
            x_interp = self.decode(z_interp)
        
        return x_interp