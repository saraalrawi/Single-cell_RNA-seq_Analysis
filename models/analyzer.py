class SingleCellAnalyzer:
    """Main class for single-cell analysis workflow"""
    
    def __init__(self, preprocessor: SingleCellPreprocessor):
        self.preprocessor = preprocessor
        self.model = None
        self.adata = None
        
    def load_and_preprocess_data(self, adata, cell_type_col: str = 'cell_type'):
        """Load and preprocess single-cell data"""
        print("Loading and preprocessing data...")
        self.adata = self.preprocessor.preprocess_adata(adata, cell_type_col)
        return self.adata
    
    def prepare_datasets(self, test_size: float = 0.2, random_state: int = 42):
        """Prepare train/validation datasets"""
        if self.adata is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data first.")
        
        # Get expression data and labels
        X = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        y = self.adata.obs['cell_type_encoded'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create datasets
        train_dataset = SingleCellDataset(X_train, y_train)
        val_dataset = SingleCellDataset(X_val, y_val)
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset, n_latent: int = 10, 
                   n_hidden: int = 128, n_layers: int = 2, max_epochs: int = 100,
                   batch_size: int = 512):
        """Train the VAE model"""
        # Get dimensions
        n_input = train_dataset.expression_data.shape[1]
        n_classes = len(np.unique(train_dataset.cell_types))
        
        print(f"Training model with {n_input} genes, {n_classes} cell types")
        
        # Create model
        self.model = scVAE(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_classes=n_classes
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=10
        )
        
        # Train model
        trainer.fit(self.model, train_loader, val_loader)
        
        return self.model
    
    def evaluate_model(self, val_dataset):
        """Evaluate trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_latent = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                outputs = self.model(x)
                
                # Predictions
                if 'cell_type_logits' in outputs:
                    preds = torch.argmax(outputs['cell_type_logits'], dim=1)
                    all_preds.extend(preds.cpu().numpy())
                
                all_labels.extend(y.cpu().numpy())
                all_latent.extend(outputs['z'].cpu().numpy())
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_latent = np.array(all_latent)
        
        results = {
            'labels': all_labels,
            'latent_representations': all_latent
        }
        
        if all_preds:
            all_preds = np.array(all_preds)
            results['predictions'] = all_preds
            
            # Classification report
            label_mapping = self.preprocessor.get_label_mapping()
            target_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
            
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=target_names))
            
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, 
                       yticklabels=target_names, cmap='Blues')
            plt.title('Cell Type Prediction Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
        
        return results