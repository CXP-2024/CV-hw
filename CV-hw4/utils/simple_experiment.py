#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.trainer import Trainer
from utils.visualization import evaluate_model, visualize_predictions, visualize_class_performance, create_confusion_matrix_visualization

class SimpleExperiment:
    """Class for running experiments using standard train/val split instead of cross-validation."""
    
    def __init__(self, model_class, dataset_module, experiment_name, device,
                 base_save_dir='outputs', config=None):
        """
        Initialize the simple experiment.
        
        Args:
            model_class: Class of the model to train
            dataset_module: Dataset module
            experiment_name: Name of the experiment (used for saving)
            device: Device to use (cuda/cpu)
            base_save_dir: Base directory for saving results
            config: Configuration for the experiment
        """
        self.model_class = model_class
        self.dataset_module = dataset_module
        self.experiment_name = experiment_name
        self.device = device
        self.base_save_dir = base_save_dir
        self.config = config or {}
        
        # Create experiment directory
        self.experiment_dir = os.path.join(base_save_dir, experiment_name)
        self.models_dir = os.path.join(self.experiment_dir, 'models')
        self.logs_dir = os.path.join(self.experiment_dir, 'logs')
        self.figures_dir = os.path.join(self.experiment_dir, 'figures')
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Save experiment configuration
        self.save_config()
    
    def setup_logging(self):
        """Setup logging for the experiment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.logs_dir, 'experiment.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_config(self):
        """Save experiment configuration to a file."""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.logger.info(f"Saved experiment configuration to {config_path}")
    
    def setup_dataloaders(self):
        """Setup train and validation dataloaders."""
        self.dataset_module.setup()
        
        train_loader = DataLoader(
            self.dataset_module.train_dataset,
            batch_size=self.dataset_module.batch_size,
            shuffle=True,
            num_workers=self.dataset_module.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.dataset_module.val_dataset,
            batch_size=self.dataset_module.batch_size,
            shuffle=False,
            num_workers=self.dataset_module.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def run_simple_experiment(self, num_classes=19, num_epochs=50, criterion=None, optimizer_params=None, early_stopping_patience=10, checkpoint_path=None, pretrained_path=None):
        """
        Run experiment with standard train/val split.
        
        Args:
            num_classes: Number of classes for the model
            num_epochs: Number of epochs to train
            criterion: Loss function (default: CrossEntropyLoss with class weights)
            optimizer_params: Parameters for the optimizer
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            checkpoint_path: Path to checkpoint to resume training from (optional)
            pretrained_path: Path to pretrained model weights to use as initialization (optional)
            
        Returns:
            results: Dictionary with training and validation results
        """
        # Get dataloaders
        train_loader, val_loader = self.setup_dataloaders()
        
        self.logger.info(f"Starting training with {num_epochs} epochs")
        self.logger.info(f"Training dataset size: {len(self.dataset_module.train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(self.dataset_module.val_dataset)}")
        
        # Set default optimizer parameters if not provided
        optimizer_params = optimizer_params or {
            'lr': 0.001, 
            'weight_decay': 1e-4
        }
        
        # Initialize model
        model = self.model_class(num_classes=num_classes).to(self.device)
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.logger.info(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=self.device)
            # Handle both cases: full checkpoint and just model state dict
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            self.logger.info("Pretrained weights loaded successfully")
        
        # Initialize criterion if not provided
        if criterion is None:
            # Handle ignore_index if specified in config
            ignore_index = self.config.get('data', {}).get('ignore_index', 255)
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # Initialize optimizer with explicit type conversion
        lr = float(optimizer_params.get('lr', 0.001))
        weight_decay = float(optimizer_params.get('weight_decay', 1e-4))
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
          # Initialize scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler,
            save_dir=self.models_dir,    
            log_dir=self.logs_dir
        )
        
        # Handle resuming from checkpoint
        start_epoch = 0
        if checkpoint_path is not None:
            self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            self.logger.info(f"Resuming from epoch {start_epoch}")
        
        # Train model
        train_metrics, val_metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=num_epochs,
            start_epoch=start_epoch,
            early_stopping_patience=early_stopping_patience
        )
        
        # Save training history
        history = {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss']
        }
        
        history_df = pd.DataFrame(history)
        history_path = os.path.join(self.logs_dir, 'history.csv')
        history_df.to_csv(history_path, index=False)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Evaluate model on validation set
        best_model_path = os.path.join(self.models_dir, 'best_model.pth')
        model.load_state_dict(torch.load(best_model_path))
        
        self.logger.info("开始使用优化的方法评估模型...")
        val_results = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=self.device,
            num_classes=num_classes,
            ignore_index=self.config.get('data', {}).get('ignore_index', 255)
        )
        self.logger.info(f"评估完成！mIoU: {val_results['miou']:.4f}, 加权mIoU: {val_results['weighted_miou']:.4f}")
        
        # Generate visualizations
        visualize_predictions(
            model=model,
            dataloader=val_loader,
            device=self.device,
            num_samples=min(3, len(val_loader)),
            save_dir=self.figures_dir
        )
        
        visualize_class_performance(
            confusion_matrix=val_results['confusion_matrix'],
            class_names=[f'Class {i}' for i in range(num_classes)],
            save_path=os.path.join(self.figures_dir, 'class_performance.png')
        )
        
        create_confusion_matrix_visualization(
            confusion_matrix=val_results['confusion_matrix'],
            class_names=[f'Class {i}' for i in range(num_classes)],
            save_path=os.path.join(self.figures_dir, 'confusion_matrix.png')
        )
        
        # Save results
        results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'val_results': val_results,
            'best_epoch': trainer.best_epoch
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy_to_python(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        # Convert numpy arrays to Python native types
        json_serializable_results = convert_numpy_to_python(results)
        
        results_path = os.path.join(self.logs_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(json_serializable_results, f, indent=4)
        
        self.logger.info(f"Training completed. Best validation mIoU: {trainer.best_metric:.4f} at epoch {trainer.best_epoch}")
        
        return results
    
    def plot_training_history(self, history):
        """Plot training history."""
        plt.figure(figsize=(10, 6))
        
        # Plot loss only
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'training_history.png'))
        plt.close()
