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
from utils.trainer import Trainer
from utils.visualization import evaluate_model, visualize_predictions, visualize_class_performance, create_confusion_matrix_visualization

class CrossValidationExperiment:
    """Class for running cross-validation experiments."""
    
    def __init__(self, model_class, dataset_module, experiment_name, device,
                 base_save_dir='outputs', config=None):
        """
        Initialize the cross-validation experiment.
        
        Args:
            model_class: Class of the model to train
            dataset_module: Dataset module with cross-validation support
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
    
    def run_cross_validation(self, num_classes=19, num_epochs=50, criterion=None, optimizer_params=None):
        """
        Run cross-validation experiment.
        
        Args:
            num_classes: Number of classes for the model
            num_epochs: Number of epochs to train each fold
            criterion: Loss function (default: CrossEntropyLoss with class weights)
            optimizer_params: Parameters for the optimizer
            
        Returns:
            cv_results: Dictionary with cross-validation results
        """
        # Get cross-validation folds
        cv_folds = self.dataset_module.get_cross_validation_folds()
        n_folds = len(cv_folds)
        
        self.logger.info(f"Starting {n_folds}-fold cross-validation with {num_epochs} epochs per fold")
        
        # Set default optimizer parameters if not provided
        optimizer_params = optimizer_params or {
            'lr': 0.001, 
            'weight_decay': 1e-4
        }
        
        # Initialize results storage
        cv_results = {
            'fold_metrics': [],
            'best_miou': 0.0,
            'best_fold': -1,
            'all_mious': [],
        }
        
        # Run training for each fold
        for fold_idx, fold_data in enumerate(cv_folds):
            self.logger.info(f"Training fold {fold_idx+1}/{n_folds}")
            
            # Initialize model
            model = self.model_class(num_classes=num_classes)
            model.to(self.device)
            
            # Get data loaders for this fold
            train_loader = fold_data['train_loader']
            val_loader = fold_data['val_loader']
            
            # Create criterion if not provided
            if criterion is None:
                # For semantic segmentation, using weighted cross entropy is common
                # to handle class imbalance
                criterion = nn.CrossEntropyLoss(ignore_index=255)
            
            # Create optimizer
            optimizer = optim.Adam(model.parameters(), **optimizer_params)
            
            # Create scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            
            # Create trainer
            fold_save_dir = os.path.join(self.models_dir, f'fold_{fold_idx}')
            fold_log_dir = os.path.join(self.logs_dir, f'fold_{fold_idx}')
            
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                scheduler=scheduler,
                save_dir=fold_save_dir,
                log_dir=fold_log_dir
            )
            
            # Train the model
            best_miou = trainer.train(train_loader, val_loader, num_epochs)
            
            # Evaluate the model on validation set
            model.load_state_dict(torch.load(
                os.path.join(fold_save_dir, 'model_best.pth'), 
                map_location=self.device)['model_state_dict'])
            
            val_miou, val_iou_per_class, confusion_matrix = evaluate_model(
                model, val_loader, self.device, num_classes)
            
            # Log fold results
            fold_metrics = {
                'fold': fold_idx,
                'best_miou': best_miou,
                'val_miou': val_miou,
                'val_iou_per_class': val_iou_per_class.tolist(),
                'learning_curve': {
                    'train_loss': trainer.train_loss_history,
                    'val_loss': trainer.val_loss_history,
                    'train_miou': trainer.train_miou_history,
                    'val_miou': trainer.val_miou_history,
                }
            }
            cv_results['fold_metrics'].append(fold_metrics)
            cv_results['all_mious'].append(val_miou)
            
            self.logger.info(f"Fold {fold_idx+1}/{n_folds} - Best mIoU: {best_miou:.4f}, Val mIoU: {val_miou:.4f}")
            
            # Create visualizations for this fold
            if hasattr(self.dataset_module.val_dataset, 'get_class_names'):
                class_names = self.dataset_module.val_dataset.get_class_names()
            else:
                class_names = [f"Class {i}" for i in range(num_classes)]
            
            # Visualize class performance
            visualize_class_performance(
                val_iou_per_class, 
                class_names, 
                save_path=os.path.join(self.figures_dir, f'fold_{fold_idx}_class_performance.png')
            )
            
            # Visualize confusion matrix
            create_confusion_matrix_visualization(
                confusion_matrix, 
                class_names,
                save_path=os.path.join(self.figures_dir, f'fold_{fold_idx}_confusion_matrix.png')
            )
            
            # Visualize predictions
            if hasattr(self.dataset_module.val_dataset, 'get_class_colors'):
                class_colors = self.dataset_module.val_dataset.get_class_colors()
            else:
                class_colors = None
                
            visualize_predictions(
                model, 
                val_loader, 
                self.device, 
                num_samples=5,
                save_dir=os.path.join(self.figures_dir, f'fold_{fold_idx}_predictions'),
                class_colors=class_colors
            )
            
            # Update best fold
            if val_miou > cv_results['best_miou']:
                cv_results['best_miou'] = val_miou
                cv_results['best_fold'] = fold_idx
                
                # Copy best model to experiment root
                import shutil
                shutil.copy(
                    os.path.join(fold_save_dir, 'model_best.pth'),
                    os.path.join(self.models_dir, 'best_model.pth')
                )
        
        # Calculate overall cross-validation results
        cv_results['mean_miou'] = np.mean(cv_results['all_mious'])
        cv_results['std_miou'] = np.std(cv_results['all_mious'])
        
        self.logger.info(f"Cross-validation completed - Mean mIoU: {cv_results['mean_miou']:.4f} ± {cv_results['std_miou']:.4f}")
        
        # Save cross-validation results
        cv_results_path = os.path.join(self.experiment_dir, 'cv_results.json')
        with open(cv_results_path, 'w') as f:
            json.dump({k: v for k, v in cv_results.items() if k != 'fold_metrics'}, f, indent=4)
        
        # Save fold metrics separately as they can be large
        fold_metrics_path = os.path.join(self.experiment_dir, 'fold_metrics.json')
        with open(fold_metrics_path, 'w') as f:
            json.dump({'fold_metrics': cv_results['fold_metrics']}, f, indent=4)
        
        # Visualize cross-validation results
        self.visualize_cv_results(cv_results)
        
        return cv_results
    
    def visualize_cv_results(self, cv_results):
        """
        Visualize cross-validation results.
        
        Args:
            cv_results: Dictionary with cross-validation results
        """
        # Create a bar plot of mIoU for each fold
        plt.figure(figsize=(10, 6))
        fold_indices = list(range(len(cv_results['all_mious'])))
        plt.bar(fold_indices, cv_results['all_mious'])
        plt.axhline(y=cv_results['mean_miou'], color='r', linestyle='-', 
                   label=f'Mean: {cv_results["mean_miou"]:.4f} ± {cv_results["std_miou"]:.4f}')
        plt.xlabel('Fold')
        plt.ylabel('mIoU')
        plt.title('Cross-Validation Results: mIoU per Fold')
        plt.legend()
        plt.xticks(fold_indices, [f'Fold {i}' for i in fold_indices])
        plt.grid(True)
        plt.savefig(os.path.join(self.figures_dir, 'cv_miou_per_fold.png'))
        plt.close()
        
        # Create a plot of learning curves for all folds
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 2, 1)
        for fold_idx, fold_data in enumerate(cv_results['fold_metrics']):
            plt.plot(fold_data['learning_curve']['train_loss'], label=f'Fold {fold_idx}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        for fold_idx, fold_data in enumerate(cv_results['fold_metrics']):
            plt.plot(fold_data['learning_curve']['val_loss'], label=f'Fold {fold_idx}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        for fold_idx, fold_data in enumerate(cv_results['fold_metrics']):
            plt.plot(fold_data['learning_curve']['train_miou'], label=f'Fold {fold_idx}')
        plt.xlabel('Epochs')
        plt.ylabel('mIoU')
        plt.title('Training mIoU')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        for fold_idx, fold_data in enumerate(cv_results['fold_metrics']):
            plt.plot(fold_data['learning_curve']['val_miou'], label=f'Fold {fold_idx}')
        plt.xlabel('Epochs')
        plt.ylabel('mIoU')
        plt.title('Validation mIoU')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cv_learning_curves.png'))
        plt.close()
