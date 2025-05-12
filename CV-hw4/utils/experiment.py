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
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.trainer import Trainer
from utils.visualization import evaluate_model, visualize_predictions, visualize_class_performance, create_confusion_matrix_visualization
from sklearn.model_selection import KFold

class CrossValidationExperiment:
    """Class for running experiments using k-fold cross-validation."""
    
    def __init__(self, model_class, dataset_module, experiment_name, device,
                 base_save_dir='outputs', config=None):
        """
        Initialize the cross-validation experiment.
        
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
    
    def setup_kfold_dataloaders(self, fold_idx, k_folds):
        """Setup train and validation dataloaders for a specific fold."""
        self.dataset_module.setup()
        
        # For cross-validation, we'll use the train_dataset and split it
        dataset = self.dataset_module.train_dataset
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        # Use KFold from sklearn to split indices
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        folds = list(kfold.split(indices))
        
        # Get train and validation indices for current fold
        train_indices, val_indices = folds[fold_idx]
        
        # Create samplers for train and validation sets
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Create DataLoaders with samplers
        train_loader = DataLoader(
            dataset,
            batch_size=self.dataset_module.batch_size,
            sampler=train_sampler,
            num_workers=self.dataset_module.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=self.dataset_module.batch_size,
            sampler=val_sampler,
            num_workers=self.dataset_module.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, len(train_indices), len(val_indices)
    
    def run_cross_validation(self, num_classes=19, num_epochs=50, k_folds=5, criterion=None, 
                             optimizer_params=None, early_stopping_patience=10, 
                             checkpoint_path=None, pretrained_path=None):
        """
        Run experiment with k-fold cross-validation.
        
        Args:
            num_classes: Number of classes for the model
            num_epochs: Number of epochs to train
            k_folds: Number of folds for cross-validation
            criterion: Loss function (default: CrossEntropyLoss with class weights)
            optimizer_params: Parameters for the optimizer
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            checkpoint_path: Path to checkpoint to resume training from (optional)
            pretrained_path: Path to pretrained model weights to use as initialization (optional)
            
        Returns:
            results: Dictionary with training and validation results across all folds
        """
        # Set default optimizer parameters if not provided
        optimizer_params = optimizer_params or {
            'lr': 0.001, 
            'weight_decay': 1e-4
        }
        
        # Initialize storage for cross-validation results
        cv_results = {
            'fold_results': [],
            'mean_miou': 0.0,
            'std_miou': 0.0,
            'best_fold': 0,
            'best_miou': 0.0
        }
        
        # Loop through each fold
        for fold in range(k_folds):
            self.logger.info(f"Starting fold {fold+1}/{k_folds}")
            
            # Create fold-specific directories
            fold_models_dir = os.path.join(self.models_dir, f'fold_{fold}')
            fold_logs_dir = os.path.join(self.logs_dir, f'fold_{fold}')
            fold_figures_dir = os.path.join(self.figures_dir, f'fold_{fold}')
            
            os.makedirs(fold_models_dir, exist_ok=True)
            os.makedirs(fold_logs_dir, exist_ok=True)
            os.makedirs(fold_figures_dir, exist_ok=True)
            
            # Get dataloaders for this fold
            train_loader, val_loader, train_size, val_size = self.setup_kfold_dataloaders(fold, k_folds)
            
            self.logger.info(f"Fold {fold+1} - Training dataset size: {train_size}")
            self.logger.info(f"Fold {fold+1} - Validation dataset size: {val_size}")
            
            # Initialize model for this fold
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
                save_dir=fold_models_dir,    
                log_dir=fold_logs_dir
            )
            
            # Handle resuming from checkpoint for this fold
            start_epoch = 0
            fold_checkpoint_path = None
            if checkpoint_path is not None:
                # Check if there's a fold-specific checkpoint
                fold_checkpoint_dir = os.path.dirname(checkpoint_path)
                fold_checkpoint_name = os.path.basename(checkpoint_path)
                fold_checkpoint_path = os.path.join(fold_checkpoint_dir, f'fold_{fold}', fold_checkpoint_name)
                
                if os.path.exists(fold_checkpoint_path):
                    self.logger.info(f"Resuming fold {fold+1} training from checkpoint: {fold_checkpoint_path}")
                    start_epoch = trainer.load_checkpoint(fold_checkpoint_path)
                    self.logger.info(f"Resuming from epoch {start_epoch}")
                else:
                    self.logger.warning(f"No checkpoint found for fold {fold+1}, starting from scratch")
            
            # Train model for this fold
            train_metrics, val_metrics = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=num_epochs,
                start_epoch=start_epoch,
                early_stopping_patience=early_stopping_patience
            )
            
            # Save training history for this fold
            history = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss']
            }
            
            history_df = pd.DataFrame(history)
            history_path = os.path.join(fold_logs_dir, 'history.csv')
            history_df.to_csv(history_path, index=False)
            
            # Plot training history for this fold
            self.plot_training_history(history, save_path=os.path.join(fold_figures_dir, 'training_history.png'))
            
            # Evaluate model on validation set for this fold
            best_model_path = os.path.join(fold_models_dir, 'best_model.pth')
            model.load_state_dict(torch.load(best_model_path))
            
            self.logger.info(f"评估第 {fold+1} 折的模型...")
            val_results = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=self.device,
                num_classes=num_classes,
                ignore_index=self.config.get('data', {}).get('ignore_index', 255)
            )
            self.logger.info(f"第 {fold+1} 折评估完成！mIoU: {val_results['miou']:.4f}, 加权mIoU: {val_results['weighted_miou']:.4f}")
            
            # Generate visualizations for this fold
            visualize_predictions(
                model=model,
                dataloader=val_loader,
                device=self.device,
                num_samples=min(2, len(val_loader)),
                save_dir=fold_figures_dir
            )
            
            visualize_class_performance(
                confusion_matrix=val_results['confusion_matrix'],
                class_names=[f'Class {i}' for i in range(num_classes)],
                save_path=os.path.join(fold_figures_dir, 'class_performance.png')
            )
            
            create_confusion_matrix_visualization(
                confusion_matrix=val_results['confusion_matrix'],
                class_names=[f'Class {i}' for i in range(num_classes)],
                save_path=os.path.join(fold_figures_dir, 'confusion_matrix.png')
            )
            
            # Save results for this fold
            fold_results = {
                'fold': fold,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'val_results': val_results,
                'best_epoch': trainer.best_epoch
            }
            
            # Convert numpy arrays to lists for JSON serialization
            json_serializable_fold_results = self.convert_numpy_to_python(fold_results)
            
            fold_results_path = os.path.join(fold_logs_dir, 'results.json')
            with open(fold_results_path, 'w') as f:
                json.dump(json_serializable_fold_results, f, indent=4)
            
            # Store fold results for aggregation
            cv_results['fold_results'].append(json_serializable_fold_results)
            
            # Track best fold
            if val_results['miou'] > cv_results['best_miou']:
                cv_results['best_miou'] = val_results['miou']
                cv_results['best_fold'] = fold
                
                # Copy the best model to the experiment root directory
                best_model_path = os.path.join(fold_models_dir, 'best_model.pth')
                best_model_copy_path = os.path.join(self.models_dir, 'best_overall_model.pth')
                torch.save(torch.load(best_model_path), best_model_copy_path)
        
        # Calculate aggregate statistics
        mious = [fold_res['val_results']['miou'] for fold_res in cv_results['fold_results']]
        cv_results['mean_miou'] = np.mean(mious)
        cv_results['std_miou'] = np.std(mious)
        
        # Generate aggregated validation metrics across folds
        self.logger.info(f"交叉验证完成！平均 mIoU: {cv_results['mean_miou']:.4f} ± {cv_results['std_miou']:.4f}")
        self.logger.info(f"最佳折: {cv_results['best_fold'] + 1}, mIoU: {cv_results['best_miou']:.4f}")
        
        # Final evaluation on the official validation set using the best model
        self.logger.info(f"使用最佳模型在官方验证集上进行最终评估...")
        
        # Load best model from cross-validation
        best_model_path = os.path.join(self.models_dir, 'best_overall_model.pth')
        best_model = self.model_class(num_classes=num_classes).to(self.device)
        best_model.load_state_dict(torch.load(best_model_path))
        
        # Create a loader for the official validation set
        val_loader = DataLoader(
            self.dataset_module.val_dataset,
            batch_size=self.dataset_module.batch_size,
            shuffle=False,
            num_workers=self.dataset_module.num_workers,
            pin_memory=True
        )
        
        # Evaluate on the official validation set
        val_results = evaluate_model(
            model=best_model,
            dataloader=val_loader,
            device=self.device,
            num_classes=num_classes,
            ignore_index=self.config.get('data', {}).get('ignore_index', 255)
        )
        
        self.logger.info(f"官方验证集评估结果 - mIoU: {val_results['miou']:.4f}, 加权mIoU: {val_results['weighted_miou']:.4f}")
        
        # Save the official validation set results
        cv_results['official_val_results'] = self.convert_numpy_to_python(val_results)
        
        # Generate visualizations on the official validation set
        final_figures_dir = os.path.join(self.figures_dir, 'final_validation')
        os.makedirs(final_figures_dir, exist_ok=True)
        
        visualize_predictions(
            model=best_model,
            dataloader=val_loader,
            device=self.device,
            num_samples=min(5, len(val_loader)),
            save_dir=final_figures_dir
        )
        
        visualize_class_performance(
            confusion_matrix=val_results['confusion_matrix'],
            class_names=[f'Class {i}' for i in range(num_classes)],
            save_path=os.path.join(final_figures_dir, 'class_performance.png')
        )
        
        create_confusion_matrix_visualization(
            confusion_matrix=val_results['confusion_matrix'],
            class_names=[f'Class {i}' for i in range(num_classes)],
            save_path=os.path.join(final_figures_dir, 'confusion_matrix.png')
        )
        
        # Plot cross-validation results
        self.plot_cv_results(cv_results)
        
        # Save aggregated results
        results_path = os.path.join(self.logs_dir, 'cv_results.json')
        with open(results_path, 'w') as f:
            json.dump(cv_results, f, indent=4)
        
        return cv_results
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history for a fold."""
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
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.figures_dir, 'training_history.png'))
        plt.close()
    
    def plot_cv_results(self, cv_results):
        """Plot cross-validation results."""
        # Extract mIoU values from each fold
        folds = list(range(1, len(cv_results['fold_results']) + 1))
        mious = [fold_res['val_results']['miou'] for fold_res in cv_results['fold_results']]
        
        plt.figure(figsize=(12, 6))
        
        # Plot mIoU for each fold
        plt.bar(folds, mious, color='skyblue')
        plt.axhline(y=cv_results['mean_miou'], color='r', linestyle='-', 
                   label=f"Mean mIoU: {cv_results['mean_miou']:.4f} ± {cv_results['std_miou']:.4f}")
        
        plt.xlabel('Fold')
        plt.ylabel('mIoU')
        plt.title('Cross-Validation Results')
        plt.xticks(folds)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add values on top of bars
        for i, miou in enumerate(mious):
            plt.text(i+1, miou+0.01, f'{miou:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cv_results.png'))
        plt.close()
    
    def convert_numpy_to_python(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self.convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
