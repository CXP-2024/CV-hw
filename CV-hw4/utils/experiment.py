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
from utils.losses import LabelSmoothCrossEntropyLoss, DiceLoss, CombinedLoss

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
        
        # Use the new method in CityscapesDataModule that handles cross-validation properly
        train_loader, val_loader, train_size, val_size = self.dataset_module.get_cross_val_dataloaders(fold_idx)
        
        return train_loader, val_loader, train_size, val_size
    
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
                
                # Get loss function type
                loss_type = self.config.get('training', {}).get('loss', 'cross_entropy')
                self.logger.info(f"Using loss function: {loss_type}")
                
                if loss_type == 'label_smoothing':
                    criterion = LabelSmoothCrossEntropyLoss(
                        smoothing=0.1, 
                        ignore_index=ignore_index
                    )
                    self.logger.info("Using Label Smoothing Cross Entropy Loss, smoothing factor=0.1")
                elif loss_type == 'dice':
                    criterion = DiceLoss(ignore_index=ignore_index)
                    self.logger.info("Using Dice Loss")
                elif loss_type == 'combined':
                    criterion = CombinedLoss(
                        ce_weight=0.6, 
                        dice_weight=0.4, 
                        smoothing=0.1, 
                        ignore_index=ignore_index
                    )
                    self.logger.info("Using Combined Loss: 0.6 * CE + 0.4 * Dice")
                else:
                    # Default: use cross entropy loss
                    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
                    self.logger.info("Using standard Cross Entropy Loss")
            
            # Initialize optimizer with explicit type conversion
            lr = float(optimizer_params.get('lr', 0.001))
            weight_decay = float(optimizer_params.get('weight_decay', 1e-4))
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )            # Initialize scheduler based on config
            lr_schedule = self.config.get('training', {}).get('lr_schedule', 'plateau')
            
            if lr_schedule == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs, eta_min=lr * 0.05
                )
                self.logger.info(f"Using Cosine Annealing LR scheduler, initial LR: {lr}, min LR: {lr * 0.01}")
            elif lr_schedule == 'warmrestart':
                # Using cosine annealing with warm restarts
                T_0 = self.config.get('training', {}).get('lr_warmrestart_T0', 10)  # First restart cycle
                T_mult = self.config.get('training', {}).get('lr_warmrestart_T_mult', 2)  # Cycle multiplier after each restart
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6
                )
                self.logger.info(f"Using Cosine Annealing Warm Restart scheduler, T_0={T_0}, T_mult={T_mult}, initial LR: {lr}")
            elif lr_schedule == 'onecycle':
                steps_per_epoch = len(train_loader) 
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=lr, total_steps=steps_per_epoch * num_epochs,
                    pct_start=0.3, anneal_strategy='cos'
                )
                self.logger.info(f"使用单循环学习率调度，初始学习率: {lr}, 峰值学习率: {lr}")
            else:
                # Default: ReduceLROnPlateau
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3
                )
                self.logger.info(f"使用ReduceLROnPlateau学习率调度，初始学习率: {lr}, 衰减因子: 0.5, 耐心值: 3")
            
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
                'val_loss': val_metrics['loss'],
                'train_miou': train_metrics['miou'],
                'val_miou': val_metrics['miou'],
                'train_weighted_miou': train_metrics['weighted_miou'],
                'val_weighted_miou': val_metrics['weighted_miou']
            }
            
            history_df = pd.DataFrame(history)
            history_path = os.path.join(fold_logs_dir, 'history.csv')
            history_df.to_csv(history_path, index=False)
            
            # Plot training history for this fold
            self.plot_training_history(history, save_path=os.path.join(fold_figures_dir, 'training_history.png'))
            
            # Evaluate model on validation set for this fold
            best_model_path = os.path.join(fold_models_dir, 'best_model.pth')
            model.load_state_dict(torch.load(best_model_path))
            
            # Evaluate the current fold model on official validation set
            self.logger.info(f"Evaluating fold {fold+1} model on official validation set...")
            
            from utils.visualization import visualize_iou_metrics
            
            # Create dataloader for official validation set
            official_val_loader = DataLoader(
                self.dataset_module.val_dataset,
                batch_size=self.dataset_module.batch_size,
                shuffle=False,
                num_workers=self.dataset_module.num_workers,
                pin_memory=True
            )
            
            # Evaluate model performance on official validation set
            official_val_results = evaluate_model(
                model=model,
                dataloader=official_val_loader,
                device=self.device,
                num_classes=num_classes,
                ignore_index=self.config.get('data', {}).get('ignore_index', 255)
            )
            
            self.logger.info(f"Fold {fold+1} evaluation on official validation set - mIoU: {official_val_results['miou']:.4f}, Weighted mIoU: {official_val_results['weighted_miou']:.4f}")
            
            # Generate visualizations for official validation set
            official_fold_figures_dir = os.path.join(fold_figures_dir, 'official_val')
            os.makedirs(official_fold_figures_dir, exist_ok=True)
            
            visualize_predictions(
                model=model,
                dataloader=official_val_loader,
                device=self.device,
                num_samples=min(3, len(official_val_loader)),
                save_dir=official_fold_figures_dir
            )
            
            visualize_class_performance(
                confusion_matrix=official_val_results['confusion_matrix'],
                class_names=[f'Class {i}' for i in range(num_classes)],
                save_path=os.path.join(official_fold_figures_dir, 'class_performance.png')
            )
            
            create_confusion_matrix_visualization(
                confusion_matrix=official_val_results['confusion_matrix'],
                class_names=[f'Class {i}' for i in range(num_classes)],
                save_path=os.path.join(official_fold_figures_dir, 'confusion_matrix.png')
            )
            
            visualize_iou_metrics(
                iou_per_class=official_val_results['iou_per_class'],
                class_names=[f'Class {i}' for i in range(num_classes)],
                save_path=os.path.join(official_fold_figures_dir, 'iou_per_class.png')
            )
            
            # Save results for this fold
            fold_results = {
                'fold': fold,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'official_val_results': official_val_results,
                'best_epoch': trainer.best_epoch
            }
            
            # Convert numpy arrays to lists for JSON serialization
            json_serializable_fold_results = self.convert_numpy_to_python(fold_results)
            
            fold_results_path = os.path.join(fold_logs_dir, 'results.json')
            with open(fold_results_path, 'w') as f:
                json.dump(json_serializable_fold_results, f, indent=4)
            
            # Store fold results for aggregation
            cv_results['fold_results'].append(json_serializable_fold_results)
            
            # Track best fold based on official validation set
            if official_val_results['miou'] > cv_results.get('best_miou', 0):
                cv_results['best_miou'] = official_val_results['miou']
                cv_results['best_fold'] = fold
                
                # Copy the best model to the experiment root directory
                best_model_path = os.path.join(fold_models_dir, 'best_model.pth')
                best_model_copy_path = os.path.join(self.models_dir, 'best_overall_model.pth')
                torch.save(torch.load(best_model_path), best_model_copy_path)
                
            # 保存每一折的最佳模型权重 - 用于后续可能的模型融合
            best_fold_model_path = os.path.join(fold_models_dir, 'best_model.pth')
            best_fold_model_state = torch.load(best_fold_model_path)
            all_fold_models_dir = os.path.join(self.models_dir, 'all_folds')
            os.makedirs(all_fold_models_dir, exist_ok=True)
            torch.save(best_fold_model_state, os.path.join(all_fold_models_dir, f'fold_{fold}_best_model.pth'))
        
        # Calculate aggregate statistics for official validation set
        official_mious = [fold_res['official_val_results']['miou'] for fold_res in cv_results['fold_results']]
        cv_results['mean_miou'] = np.mean(official_mious)
        cv_results['std_miou'] = np.std(official_mious)
            
        # Best fold is already tracked during the loop
        
        ###########################################################################################################
        # 实现多种"最佳"模型保存模式
        # 1. 平均权重模型 - 将所有折最佳模型的权重平均
        self.logger.info("创建平均权重模型...")
        avg_model = self.model_class(num_classes=num_classes).to(self.device)
        avg_state_dict = {}
        
        # 首先加载第一个模型的权重作为基准
        first_fold_model_path = os.path.join(all_fold_models_dir, 'fold_0_best_model.pth')
        first_fold_state_dict = torch.load(first_fold_model_path)
        
        # 初始化平均权重字典
        for key in first_fold_state_dict.keys():
            avg_state_dict[key] = torch.zeros_like(first_fold_state_dict[key])
          # 累加所有模型权重
        for fold in range(k_folds):
            fold_model_path = os.path.join(all_fold_models_dir, f'fold_{fold}_best_model.pth')
            fold_state_dict = torch.load(fold_model_path)
            for key in avg_state_dict.keys():
                avg_state_dict[key] += fold_state_dict[key]
        
        # 计算平均值
        for key in avg_state_dict.keys():
            # 检查是否是整型张量，如果是则需要特殊处理
            if avg_state_dict[key].dtype == torch.long or avg_state_dict[key].dtype == torch.int:
                # 对于整型张量，先转换为浮点型，再计算，然后转回整型
                avg_state_dict[key] = (avg_state_dict[key].float() / k_folds).long()
            else:
                # 浮点型张量可以直接除
                avg_state_dict[key] /= k_folds
        
        # 保存平均权重模型
        avg_model.load_state_dict(avg_state_dict)
        avg_model_path = os.path.join(self.models_dir, 'avg_weights_model.pth')
        torch.save(avg_state_dict, avg_model_path)
        self.logger.info(f"平均权重模型已保存到: {avg_model_path}")
        
        # 2. 最高性能模型 - 已在每个fold循环中实现
        self.logger.info(f"最高性能单折模型已保存 (Fold {cv_results['best_fold']+1})")
        
        # 3. 最近K折平均模型 (取最近的前K个性能最好的折)
        k_best = min(3, k_folds)  # 取前3个最好的折，或者全部折如果小于3
        self.logger.info(f"创建Top-{k_best}折平均模型...")
        
        # 找到性能最好的k_best个折
        fold_performances = [(fold, fold_res['official_val_results']['miou']) 
                             for fold, fold_res in enumerate(cv_results['fold_results'])]
        top_k_folds = sorted(fold_performances, key=lambda x: x[1], reverse=True)[:k_best]
        
        # 创建并初始化top-k平均模型
        top_k_avg_model = self.model_class(num_classes=num_classes).to(self.device)
        top_k_avg_state_dict = {}
        
        # 初始化平均权重字典
        for key in first_fold_state_dict.keys():
            top_k_avg_state_dict[key] = torch.zeros_like(first_fold_state_dict[key])
          # 累加前k个最佳模型的权重
        for fold_idx, _ in top_k_folds:
            fold_model_path = os.path.join(all_fold_models_dir, f'fold_{fold_idx}_best_model.pth')
            fold_state_dict = torch.load(fold_model_path)
            for key in top_k_avg_state_dict.keys():
                top_k_avg_state_dict[key] += fold_state_dict[key]
        
        # 计算平均值
        for key in top_k_avg_state_dict.keys():
            # 检查是否是整型张量，如果是则需要特殊处理
            if top_k_avg_state_dict[key].dtype == torch.long or top_k_avg_state_dict[key].dtype == torch.int:
                # 对于整型张量，先转换为浮点型，再计算，然后转回整型
                top_k_avg_state_dict[key] = (top_k_avg_state_dict[key].float() / k_best).long()
            else:
                # 浮点型张量可以直接除
                top_k_avg_state_dict[key] /= k_best
        
        # 保存top-k平均权重模型
        top_k_avg_model.load_state_dict(top_k_avg_state_dict)
        top_k_avg_model_path = os.path.join(self.models_dir, f'top_{k_best}_avg_model.pth')
        torch.save(top_k_avg_state_dict, top_k_avg_model_path)
        self.logger.info(f"Top-{k_best}折平均模型已保存到: {top_k_avg_model_path}")
        
        # 记录不同模型的保存路径
        cv_results['model_paths'] = {
            'best_single_fold': os.path.join(self.models_dir, 'best_overall_model.pth'),
            'avg_weights': avg_model_path,
            f'top_{k_best}_avg': top_k_avg_model_path
        }
        
        # Generate aggregated validation metrics across folds
        self.logger.info(f"Cross-validation completed! Average mIoU on official validation set: {cv_results['mean_miou']:.4f} ± {cv_results['std_miou']:.4f}")
        self.logger.info(f"Best fold: {cv_results['best_fold'] + 1}, mIoU: {cv_results['best_miou']:.4f}")
        
        # Final evaluation on the official validation set using the best model
        self.logger.info(f"Performing final evaluation on official validation set using the best model...")
        
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
        
        self.logger.info(f"Official validation set evaluation results - mIoU: {val_results['miou']:.4f}, Weighted mIoU: {val_results['weighted_miou']:.4f}")
        
        # Save the official validation set results
        cv_results['final_val_results'] = self.convert_numpy_to_python(val_results)
        
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
        
        # Evaluate all saved models
        self.logger.info("Evaluating all saved model variants...")
        model_evaluation = self.evaluate_saved_models(num_classes=num_classes)
        cv_results['model_evaluation'] = self.convert_numpy_to_python(model_evaluation)
        
        # Update results file with model evaluation information
        with open(results_path, 'w') as f:
            json.dump(cv_results, f, indent=4)
            
        self.logger.info("Cross-validation experiment completed!")
        
        return cv_results
    def plot_training_history(self, history, save_path=None):
        """Plot training history for a fold."""
        plt.figure(figsize=(10, 6))
        
        # Plot loss
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
        
        # 如果存在IoU指标，则绘制IoU图表
        if 'train_miou' in history and 'val_miou' in history:
            # 从save_path中提取目录和基本文件名
            if save_path:
                save_dir = os.path.dirname(save_path)
                iou_save_path = os.path.join(save_dir, 'iou_history.png')
            else:
                iou_save_path = os.path.join(self.figures_dir, 'iou_history.png')
            self.plot_iou_history(history, save_path=iou_save_path)
    
    def plot_iou_history(self, history, save_path=None):
        """Plot IoU metrics history."""
        plt.figure(figsize=(12, 10))
        
        # 创建2x1的子图布局
        plt.subplot(2, 1, 1)
        plt.plot(history['train_miou'], label='Train mIoU')
        plt.plot(history['val_miou'], label='Validation mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.title('Training and Validation Mean IoU')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(history['train_weighted_miou'], label='Train Weighted mIoU')
        plt.plot(history['val_weighted_miou'], label='Validation Weighted mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('Weighted Mean IoU')
        plt.title('Training and Validation Weighted Mean IoU')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.figures_dir, 'iou_history.png'))
        plt.close()
    
    def plot_cv_results(self, cv_results):
        """Plot cross-validation results."""
        # Extract mIoU values from each fold
        folds = list(range(1, len(cv_results['fold_results']) + 1))
        official_mious = [fold_res['official_val_results']['miou'] for fold_res in cv_results['fold_results']]
        
        plt.figure(figsize=(12, 6))
        
        # Plot mIoU for each fold
        bars = plt.bar(folds, official_mious, color='lightcoral')
                                   
        plt.axhline(y=cv_results['mean_miou'], color='red', linestyle='--', 
                   label=f"Mean: {cv_results['mean_miou']:.4f} ± {cv_results['std_miou']:.4f}")
        
        plt.xlabel('Fold')
        plt.ylabel('mIoU')
        plt.title('Cross-Validation Results on Official Validation Set')
        plt.xticks(folds)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add values on top of bars
        for i, miou in enumerate(official_mious):
            plt.text(i+1, miou + 0.01, f'{miou:.4f}', ha='center', fontsize=9)
        
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
            
    def evaluate_saved_models(self, num_classes=19):
        """Evaluate all saved models and compare their performance.
        
        This method loads different types of models saved during cross-validation,
        evaluates their performance on the validation set, and generates a comparison report.
        
        Args:
            num_classes: Number of classes
            
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("Starting evaluation of all saved models...")
        
        # 创建验证数据加载器
        val_loader = DataLoader(
            self.dataset_module.val_dataset,
            batch_size=self.dataset_module.batch_size,
            shuffle=False,
            num_workers=self.dataset_module.num_workers,
            pin_memory=True
        )
        
        # Get paths of models to evaluate
        model_paths = {}
        
        # 1. Best single fold model
        best_model_path = os.path.join(self.models_dir, 'best_overall_model.pth')
        if os.path.exists(best_model_path):
            model_paths['best_single_fold'] = best_model_path
        
        # 2. Average weights model
        avg_model_path = os.path.join(self.models_dir, 'avg_weights_model.pth')
        if os.path.exists(avg_model_path):
            model_paths['avg_weights'] = avg_model_path
        
        # 3. Top-K平均模型
        # 检查可能的top-k模型
        for k in [2, 3, 5]:
            top_k_path = os.path.join(self.models_dir, f'top_{k}_avg_model.pth')
            if os.path.exists(top_k_path):
                model_paths[f'top_{k}_avg'] = top_k_path
        
        # 初始化结果字典
        results = {}
        
        # 评估每个模型
        for model_name, model_path in model_paths.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Load model
            model = self.model_class(num_classes=num_classes).to(self.device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # Evaluate on validation set
            val_results = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=self.device,
                num_classes=num_classes,
                ignore_index=self.config.get('data', {}).get('ignore_index', 255)
            )
            
            self.logger.info(f"{model_name} - mIoU: {val_results['miou']:.4f}, Weighted mIoU: {val_results['weighted_miou']:.4f}")
            
            # 保存结果
            results[model_name] = {
                'miou': val_results['miou'],
                'weighted_miou': val_results['weighted_miou'],
                'iou_per_class': val_results['iou_per_class'],
                'pixel_accuracy': val_results.get('pixel_accuracy', None)
            }
        
        # Save evaluation results
        model_eval_dir = os.path.join(self.figures_dir, 'model_evaluation')
        os.makedirs(model_eval_dir, exist_ok=True)
        
        # Save results as JSON
        eval_results_path = os.path.join(model_eval_dir, 'model_comparison.json')
        with open(eval_results_path, 'w') as f:
            json.dump(self.convert_numpy_to_python(results), f, indent=4)
            
        # Plot comparison charts
        self._plot_model_comparison(results, save_dir=model_eval_dir)
        
        self.logger.info(f"Model evaluation completed, results saved to: {eval_results_path}")
        return results
    
    def _plot_model_comparison(self, results, save_dir):
        """Plot performance comparison between different models."""
        model_names = list(results.keys())
        mious = [results[name]['miou'] for name in model_names]
        weighted_mious = [results[name]['weighted_miou'] for name in model_names]
        
        # Set figure size and style
        plt.figure(figsize=(14, 8))
        plt.style.use('ggplot')
        
        # Bar width
        width = 0.35
        x = np.arange(len(model_names))
        
        # Create bar charts
        bar1 = plt.bar(x - width/2, mious, width, label='mIoU', color='skyblue')
        bar2 = plt.bar(x + width/2, weighted_mious, width, label='Weighted mIoU', color='lightcoral')
          # 添加标题和标签 (使用英文替代中文)
        plt.title('Performance Comparison of Different Model Saving Strategies', fontsize=16)
        plt.xlabel('Model Type', fontsize=14)
        plt.ylabel('IoU Score', fontsize=14)
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.ylim(min(min(mious), min(weighted_mious)) * 0.9, max(max(mious), max(weighted_mious)) * 1.05)
        
        # 在柱状图顶部添加数值
        for bar in [bar1, bar2]:
            for rect in bar:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height + 0.005,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.legend()
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300)
        plt.close()
        
        # 绘制每个模型的类别IoU比较
        if all('iou_per_class' in results[name] for name in model_names):
            plt.figure(figsize=(15, 10))
            
            # 获取类别数量
            num_classes = len(results[list(results.keys())[0]]['iou_per_class'])
            class_indices = list(range(num_classes))
            
            # 为每个模型绘制线图
            for model_name in model_names:
                plt.plot(class_indices, results[model_name]['iou_per_class'], 
                        marker='o', linestyle='-', label=model_name)
            plt.title('Per-class IoU Comparison Across Different Models', fontsize=16)
            plt.xlabel('Class Index', fontsize=14)
            plt.ylabel('IoU Score', fontsize=14)
            plt.xticks(class_indices)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'per_class_iou_comparison.png'), dpi=300)
            plt.close()
