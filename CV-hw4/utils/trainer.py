#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:
    """Trainer class for semantic segmentation models."""
    
    def __init__(self, model, criterion, optimizer, device, scheduler=None,
                 save_dir='outputs/models', log_dir='outputs'):
        """
        Initialize the trainer.
        
        Args:
            model: The semantic segmentation model
            criterion: Loss function
            optimizer: Optimizer for training
            device: Device to use (cuda/cpu)
            scheduler: Learning rate scheduler (optional)
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Metrics history
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_miou_history = []  # Added mIoU history for training
        self.val_miou_history = []    # Added mIoU history for validation
        self.train_weighted_miou_history = []  # Added weighted mIoU history for training
        self.val_weighted_miou_history = []    # Added weighted mIoU history for validation
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the epoch
            miou: Mean IoU for the epoch
            weighted_miou: Weighted mean IoU for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        # Get number of classes for confusion matrix
        if hasattr(self.model, 'module'):  # For DataParallel
            num_classes = self.model.module.num_classes
        else:
            num_classes = self.model.num_classes
            
        # Initialize confusion matrix for mIoU calculation
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for i, (inputs, targets) in enumerate(pbar):
            #if i > 11: break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            
            # Calculate confusion matrix for mIoU (only for a subset to save time)
            if i % 10 == 0:  # Process every 10th batch for faster training
                with torch.no_grad():
                    # Get predictions
                    preds = torch.argmax(outputs, dim=1)
                    
                    # Move tensors to CPU for numpy processing
                    preds = preds.cpu().numpy()
                    target_cpu = targets.cpu().numpy()
                    
                    # Update confusion matrix
                    ignore_index = 255  # Default ignore index
                    for t, p in zip(target_cpu, preds):
                        mask = (t != ignore_index)
                        t_flat = t[mask].flatten()
                        p_flat = p[mask].flatten()
                        
                        if len(t_flat) > 0:
                            valid_indices = (t_flat < num_classes) & (p_flat < num_classes)
                            t_flat = t_flat[valid_indices]
                            p_flat = p_flat[valid_indices]
                            
                            if len(t_flat) > 0:
                                np.add.at(confusion_matrix, (t_flat, p_flat), 1)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
        # Calculate metrics
        avg_loss = running_loss / len(train_loader)
        miou, iou_per_class, weighted_miou, _ = self.calculate_miou(confusion_matrix)
        
        # Update history
        self.train_loss_history.append(avg_loss)
        self.train_miou_history.append(miou)
        self.train_weighted_miou_history.append(weighted_miou)
        
        return avg_loss, miou, weighted_miou
    
    def validate(self, val_loader, epoch):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the validation set
            miou: Mean IoU for the validation set
            weighted_miou: Weighted mean IoU for the validation set
        """
        self.model.eval()
        running_loss = 0.0
        
        # Get number of classes for confusion matrix
        if hasattr(self.model, 'module'):  # For DataParallel
            num_classes = self.model.module.num_classes
        else:
            num_classes = self.model.num_classes
            
        # Initialize confusion matrix for mIoU calculation
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for i, (inputs, targets) in enumerate(pbar):
                #if i > 5: break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                running_loss += loss.item()
                
                # Get predictions for mIoU calculation
                preds = torch.argmax(outputs, dim=1)
                
                # Move tensors to CPU for numpy processing
                preds = preds.cpu().numpy()
                target_cpu = targets.cpu().numpy()
                
                # Update confusion matrix
                ignore_index = 255  # Default ignore index
                for t, p in zip(target_cpu, preds):
                    mask = (t != ignore_index)
                    t_flat = t[mask].flatten()
                    p_flat = p[mask].flatten()
                    
                    if len(t_flat) > 0:
                        valid_indices = (t_flat < num_classes) & (p_flat < num_classes)
                        t_flat = t_flat[valid_indices]
                        p_flat = p_flat[valid_indices]
                        
                        if len(t_flat) > 0:
                            np.add.at(confusion_matrix, (t_flat, p_flat), 1)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = running_loss / len(val_loader)
        miou, iou_per_class, weighted_miou, _ = self.calculate_miou(confusion_matrix)
        
        # Update history
        self.val_loss_history.append(avg_loss)
        self.val_miou_history.append(miou)
        self.val_weighted_miou_history.append(weighted_miou)
        
        return avg_loss, miou, weighted_miou
    
    def train(self, train_loader, val_loader, epochs, start_epoch=0, early_stopping_patience=None):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming training)
            early_stopping_patience: Number of epochs to wait for improvement before stopping (optional)
            
        Returns:
            train_metrics: Dictionary with training metrics
            val_metrics: Dictionary with validation metrics
        """
        best_val_loss = float('inf')
        best_val_miou = 0.0  # Track best validation mIoU (higher is better)
        best_epoch = 0
        patience_counter = 0
        
        self.logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(start_epoch, epochs):
            # Train for one epoch
            train_loss, train_miou, train_weighted_miou = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_miou, val_weighted_miou = self.validate(val_loader, epoch)
            
            # Log metrics
            self.logger.info(f"Epoch {epoch}/{epochs-1} - Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, Train wIoU: {train_weighted_miou:.4f}")
            self.logger.info(f"Epoch {epoch}/{epochs-1} - Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}, Val wIoU: {val_weighted_miou:.4f}")
            
            if self.scheduler:
                self.scheduler.step(val_loss)  # Adjust learning rate based on validation loss
            
            # Check for improvement (now considering both val_loss and val_miou)
            improved = False
            
            # Save if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                self.logger.info(f"New best validation loss: {best_val_loss:.4f}")
            
            # Or save if validation mIoU improves
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_epoch = epoch
                improved = True
                # Save the best model
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                self.logger.info(f"New best model saved with Val mIoU: {best_val_miou:.4f} at epoch {epoch}")
            
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.info(f"No improvement in validation metrics for {patience_counter} epochs")
                # Still save the latest model
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'latest_model.pth'))
                self.logger.info(f"Latest model saved at epoch {epoch}")

            if early_stopping_patience and patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch} due to no improvement for {early_stopping_patience} epochs.")
                break
        
        self.logger.info(f"Training completed. Best validation mIoU: {best_val_miou:.4f} at epoch {best_epoch}")
        
        # Store best epoch and metrics for reference
        self.best_epoch = best_epoch
        self.best_metric = best_val_miou  # Store best validation mIoU
        
        # Return metrics
        train_metrics = {
            'loss': self.train_loss_history,
            'miou': self.train_miou_history,
            'weighted_miou': self.train_weighted_miou_history
        }
        val_metrics = {
            'loss': self.val_loss_history,
            'miou': self.val_miou_history,
            'weighted_miou': self.val_weighted_miou_history
        }
        
        return train_metrics, val_metrics
    
    def calculate_miou(self, confusion_matrix):
        """
        Calculate Mean Intersection over Union from confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
            
        Returns:
            miou: Mean IoU score (simple average)
            iou_per_class: IoU for each class
            weighted_miou: Mean IoU weighted by class pixel frequency
            class_weights: Weight for each class based on pixel frequency
        """
        # Calculate IoU for each class
        iou_per_class = []
        for i in range(confusion_matrix.shape[0]):
            # True positives: diagonal elements
            tp = confusion_matrix[i, i]
            # False positives: sum of column i - true positives
            fp = confusion_matrix[:, i].sum() - tp
            # False negatives: sum of row i - true positives
            fn = confusion_matrix[i, :].sum() - tp
            
            # Calculate IoU if the denominator is not zero
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            iou_per_class.append(iou)
        
        # Convert to numpy array for easier math operations
        iou_per_class = np.array(iou_per_class)
        
        # Calculate mean IoU (simple average)
        miou = np.mean([iou for iou in iou_per_class if iou > 0])
        
        # Calculate class weights based on pixel frequency
        class_pixels = np.sum(confusion_matrix, axis=1)
        total_pixels = np.sum(class_pixels)
        class_weights = class_pixels / total_pixels if total_pixels > 0 else np.zeros_like(class_pixels)
        
        # Calculate weighted mean IoU
        # First handle possible NaN values
        valid_mask = ~np.isnan(iou_per_class)
        valid_ious = iou_per_class[valid_mask]
        valid_weights = class_weights[valid_mask]
        
        # Calculate weighted average (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        weight_sum = np.sum(valid_weights) + epsilon
        weighted_miou = np.sum(valid_ious * valid_weights) / weight_sum
        
        return miou, iou_per_class, weighted_miou, class_weights
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save checkpoint of the model.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'train_miou_history': self.train_miou_history,
            'val_miou_history': self.val_miou_history,
            'train_weighted_miou_history': self.train_weighted_miou_history,
            'val_weighted_miou_history': self.val_weighted_miou_history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save the latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_latest.pth'))
        
        # Save periodic checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save best model
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
            self.logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            epoch: The epoch to start training from
        """
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training and validation history
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        self.train_miou_history = checkpoint.get('train_miou_history', [])
        self.val_miou_history = checkpoint.get('val_miou_history', [])
        self.train_weighted_miou_history = checkpoint.get('train_weighted_miou_history', [])
        self.val_weighted_miou_history = checkpoint.get('val_weighted_miou_history', [])
        
        epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        
        self.logger.info(f"Loaded checkpoint from epoch {epoch-1}")
        return epoch
    
    def plot_learning_curves(self, epoch):
        """
        Plot and save learning curves.
        
        Args:
            epoch: Current epoch number
        """
        epochs_range = list(range(len(self.train_loss_history)))
        
        # Create a figure with 2 subplots (loss and mIoU)
        plt.figure(figsize=(15, 10))
        
        # Plot loss curves
        plt.subplot(2, 1, 1)
        plt.plot(epochs_range, self.train_loss_history, label='Train')
        plt.plot(epochs_range, self.val_loss_history, label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        
        # Plot mIoU curves
        plt.subplot(2, 1, 2)
        plt.plot(epochs_range, self.train_miou_history, label='Train mIoU')
        plt.plot(epochs_range, self.val_miou_history, label='Val mIoU')
        plt.plot(epochs_range, self.train_weighted_miou_history, '--', label='Train weighted mIoU')
        plt.plot(epochs_range, self.val_weighted_miou_history, '--', label='Val weighted mIoU')
        plt.xlabel('Epochs')
        plt.ylabel('mIoU')
        plt.title('Training and Validation mIoU Curves')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'learning_curves.png'))
        plt.close()
