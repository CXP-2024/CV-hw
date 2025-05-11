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
        # self.train_miou_history = [] # Removed mIoU history for training
        # self.val_miou_history = [] # Removed mIoU history for validation
        
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
        """
        self.model.train()
        running_loss = 0.0
        # confusion_matrix = np.zeros((self.model.module.num_classes if isinstance(self.model, nn.DataParallel) 
        #                            else self.model.num_classes, 
        #                            self.model.module.num_classes if isinstance(self.model, nn.DataParallel) 
        #                            else self.model.num_classes))
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for i, (inputs, targets) in enumerate(pbar):
            if i > 10: break
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
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
        # Calculate metrics
        avg_loss = running_loss / len(train_loader)
        # miou = self.calculate_miou(confusion_matrix) # Removed mIoU calculation for training
        
        # Update history
        self.train_loss_history.append(avg_loss)
        # self.train_miou_history.append(miou) # Removed mIoU history update for training
        
        return avg_loss # Return only avg_loss for training
    
    def validate(self, val_loader, epoch):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the validation set
        """
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for i, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                #if i > 1: break # 只处理前1个batch
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                running_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = running_loss / len(val_loader)
        
        # Update history
        self.val_loss_history.append(avg_loss)
        
        return avg_loss # Return only avg_loss for validation
    
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
        best_val_loss = float('inf') # Changed to track best validation loss (lower is better)
        best_epoch = 0
        patience_counter = 0
        
        self.logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(start_epoch, epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch) # Now only returns val_loss
            
            self.logger.info(f"Epoch {epoch}/{epochs-1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Check for improvement (using val_loss - lower is better)            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # 保存最佳模型
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                self.logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f} at epoch {epoch}")
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.info(f"No improvement in validation loss for {patience_counter} epochs")
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'latest_model.pth'))
                self.logger.info(f"Latest model saved at epoch {epoch}")

            if early_stopping_patience and patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch} due to no improvement for {early_stopping_patience} epochs.")
                break
        
        self.logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        
        # Store best epoch and metric for reference
        self.best_epoch = best_epoch
        self.best_metric = best_val_loss # Store best validation loss here
        
        # Return metrics
        train_metrics = {'loss': self.train_loss_history}
        val_metrics = {'loss': self.val_loss_history}
        
        return train_metrics, val_metrics
    
    def calculate_miou(self, confusion_matrix):
        """
        Calculate mean Intersection over Union (mIoU) from the confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix for the predictions
        
        Returns:
            miou: Mean IoU score
        """
        # Calculate IoU for each class
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
        
        # Compute mean IoU, ignoring NaN values (classes with no samples in either prediction or ground truth)
        miou = np.nanmean(ious)
        
        return miou
    
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
        
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        
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
        
        plt.figure(figsize=(10, 6))
        
        # Plot loss curves
        plt.plot(epochs_range, self.train_loss_history, label='Train')
        plt.plot(epochs_range, self.val_loss_history, label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'learning_curves.png'))
        plt.close()
