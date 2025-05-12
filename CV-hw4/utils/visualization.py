#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from tqdm import tqdm
import sys
from PIL import Image
import logging
import cv2

def calculate_miou(confusion_matrix):
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
    class_weights = class_pixels / total_pixels
    
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

def calculate_confusion_matrix(pred, gt, num_classes, ignore_index=255):
    """Calculate confusion matrix between prediction and ground truth"""
    mask = (gt != ignore_index)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for t, p in zip(gt[mask].flatten(), pred[mask].flatten()):
        confusion_matrix[t, p] += 1
    
    return confusion_matrix


def evaluate_model(model, dataloader, device, num_classes, ignore_index=255):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to use for evaluation
        num_classes: Number of classes in the dataset
        ignore_index: Index to ignore in evaluation (default: 255)
        
    Returns:
        Dictionary with evaluation results including confusion_matrix and metrics
    """
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        i = 0 
        for inputs, targets in tqdm(dataloader, desc='Evaluating'):
            if i > 5: break # for debugging, remove this line in production
            i += 1
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Move tensors back to CPU for numpy processing
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()
            
            # 高效批量处理所有样本
            for target, pred in zip(targets, preds):
                # 使用快速的计算混淆矩阵方法
                mask = (target != ignore_index)
                t_flat = target[mask].flatten()
                p_flat = pred[mask].flatten()
                
                # 使用numpy的高效运算更新混淆矩阵
                if len(t_flat) > 0:
                    # 确保索引在有效范围内
                    valid_indices = (t_flat < num_classes) & (p_flat < num_classes)
                    t_flat = t_flat[valid_indices]
                    p_flat = p_flat[valid_indices]
                    
                    # 使用numpy的直接索引加速
                    if len(t_flat) > 0:
                        np.add.at(confusion_matrix, (t_flat, p_flat), 1)
    
    # Calculate metrics, including weighted mIoU
    miou, iou_per_class, weighted_miou, class_weights = calculate_miou(confusion_matrix)
    
    # Return results as dictionary
    results = {
        'confusion_matrix': confusion_matrix,
        'miou': miou,
        'weighted_miou': weighted_miou,
        'iou_per_class': iou_per_class,
        'class_weights': class_weights
    }
    
    return results

def visualize_predictions(model, dataloader, device, num_samples=5, save_dir='outputs/figures', class_colors=None):
    """
    Visualize model predictions on random samples.
    
    Args:
        model: The trained model
        dataloader: DataLoader for the dataset
        device: Device to use for inference
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
        class_colors: List of RGB colors for each class
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # If class colors not provided, use a colormap
    if class_colors is None:
        colormap = cm.get_cmap('tab20', 20)
        class_colors = [tuple(int(255 * c) for c in colormap(i)[:3]) for i in range(20)]
    
    model.eval()
    all_samples = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            all_samples.append((inputs, targets))
            if len(all_samples) >= num_samples:
                break
    
    # If we don't have enough samples
    num_samples = min(num_samples, len(all_samples))
    
    for i, (inputs, targets) in enumerate(all_samples[:num_samples]):
        # Move input to device and get prediction
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        
        # Move everything back to CPU and convert to numpy
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        preds = preds.cpu().numpy()
        
        # Process each image in the batch
        for j in range(inputs.shape[0]):
            # Create a figure with 3 subplots: input, ground truth, prediction
            plt.figure(figsize=(15, 5))
            
            # Plot input image
            plt.subplot(1, 3, 1)
            # Unnormalize the image
            img = inputs[j].transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title('Input Image')
            plt.axis('off')
            
            # Plot ground truth
            plt.subplot(1, 3, 2)
            gt_colored = np.zeros((targets[j].shape[0], targets[j].shape[1], 3), dtype=np.uint8)
            for c in range(len(class_colors)):
                mask = (targets[j] == c)
                gt_colored[mask] = class_colors[c]
            plt.imshow(gt_colored)
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Plot prediction
            plt.subplot(1, 3, 3)
            pred_colored = np.zeros((preds[j].shape[0], preds[j].shape[1], 3), dtype=np.uint8)
            for c in range(len(class_colors)):
                mask = (preds[j] == c)
                pred_colored[mask] = class_colors[c]
            plt.imshow(pred_colored)
            plt.title('Prediction')
            plt.axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{i}_{j}.png'))
            plt.close()

def visualize_class_performance(confusion_matrix, class_names, save_path='outputs/figures/class_performance.png'):
    """
    Visualize IoU performance for each class.
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: List of class names
        save_path: Path to save the visualization
    """
    # Calculate IoU and weighted mIoU
    miou, iou_per_class, weighted_miou, class_weights = calculate_miou(confusion_matrix)
    iou_per_class = np.array(iou_per_class)  # Convert to numpy array
    
    # Create a figure
    plt.figure(figsize=(15, 8))
    
    # Plot IoU for each class, with bar transparency proportional to class weight
    bars = plt.bar(range(len(iou_per_class)), iou_per_class, color='skyblue', width=0.6)
    
    # Add transparency to bars based on class weights
    if len(class_weights) > 0 and max(class_weights) > 0:  # Avoid division by zero
        for i, bar in enumerate(bars):
            bar.set_alpha(0.3 + 0.7 * class_weights[i] / max(class_weights))
    
    plt.xticks(range(len(iou_per_class)), class_names, rotation=90)
    plt.xlabel('Class')
    plt.ylabel('IoU')
    plt.title('IoU per Class')
    plt.grid(axis='y')
    
    # Add mean IoU and weighted mean IoU lines
    plt.axhline(y=miou, color='r', linestyle='-', label=f'mIoU (Simple Mean): {miou:.4f}')
    plt.axhline(y=weighted_miou, color='g', linestyle='--', label=f'wmIoU (Weighted Mean): {weighted_miou:.4f}')
    plt.legend()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Create an additional visualization showing the relationship between class weight and IoU
    weight_vs_iou_path = os.path.join(os.path.dirname(save_path), "class_weight_vs_iou.png")
    visualize_class_weight_vs_iou(iou_per_class, class_weights, class_names, miou, weighted_miou, weight_vs_iou_path)
    
    return miou, weighted_miou


def visualize_class_weight_vs_iou(iou_per_class, class_weights, class_names, miou, weighted_miou, save_path):
    """
    Visualize the relationship between class weight and IoU.
    
    Args:
        iou_per_class: IoU for each class
        class_weights: Weight for each class based on pixel frequency
        class_names: List of class names
        miou: Mean IoU (simple average)
        weighted_miou: Weighted mean IoU
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with point size proportional to class weight
    for i, (iou, weight) in enumerate(zip(iou_per_class, class_weights)):
        plt.scatter(i, iou, s=weight*5000, alpha=0.6, color='blue')
        if weight > 0.01:  # Only label classes with significant weight
            plt.annotate(f"{weight:.3f}", (i, iou), 
                       textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('Class Weight vs. IoU')
    plt.ylabel('IoU')
    plt.xlabel('Class')
    plt.xticks(range(len(iou_per_class)), class_names, rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=miou, color='r', linestyle='-', label=f'Simple average: {miou:.4f}')
    plt.axhline(y=weighted_miou, color='g', linestyle='--', label=f'Weighted average: {weighted_miou:.4f}')
    plt.legend()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_confusion_matrix_visualization(confusion_matrix, class_names, 
                                         save_path='outputs/figures/confusion_matrix.png'):
    """
    Create and save a visualization of the confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix to visualize
        class_names: List of class names
        save_path: Path to save the visualization
    """
    # Normalize confusion matrix
    row_sums = confusion_matrix.sum(axis=1)
    norm_confusion_matrix = np.zeros_like(confusion_matrix, dtype=float)
    
    # Avoid division by zero
    for i in range(confusion_matrix.shape[0]):
        if row_sums[i] > 0:
            norm_confusion_matrix[i] = confusion_matrix[i] / row_sums[i]
    
    # Create a figure
    plt.figure(figsize=(15, 15))
    plt.imshow(norm_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    # Add labels and ticks
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Add values inside the cells
    fmt = '.2f'
    thresh = norm_confusion_matrix.max() / 2.
    for i in range(norm_confusion_matrix.shape[0]):
        for j in range(norm_confusion_matrix.shape[1]):
            plt.text(j, i, format(norm_confusion_matrix[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if norm_confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def decode_segmap(segmap, class_colors=None):
    """
    Convert a segmentation mask to a color image.
    
    Args:
        segmap: Segmentation mask
        class_colors: Dictionary or list of class colors
        
    Returns:
        RGB image
    """
    if class_colors is None:
        # Default color map
        colormap = cm.get_cmap('tab20', 20)
        class_colors = [tuple(int(255 * c) for c in colormap(i)[:3]) for i in range(20)]
    
    r = np.zeros_like(segmap).astype(np.uint8)
    g = np.zeros_like(segmap).astype(np.uint8)
    b = np.zeros_like(segmap).astype(np.uint8)
    
    for l, color in enumerate(class_colors):
        idx = (segmap == l)
        r[idx] = color[0]
        g[idx] = color[1]
        b[idx] = color[2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb
