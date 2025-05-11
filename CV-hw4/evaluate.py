#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import numpy as np
import logging
from pathlib import Path
import json

from datasets import CityscapesDataModule, CityscapesDataset
from models.unet import UNet
from models.deeplabv3 import DeepLabV3
from utils.visualization import evaluate_model, visualize_predictions, visualize_class_performance, create_confusion_matrix_visualization

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup data
    transform = None
    target_transform = None
    
    # Create dataset
    val_dataset = CityscapesDataset(
        root_dir=config['data']['root'],
        split=args.split,
        transform=transform,
        target_transform=target_transform
    )
    
    # Create dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load model
    if os.path.splitext(args.checkpoint)[1] == '':
        # If no extension provided, assume it's a directory with a best_model.pth file
        checkpoint_path = os.path.join(args.checkpoint, 'models', 'best_model.pth')
        experiment_dir = args.checkpoint
    else:
        # Otherwise use the provided path directly
        checkpoint_path = args.checkpoint
        experiment_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    
    # Determine model type from the checkpoint
    try:
        with open(os.path.join(experiment_dir, 'config.json'), 'r') as f:
            exp_config = json.load(f)
            model_type = exp_config.get('model', {}).get('type', 'unet').lower()
    except:
        # If config not found, try to infer from checkpoint path
        if 'unet' in checkpoint_path.lower():
            model_type = 'unet'
        elif 'deeplabv3' in checkpoint_path.lower():
            model_type = 'deeplabv3'
        else:
            model_type = args.model_type.lower()
    
    # Create model
    if model_type == 'unet':
        model = UNet(num_classes=config['model']['n_classes'])
    elif model_type == 'deeplabv3':
        model = DeepLabV3(num_classes=config['model']['n_classes'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Evaluate model
    num_classes = config['model']['n_classes']
    miou, iou_per_class, confusion_matrix = evaluate_model(model, val_loader, device, num_classes)
    
    # Print results
    print(f"Evaluation on {args.split} split:")
    print(f"mIoU: {miou:.4f}")
    
    # Get class names
    if hasattr(val_dataset, 'get_class_names'):
        class_names = val_dataset.get_class_names()
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Print per-class IoU
    print("\nPer-class IoU:")
    for i, (name, iou) in enumerate(zip(class_names, iou_per_class)):
        print(f"{name}: {iou:.4f}")
    
    # Create output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, f'eval_{args.split}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save evaluation results
    results = {
        'miou': float(miou),
        'iou_per_class': {name: float(iou) for name, iou in zip(class_names, iou_per_class)},
        'confusion_matrix': confusion_matrix.tolist()
    }
    
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create visualizations
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Visualize class performance
    visualize_class_performance(
        iou_per_class, 
        class_names, 
        save_path=os.path.join(figures_dir, 'class_performance.png')
    )
    
    # Visualize confusion matrix
    create_confusion_matrix_visualization(
        confusion_matrix, 
        class_names,
        save_path=os.path.join(figures_dir, 'confusion_matrix.png')
    )
    
    # Visualize predictions
    if hasattr(val_dataset, 'get_class_colors'):
        class_colors = val_dataset.get_class_colors()
    else:
        class_colors = None
    
    visualize_predictions(
        model, 
        val_loader, 
        device, 
        num_samples=args.num_samples,
        save_dir=os.path.join(figures_dir, 'predictions'),
        class_colors=class_colors
    )
    
    print(f"\nEvaluation results and visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation models")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint or experiment directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (e.g., cuda, cpu)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'deeplabv3'], 
                        help='Model type (if not inferrable from checkpoint)')
    
    args = parser.parse_args()
    main(args)
