#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime

from datasets import CityscapesDataModule, num_classes
from models.unet import UNet
from models.deeplabv3 import DeepLabV3
from utils.experiments import CrossValidationExperiment

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup data module
    data_module = CityscapesDataModule(
        data_dir=config['data']['root'],
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers,
        k_folds=config['training']['k_folds'],
        image_size=tuple(config['data']['image_size']),
        augment=config['data']['augment']
    )
    
    # Setup experiment
    experiment_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model_class = None
    if args.model.lower() == 'unet':
        model_class = UNet
    elif args.model.lower() == 'deeplabv3':
        model_class = DeepLabV3
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Setup cross-validation experiment
    experiment = CrossValidationExperiment(
        model_class=model_class,
        dataset_module=data_module,
        experiment_name=experiment_name,
        device=device,
        base_save_dir=config['output']['base_dir'],
        config=config
    )
    
    # Run cross-validation
    experiment.run_cross_validation(
        num_classes=config['model']['n_classes'],
        num_epochs=config['training']['epochs'],
        optimizer_params={
            'lr': config['training']['lr'],
            'weight_decay': config['training']['weight_decay']
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train semantic segmentation models with cross-validation")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeplabv3'], help='Model to train')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (e.g., cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
