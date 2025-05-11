#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import logging
import random
import numpy as np
from datetime import datetime

from datasets import CityscapesDataModule, num_classes
from models.unet import UNet
from models.deeplabv3 import DeepLabV3
from utils.simple_experiment import SimpleExperiment

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
    
    # Check for resume checkpoint
    checkpoint_path = args.resume
    resume_training = checkpoint_path != ''
    if resume_training:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Check for pretrained model
    pretrained_path = args.pretrained
    use_pretrained = pretrained_path != ''
    if use_pretrained:
        print(f"Using pretrained model weights from: {pretrained_path}")
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model file not found: {pretrained_path}")
    
    # Setup data module
    data_module = CityscapesDataModule(
        data_dir=config['data']['root'],
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers,
        k_folds=config['training']['k_folds'],  # Not used in simple experiment, but kept for compatibility
        image_size=tuple(config['data']['image_size']),
        augment=config['data']['augment']
    )
    
    # Setup experiment
    experiment_name = f"{args.model}_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model_class = None
    if args.model.lower() == 'unet':
        model_class = UNet
    elif args.model.lower() == 'deeplabv3':
        model_class = DeepLabV3
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Setup simple experiment
    experiment = SimpleExperiment(
        model_class=model_class,
        dataset_module=data_module,
        experiment_name=experiment_name,
        device=device,
        base_save_dir=config['output']['base_dir'],
        config=config    )
    
    # Run simple experiment
    num_epochs = args.epochs if args.epochs > 0 else config['training']['epochs']
    experiment.run_simple_experiment(
        num_classes=int(config['model']['n_classes']),
        num_epochs=int(num_epochs),
        optimizer_params={
            'lr': float(config['training']['lr']),
            'weight_decay': float(config['training']['weight_decay'])
        },
        early_stopping_patience=int(config['training'].get('early_stopping_patience', 10)),
        checkpoint_path=args.resume if args.resume else None,
        pretrained_path=args.pretrained if args.pretrained else None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train semantic segmentation models using original train/val split")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeplabv3'], help='Model to train')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (e.g., cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train (0 = use config value)')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training from')
    parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model weights to use as initialization')
    
    args = parser.parse_args()
    main(args)
