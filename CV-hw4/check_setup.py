#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import yaml

def check_imports():
    """Test importing key modules"""
    try:
        import numpy as np
        import torch
        import torchvision
        import yaml
        import sklearn
        import pandas as pd
        import matplotlib.pyplot as plt
        import PIL
        import cv2
        
        print("✓ All basic packages imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def check_project_modules():
    """Test importing project modules"""
    try:
        from datasets import CityscapesDataset, CityscapesDataModule
        from models.unet import UNet
        from models.deeplabv3 import DeepLabV3
        from utils.trainer import Trainer
        from utils.visualization import evaluate_model
        
        print("✓ All project modules imported successfully")
    except ImportError as e:
        print(f"✗ Project module import error: {e}")
        return False
    
    return True

def check_yaml_config():
    """Test loading the YAML config file"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"✓ YAML config loaded successfully: {config['model']['type']} model configured")
    except Exception as e:
        print(f"✗ YAML config error: {e}")
        return False
    
    return True

def check_gpu():
    """Check if CUDA is available"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print(f"✓ CUDA is available: {device_count} device(s), using {device_name}")
    else:
        print("⚠ CUDA is not available, will use CPU (much slower)")
    
    return True

def check_cityscapes_data():
    """Check if Cityscapes dataset is available"""
    data_dir = os.path.join(os.getcwd(), 'data')
    leftimg_dir = os.path.join(data_dir, 'leftImg8bit')
    gtfine_dir = os.path.join(data_dir, 'gtFine')
    
    if not os.path.exists(leftimg_dir) or not os.path.exists(gtfine_dir):
        print("⚠ Cityscapes data directories not found. Please ensure:")
        print(f"  - {leftimg_dir} (for images)")
        print(f"  - {gtfine_dir} (for annotations)")
        return False
    
    # Check if train/val/test dirs exist
    splits = ['train', 'val', 'test']
    for split in splits:
        img_split_dir = os.path.join(leftimg_dir, split)
        if not os.path.exists(img_split_dir):
            print(f"⚠ Missing {split} split in leftImg8bit.")
            
        gt_split_dir = os.path.join(gtfine_dir, split)
        if not os.path.exists(gt_split_dir):
            print(f"⚠ Missing {split} split in gtFine.")
    
    # Try to instantiate the dataset
    try:
        from datasets import CityscapesDataset
        dataset = CityscapesDataset(root_dir=data_dir, split='val')
        print(f"✓ Cityscapes dataset loaded successfully, found {len(dataset)} validation samples")
    except Exception as e:
        print(f"✗ Error loading Cityscapes dataset: {e}")
        return False
    
    return True

def main():
    """Run all checks"""
    print("\n===== Semantic Segmentation Setup Check =====\n")
    
    print("\n----- Checking Python and packages -----")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    all_success = True
    
    all_success = check_imports() and all_success
    
    print("\n----- Checking project modules -----")
    all_success = check_project_modules() and all_success
    
    print("\n----- Checking configuration -----")
    all_success = check_yaml_config() and all_success
    
    print("\n----- Checking GPU availability -----")
    check_gpu()  # Don't fail if GPU is not available
    
    print("\n----- Checking Cityscapes dataset -----")
    all_success = check_cityscapes_data() and all_success
    
    print("\n=======================================")
    if all_success:
        print("\n✓ All checks passed! You're ready to run the project.")
        print("\nSuggested steps:")
        print("1. Train a model:     python train.py --model unet")
        print("2. Evaluate:          python evaluate.py --checkpoint <path/to/checkpoint>")
        print("3. Visualize results: python visualize.py --checkpoint <path/to/checkpoint> --input <input_image>")
    else:
        print("\n⚠ Some checks failed. Please fix the issues before running the project.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
