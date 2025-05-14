#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to test and visualize advanced data augmentations for the Cityscapes dataset.
This helps to understand the effects of advanced augmentation methods.
"""

import os
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from PIL import Image
import logging
from datasets import CityscapesDataModule
from datasets.cityscapes import decode_segmap, get_class_colors, SynchronizedTransforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def visualize_augmentations(image, mask, augmentation_level, image_size=(512, 1024), 
                           num_samples=5, save_dir=None, sample_name="sample"):
    """
    Apply and visualize augmentations on a single image-mask pair.
    
    Args:
        image (PIL.Image): Original image
        mask (PIL.Image): Original segmentation mask
        augmentation_level (str): Level of augmentation to apply ('none', 'standard', 'advanced')
        image_size (tuple): Target image size (height, width)
        num_samples (int): Number of augmented samples to generate
        save_dir (str): Directory to save visualization results
        sample_name (str): Base name for saved files
    """
    # Create output directory if it doesn't exist
    if save_dir is None:
        save_dir = f"outputs/augmentation_demo/{augmentation_level}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create transforms handler
    transform = SynchronizedTransforms(
        image_size=image_size,
        augmentation_level=augmentation_level,
        ignore_index=255
    )
    
    # Visualize original image and mask
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    # Convert mask to RGB for visualization
    mask_np = np.array(mask)
    mask_rgb = decode_segmap(mask_np)
    axs[1].imshow(mask_rgb)
    axs[1].set_title("Original Mask")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{sample_name}_original.png"))
    plt.close()
    
    # Generate and visualize augmented samples
    for i in range(num_samples):
        # Apply augmentation
        aug_image, aug_mask = transform(image.copy(), mask.copy())
        
        # Visualize augmented image and mask
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(aug_image)
        axs[0].set_title(f"{augmentation_level.capitalize()} Augmentation - Image")
        axs[0].axis('off')
        
        # Convert augmented mask to RGB for visualization
        aug_mask_np = np.array(aug_mask)
        aug_mask_rgb = decode_segmap(aug_mask_np)
        axs[1].imshow(aug_mask_rgb)
        axs[1].set_title(f"{augmentation_level.capitalize()} Augmentation - Mask")
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{sample_name}_augmented_{i+1}.png"))
        plt.close()
    
    logger.info(f"Saved {num_samples} augmented samples to {save_dir}")

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create data module
    data_module = CityscapesDataModule(
        data_dir=args.data_dir,
        batch_size=1,  # We only need single samples
        augmentation_level=args.level
    )
    
    # Setup datasets
    data_module.setup()
    
    # Get a few random samples from the training set
    dataset = data_module.train_dataset
    indices = random.sample(range(len(dataset)), args.num_samples)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        # Get the original PIL images (before any transformation)
        img_path = dataset.image_paths[idx]
        mask_path = dataset.label_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        sample_name = f"sample_{i+1}"
        save_dir = os.path.join(args.output_dir, args.level)
        
        # Apply and visualize augmentations
        visualize_augmentations(
            image=image,
            mask=mask,
            augmentation_level=args.level,
            image_size=data_module.image_size,
            num_samples=args.variations,
            save_dir=save_dir,
            sample_name=sample_name
        )
    
    logger.info(f"Completed augmentation visualization for {args.num_samples} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualize data augmentations")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to Cityscapes data directory")
    parser.add_argument("--level", type=str, choices=["none", "standard", "advanced"], default="advanced",
                       help="Augmentation level to visualize")
    parser.add_argument("--num_samples", type=int, default=3, 
                       help="Number of different samples to augment")
    parser.add_argument("--variations", type=int, default=5, 
                       help="Number of augmented variations to generate for each sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="outputs/augmentation_visualization",
                       help="Directory to save visualization results")
    
    args = parser.parse_args()
    main(args)
