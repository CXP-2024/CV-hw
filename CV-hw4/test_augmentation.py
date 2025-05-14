#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import logging

from datasets.cityscapes import CityscapesDataModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def main(args):
    """Main function to test data augmentation."""
    logger.info("Testing data augmentation...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Augmentation level: {args.augmentation_level}")

    # Setup the data module
    data_module = CityscapesDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.height, args.width),
        augment=True,
        augmentation_level=args.augmentation_level
    )

    # Initialize the datasets
    data_module.setup()

    # Test the augmentation
    save_dir = os.path.join(
        'outputs', 
        'augmentation_visualization', 
        f'augmentation_level_{args.augmentation_level}'
    )
    
    data_module.test_augmentation(num_samples=args.num_samples, save_dir=save_dir)
    logger.info(f"Augmentation visualizations saved to {save_dir}")
    
    # Test that the dataloaders work correctly
    logger.info("Testing dataloaders...")
    
    # Get the dataloaders
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    
    # Test a few batches
    logger.info("Checking train dataloader...")
    for i, (images, masks) in enumerate(train_loader):
        logger.info(f"Batch {i+1} - Image shape: {images.shape}, Mask shape: {masks.shape}")
        if i >= 2:  # Just check a few batches
            break
            
    logger.info("Checking validation dataloader...")
    for i, (images, masks) in enumerate(val_loader):
        logger.info(f"Batch {i+1} - Image shape: {images.shape}, Mask shape: {masks.shape}")
        if i >= 2:  # Just check a few batches
            break
    
    # Test cross-validation dataloaders
    logger.info("Testing cross-validation dataloaders...")
    train_loader, val_loader, train_size, val_size = data_module.get_cross_val_dataloaders(fold_idx=0)
    
    logger.info(f"Cross-validation - Training size: {train_size}, Validation size: {val_size}")
    
    # Check a few batches from cross-validation loaders
    logger.info("Checking cross-validation train dataloader...")
    for i, (images, masks) in enumerate(train_loader):
        logger.info(f"Batch {i+1} - Image shape: {images.shape}, Mask shape: {masks.shape}")
        if i >= 2:  # Just check a few batches
            break
            
    logger.info("Checking cross-validation validation dataloader...")
    for i, (images, masks) in enumerate(val_loader):
        logger.info(f"Batch {i+1} - Image shape: {images.shape}, Mask shape: {masks.shape}")
        if i >= 2:  # Just check a few batches
            break
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data augmentation for Cityscapes dataset")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--augmentation_level', type=str, default='standard',
                        choices=['none', 'standard', 'advanced'],
                        help='Augmentation level')
    parser.add_argument('--num_samples', type=int, default=3, 
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    main(args)
