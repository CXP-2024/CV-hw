#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from torchvision import transforms
import glob
from pathlib import Path
import seaborn as sns
import pandas as pd


# Import custom modules
from datasets.cityscapes import CityscapesDataset
from datasets.cityscapes import get_class_colors
from models.deeplabv3plus import DeepLabV3Plus
from utils.visualization import calculate_miou


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_visualization_transform():
    """Get transform for visualization"""
    return transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path, num_classes, device):
    """Load DeepLabV3Plus model from checkpoint"""
    model = DeepLabV3Plus(num_classes=num_classes).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_image(model, image_path, transform, device):
    """Make prediction for a single image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize image to the same size used in transform (important for visualization)
    resized_image = image.resize((1024, 512), Image.BILINEAR)
    
    # Apply transform for model input
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    return resized_image, pred


def get_ground_truth(image_path, city_dir):
    """Get ground truth for an image"""
    # Extract city and file name
    parts = Path(image_path).parts
    city = parts[-2]
    file_name = parts[-1].replace('_leftImg8bit.png', '_gtFine_labelIds.png')
    
    # Construct the label path
    label_path = os.path.join(city_dir, city, file_name)
    
    # Load label
    label = Image.open(label_path)
    
    # Resize label to match the size used for prediction (512, 1024)
    label = label.resize((1024, 512), Image.NEAREST)
    
    # Convert to numpy
    label_np = np.array(label, dtype=np.int64)
    
    # Map IDs to train IDs (reusing logic from cityscapes.py)
    from datasets.cityscapes import id_to_trainid
    for k, v in id_to_trainid.items():
        label_np[label_np == k] = v
    
    # Ignore regions with trainId 255 or -1
    label_np[label_np == 255] = 255  # Keep ignore regions as 255
    label_np[label_np == -1] = 255   # Map -1 to 255 for ignore
    
    return label_np


def create_color_overlay(image, segmentation, class_colors):
    """Create a color overlay of segmentation on top of image"""
    # Create RGB image from segmentation
    seg_rgb = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label_id, color in class_colors.items():
        mask = (segmentation == label_id)
        seg_rgb[mask] = color
    
    # Convert PIL image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Blend the two images
    alpha = 0.5
    overlay = image * (1 - alpha) + seg_rgb * alpha
    overlay = overlay.astype(np.uint8)
    
    return overlay


def calculate_confusion_matrix(pred, gt, num_classes, ignore_index=255):
    """Calculate confusion matrix between prediction and ground truth"""
    mask = (gt != ignore_index)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for t, p in zip(gt[mask].flatten(), pred[mask].flatten()):
        confusion_matrix[t, p] += 1
    
    return confusion_matrix


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get class colors from the cityscapes module
    from datasets.cityscapes import get_class_colors
    class_colors = get_class_colors()
    
    # Get transform
    transform = get_visualization_transform()
    
    # Load DeepLabV3Plus model
    deeplabv3plus_model = load_model(args.checkpoint, config['model']['n_classes'], device)
    
    # Get all image paths from validation set
    val_image_dir = os.path.join(config['data']['root'], 'leftImg8bit', 'val')
    val_gt_dir = os.path.join(config['data']['root'], 'gtFine', 'val')
    all_image_paths = glob.glob(os.path.join(val_image_dir, '*', '*_leftImg8bit.png'))
    
    # Sort images by path for reproducibility
    all_image_paths.sort()
    
    # Use all images for testing (for metrics calculation)
    test_images = all_image_paths
    
    # Select images to visualize (first 30 images)
    if len(all_image_paths) > 30:
        visualization_images = all_image_paths[:30]
    else:
        visualization_images = all_image_paths
    
    print(f"Using all {len(test_images)} validation images for testing")
    print(f"Will visualize the first {len(visualization_images)} images")
    
    # Create confusion matrix for mIoU calculation
    num_classes = config['model']['n_classes']
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # First process all images to calculate mIoU
    print("Processing all images for mIoU calculation...")
    import tqdm
    for image_path in tqdm.tqdm(test_images, desc='Processing images'):
        
        # Get ground truth
        ground_truth = get_ground_truth(image_path, val_gt_dir)
        
        # Predict with DeepLabV3Plus
        _, prediction = predict_image(deeplabv3plus_model, image_path, transform, device)
        
        # Update confusion matrix
        confusion_matrix += calculate_confusion_matrix(prediction, ground_truth, num_classes)
    
    # Calculate mIoU
    miou, class_iou, class_weights, iou_df = calculate_miou(confusion_matrix)
    
    # Print results
    print(f"Overall mIoU: {miou:.4f}")
    
    # Save results to file
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Overall mIoU: {miou:.4f}\n\n")
        f.write("Per-class IoU:\n")
        for i, iou in enumerate(class_iou):
            name = class_iou.index[i] if hasattr(class_iou, 'index') else f"Class {i}"
            f.write(f"{name}: {iou:.4f}\n")
    
    # Create and save per-class IoU plot
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x=class_iou.index if hasattr(class_iou, 'index') else range(len(class_iou)), 
                     y=class_iou.values)
    plt.title('IoU per Class')
    plt.xlabel('Class')
    plt.ylabel('IoU')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'per_class_iou.png'))
    
    # Create a figure to show class weights vs IoU
    plt.figure(figsize=(15, 8))
    if hasattr(class_iou, 'index'):
        # Create a DataFrame for visualization
        viz_df = pd.DataFrame({
            'class': class_iou.index,
            'iou': class_iou.values,
            'weight': class_weights
        })
        
        # Sort by class weight
        viz_df = viz_df.sort_values('weight', ascending=False)
        
        # Create bar plot
        ax = sns.barplot(x='class', y='weight', data=viz_df, color='blue', alpha=0.5, label='Weight')
        ax2 = ax.twinx()
        sns.barplot(x='class', y='iou', data=viz_df, color='red', alpha=0.5, label='IoU', ax=ax2)
    else:
        # Simple plot if we don't have class names
        ax = plt.subplot(111)
        ax.bar(range(len(class_weights)), class_weights, alpha=0.5, label='Weight', color='blue')
        ax2 = ax.twinx()
        ax2.bar(range(len(class_iou)), class_iou, alpha=0.5, label='IoU', color='red')
    
    ax.set_ylabel('Class Weight')
    ax2.set_ylabel('IoU')
    ax.set_title('Class Weight vs IoU')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'class_weight_vs_iou.png'))
    
    # Create and save confusion matrix visualization
    plt.figure(figsize=(15, 15))
    # Normalize confusion matrix
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    norm_confusion_matrix = confusion_matrix.astype('float') / (row_sums + 1e-10)
    
    sns.heatmap(norm_confusion_matrix, cmap='Blues', annot=False)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Also save the full confusion matrix without normalization
    plt.figure(figsize=(15, 15))
    sns.heatmap(confusion_matrix, cmap='Blues', annot=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(args.output_dir, 'full_confusion_matrix.png'))
    
    # Now generate and save visualizations
    print("Generating visualizations...")
    
    # Visualization of class colors
    from datasets.cityscapes import get_class_names
    class_names = get_class_names()
    
    # Create a color legend
    plt.figure(figsize=(15, 8))
    color_patches = [plt.Rectangle((0, 0), 1, 1, fc=np.array(class_colors[i])/255) for i in range(len(class_names))]
    plt.legend(color_patches, class_names, loc='center', ncol=3)
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'class_legend.png'))
    plt.close()
    
    # Create a color palette image
    palette_img = np.zeros((100, len(class_names)*30, 3), dtype=np.uint8)
    for i, color in enumerate(class_colors.values()):
        palette_img[:, i*30:(i+1)*30, :] = color
    plt.figure(figsize=(15, 2))
    plt.imshow(palette_img)
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'class_palette.png'))
    plt.close()
    
    # Process and save visualization images
    for i, image_path in enumerate(visualization_images):
        # Get ground truth
        ground_truth = get_ground_truth(image_path, val_gt_dir)
        
        # Predict with DeepLabV3Plus
        original_image, prediction = predict_image(deeplabv3plus_model, image_path, transform, device)
        
        # Create figure with 3 subplots (original, ground truth, prediction)
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 3, 2)
        gt_color = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
        for label_id, color in class_colors.items():
            mask = (ground_truth == label_id)
            gt_color[mask] = color
        plt.imshow(gt_color)
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Prediction
        plt.subplot(1, 3, 3)
        pred_color = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for label_id, color in class_colors.items():
            mask = (prediction == label_id)
            pred_color[mask] = color
        plt.imshow(pred_color)
        plt.title('DeepLabV3Plus Prediction')
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'sample_{i+1}.png'))
        plt.close()
    
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test DeepLabV3Plus model on Cityscapes dataset")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/deeplabv3plus_test_results', help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    main(args)
