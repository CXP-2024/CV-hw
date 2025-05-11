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


# Import custom modules
from datasets.cityscapes import CityscapesDataset
from datasets.cityscapes import get_class_colors
from models.deeplabv3 import DeepLabV3
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
    """Load DeepLabV3 model from checkpoint"""
    model = DeepLabV3(num_classes=num_classes).to(device)
    
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
    
    # Load DeepLabV3 model
    deeplabv3_model = load_model(args.checkpoint, config['model']['n_classes'], device)
    
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
        
        # Predict with DeepLabV3
        _, prediction = predict_image(deeplabv3_model, image_path, transform, device)
        
        # Update confusion matrix
        confusion_matrix += calculate_confusion_matrix(prediction, ground_truth, num_classes)

    # Then create visualizations for the first 30 images
    print("\nCreating visualizations for the first images...")
    for i, image_path in enumerate(visualization_images):
        print(f"Creating visualization {i+1}/{len(visualization_images)}: {Path(image_path).name}")
        
        # Get ground truth
        ground_truth = get_ground_truth(image_path, val_gt_dir)
        
        # Predict with DeepLabV3
        image, prediction = predict_image(deeplabv3_model, image_path, transform, device)
        
        # Create overlay
        gt_overlay = create_color_overlay(image, ground_truth, class_colors)
        pred_overlay = create_color_overlay(image, prediction, class_colors)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('(Original Image)')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(gt_overlay)
        plt.title('(Ground Truth)')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_overlay)
        plt.title('(DeepLabV3 Prediction)')
        plt.axis('off')
        
        # Save visualization
        output_path = os.path.join(args.output_dir, f"sample_{i+1}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    # Calculate mIoU
    miou, iou_per_class = calculate_miou(confusion_matrix)
        
    # Convert iou_per_class to numpy array for further processing
    iou_per_class = np.array(iou_per_class)
    
    # Calculate class weights based on pixel frequency
    class_pixels = np.sum(confusion_matrix, axis=1)
    total_pixels = np.sum(class_pixels)
    class_weights = class_pixels / total_pixels
    
    # Calculate weighted average IoU
    # First handle possible NaN values
    valid_mask = ~np.isnan(iou_per_class)
    valid_ious = iou_per_class[valid_mask]
    valid_weights = class_weights[valid_mask]
    
    # Calculate weighted average (add small epsilon to avoid division by zero)
    epsilon = 1e-10
    weight_sum = np.sum(valid_weights) + epsilon
    weighted_miou = np.sum(valid_ious * valid_weights) / weight_sum
    
    # Print results
    print("\n===== Results =====")
    print(f"DeepLabV3 mIoU (Simple Average): {miou:.4f}")
    print(f"DeepLabV3 weighted mIoU (Weighted Average): {weighted_miou:.4f}")
    
    # Get class names and colors from cityscapes module
    from datasets.cityscapes import get_class_names, get_class_colors
    class_names = get_class_names()
    class_colors_dict = get_class_colors()
    
    print(f"Loaded {len(class_names)} class names: {class_names}")
        
    # Display class names, IDs, colors and metrics
    print("\n===== Class Information =====")
    print(f"{'ID':<5}{'Class Name':<30}{'Weight':<10}{'IoU':<10}{'Color':<15}")
    print("-" * 70)
    
    for i in range(num_classes):
        if i in class_colors_dict:
            color_str = f"RGB{class_colors_dict[i]}"
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            weight = class_weights[i] if i < len(class_weights) else 0
            iou = iou_per_class[i] if i < len(iou_per_class) else 0
            print(f"{i:<5}{class_name:<30}{weight:<10.4f}{iou:<10.4f}{color_str:<15}")
    
    # Create per-class IoU bar chart
    plt.figure(figsize=(15, 8))
    # Plot bar chart with width related to class weight
    bars = plt.bar(range(len(iou_per_class)), iou_per_class, color='skyblue', 
             width=0.6)
    
    # Add class weight as transparency (more intuitive visualization of each class importance)
    for i, bar in enumerate(bars):
        bar.set_alpha(0.3 + 0.7 * class_weights[i] / max(class_weights))
    
    plt.title('DeepLabV3 - Each Class IoU (Per-class IoU)')
    plt.ylabel('IoU')
    plt.xlabel('(Class)')
    plt.xticks(range(len(iou_per_class)), class_names, rotation=90)
    plt.grid(axis='y')
    plt.axhline(y=miou, color='r', linestyle='-', label=f'mIoU (Simple Mean): {miou:.4f}')
    plt.axhline(y=weighted_miou, color='g', linestyle='--', label=f'wmIoU (Weighted Mean): {weighted_miou:.4f}')
    plt.legend()
    
    # Save chart
    iou_chart_path = os.path.join(args.output_dir, "per_class_iou.png")
    plt.tight_layout()
    plt.savefig(iou_chart_path)
    plt.close()
    
    print(f"Per-class IoU chart saved to {iou_chart_path}")
    
    # Create additional chart: class weight vs IoU relationship
    plt.figure(figsize=(12, 8))
    # Scatter plot with point size proportional to class weight
    for i, (iou, weight) in enumerate(zip(iou_per_class, class_weights)):
        plt.scatter(i, iou, s=weight*5000, alpha=0.6, color='blue')
        if weight > 0.01:  # Only label more important classes
            plt.annotate(f"{weight:.3f}", (i, iou), 
                         textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('(Class Weight vs. IoU)')
    plt.ylabel('IoU')
    plt.xlabel('(Class)')
    plt.xticks(range(len(iou_per_class)), class_names, rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=miou, color='r', linestyle='-', label=f'Simple average: {miou:.4f}')
    plt.axhline(y=weighted_miou, color='g', linestyle='--', label=f'Weighted average: {weighted_miou:.4f}')
    plt.legend()
    
    # Save chart
    weight_chart_path = os.path.join(args.output_dir, "class_weight_vs_iou.png")
    plt.tight_layout()
    plt.savefig(weight_chart_path)
    plt.close()
    
    print(f"Class weight vs. IoU chart saved to {weight_chart_path}")
    



















    # Create a normalized confusion matrix visualization
    plt.figure(figsize=(14, 12))
    
    # Normalize confusion matrix to show percentages instead of raw counts
    norm_confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot the confusion matrix using seaborn for better visualization
    mask = np.zeros_like(norm_confusion_matrix)
    for i in range(num_classes):
        if class_pixels[i] == 0:  # Mask out classes with no pixels (to avoid division by zero warnings)
            mask[i, :] = 1
    
    # Plot confusion matrix only for the most significant classes (top 10 by pixel count)
    if num_classes > 10:
        # Get indices of top 10 classes by pixel count
        top_class_indices = np.argsort(class_pixels)[::-1][:10]
        
        # Extract the submatrix for top classes
        cm_subset = norm_confusion_matrix[np.ix_(top_class_indices, top_class_indices)]
        
        # Get corresponding class names
        top_class_names = [class_names[i] if i < len(class_names) else f"Class {i}" for i in top_class_indices]
        
        # Plot the subset confusion matrix
        sns.heatmap(cm_subset, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=top_class_names, yticklabels=top_class_names)
        plt.title('Top 10 Classes Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add note about showing only top classes
        plt.figtext(0.5, 0.01, f"Note: Only top 10 classes by pixel count are shown (out of {num_classes} total)",
                   ha='center', fontsize=10, style='italic')
    else:
        # If we have 10 or fewer classes, just show the full confusion matrix
        sns.heatmap(norm_confusion_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, 
                   mask=mask)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    # Save visualization
    conf_matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()
    
    print(f"Confusion matrix visualization saved to {conf_matrix_path}")
    
    # Create a full confusion matrix visualization (if there are many classes)
    if num_classes > 10:
        plt.figure(figsize=(20, 18))
        sns.heatmap(norm_confusion_matrix, annot=False, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, 
                   mask=mask)
        plt.title('Full Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save full visualization
        full_conf_matrix_path = os.path.join(args.output_dir, "full_confusion_matrix.png") 
        plt.tight_layout()
        plt.savefig(full_conf_matrix_path)
        plt.close()
        
        print(f"Full confusion matrix visualization saved to {full_conf_matrix_path}")
    




































    # Save results to text file
    results_path = os.path.join(args.output_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write("===== DeepLabV3测试结果 (Test Results) =====\n")
        f.write(f"DeepLabV3简单平均IoU (Simple Mean IoU): {miou:.4f}\n")
        f.write(f"DeepLabV3加权平均IoU (Weighted Mean IoU): {weighted_miou:.4f}\n")
        f.write(f"混淆矩阵可视化 (Confusion Matrix): 请查看 confusion_matrix.png 和 full_confusion_matrix.png\n")
        f.write("\n")
        f.write("类别信息 (Class Information):\n")
        f.write(f"{'ID':<5}{'名称 (Name)':<30}{'权重 (Weight)':<15}{'IoU':<10}{'颜色 (Color)':<15}\n")
        f.write("-" * 75 + "\n")
        for i in range(num_classes):
            if i in class_colors_dict:
                color_str = f"RGB{class_colors_dict[i]}"
                class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                weight = class_weights[i] if i < len(class_weights) else 0
                iou = iou_per_class[i] if i < len(iou_per_class) else 0
                f.write(f"{i:<5}{class_name:<30}{weight:<15.4f}{iou:<10.4f}{color_str:<15}\n")
        f.write("\n")
        f.write(f"测试图像总数 (Total images tested): {len(test_images)}\n")
        f.write("可视化图像 (Visualized images):\n")
        for i, path in enumerate(visualization_images):
            if i < 10:  # Only list first 10 visualized images to keep it concise
                f.write(f"{i+1}. {Path(path).name}\n")
        if len(visualization_images) > 10:
            f.write(f"...and {len(visualization_images) - 10} more images\n")
    
    print(f"Results saved to {results_path}")
    
    # Create a class legend image
    plt.figure(figsize=(15, num_classes * 0.5))
    for i in range(num_classes):
        if i in class_colors_dict:
            color = [c/255.0 for c in class_colors_dict[i]]  # Convert RGB values to 0-1 range
            plt.fill_between([0, 1], [i+0.8, i+0.8], [i+0.2, i+0.2], color=color)
            plt.text(1.1, i+0.5, f"{i}: {class_names[i]}", va='center', fontsize=10)
    
    plt.xlim(0, 5)
    plt.ylim(0, num_classes)
    plt.axis('off')
    plt.title('Class Legend - ID: Class Name')
    
    # Save legend image
    legend_path = os.path.join(args.output_dir, "class_legend.png")
    plt.tight_layout()
    plt.savefig(legend_path)
    plt.close()
    
    print(f"Class legend saved to {legend_path}")
    
    # Create a color palette visualization
    palette_size = 64  # Size of each color square
    num_cols = 3  # Number of columns in the grid
    num_rows = (num_classes + num_cols - 1) // num_cols  # Calculate number of rows
    
    # Create the image
    palette_img = np.ones((num_rows * palette_size, num_cols * palette_size * 3, 3), dtype=np.uint8) * 255
    
    for i in range(num_classes):
        if i in class_colors_dict:
            row = i // num_cols
            col = i % num_cols
            
            # Calculate position for color square
            y_start = row * palette_size
            y_end = (row + 1) * palette_size
            x_start = col * palette_size * 3
            x_end = (col * 3 + 1) * palette_size
            
            # Fill color square
            color_rgb = np.array(class_colors_dict[i], dtype=np.uint8)
            palette_img[y_start:y_end, x_start:x_end, :] = color_rgb
            
            # Add text with class ID and name
            from PIL import Image, ImageDraw, ImageFont
            img = Image.fromarray(palette_img)
            draw = ImageDraw.Draw(img)
            text_pos = (x_end + 10, y_start + palette_size // 2 - 10)
            draw.text(text_pos, f"{i}: {class_names[i]}", fill=(0, 0, 0))
            palette_img = np.array(img)
    
    # Save the palette image
    palette_path = os.path.join(args.output_dir, "class_palette.png")
    Image.fromarray(palette_img).save(palette_path)
    
    print(f"Class color palette saved to {palette_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DeepLabV3 model on random samples and calculate mIoU")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to DeepLabV3 model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/deeplabv3_test_results', help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (e.g., cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
