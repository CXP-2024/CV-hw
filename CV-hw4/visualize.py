#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import random
import glob

from datasets import CityscapesDataset, decode_segmap, get_class_colors
from models.unet import UNet
from models.deeplabv3 import DeepLabV3
from torchvision import transforms

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_visualization_transform():
    """Get transform for visualization"""
    return transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_single_image(model, image_path, transform, device):
    """Make a prediction on a single image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transform
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    return image, pred

def create_color_overlay(image, prediction, class_colors):
    """Create a color overlay of the prediction on the image"""
    # Convert PIL image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create RGB segmentation map
    seg_map = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        mask = prediction == class_id
        seg_map[mask] = color
    
    # Convert to BGR for OpenCV
    seg_map_bgr = cv2.cvtColor(seg_map, cv2.COLOR_RGB2BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create overlay
    overlay = cv2.addWeighted(image_bgr, 0.7, seg_map_bgr, 0.3, 0)
    
    # Convert back to RGB
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay
    # Convert prediction to colored mask
    colored_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        if class_id < 255:  # Ignore void class
            mask = (prediction == class_id)
            colored_mask[mask] = color
    
    # Resize image to match prediction size
    image_resized = np.array(image.resize((prediction.shape[1], prediction.shape[0]), Image.BILINEAR))
    
    # Create overlay
    overlay = cv2.addWeighted(image_resized, 0.7, colored_mask, 0.3, 0)
    
    return overlay

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
      # Get class colors
    cityscapes = CityscapesDataset(root_dir=config['data']['root'], split='val')
    class_colors = cityscapes.get_class_colors()
    
    # Get transform
    transform = get_visualization_transform()
    
    # Load model
    if args.model_type == 'unet':
        model = UNet(num_classes=config['model']['n_classes'])
    elif args.model_type == 'deeplabv3':
        model = DeepLabV3(num_classes=config['model']['n_classes'])
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process images
    if os.path.isdir(args.input):
        # Get all image files in the directory
        image_paths = glob.glob(os.path.join(args.input, '*.png')) + \
                      glob.glob(os.path.join(args.input, '*.jpg')) + \
                      glob.glob(os.path.join(args.input, '*.jpeg'))
        
        if args.random_samples > 0 and args.random_samples < len(image_paths):
            image_paths = random.sample(image_paths, args.random_samples)
    else:
        # Single image
        image_paths = [args.input]
    
    print(f"Processing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        # Make prediction
        image, prediction = predict_single_image(model, image_path, transform, device)
        
        # Create visualization
        overlay = create_color_overlay(image, prediction, class_colors)
        
        # Save visualization
        output_path = os.path.join(args.output_dir, f"{Path(image_path).stem}_segmented.png")
        plt.figure(figsize=(15, 10))
        
        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Display segmentation overlay
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title('Segmentation Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Processed {i+1}/{len(image_paths)}: {Path(image_path).name} -> {Path(output_path).name}")
    
    print(f"\nVisualizations saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize semantic segmentation predictions")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations', help='Output directory for visualizations')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'deeplabv3'], help='Model type')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (e.g., cuda, cpu)')
    parser.add_argument('--random_samples', type=int, default=0, 
                        help='Number of random samples to process when input is a directory (0 = all)')
    
    args = parser.parse_args()
    main(args)
