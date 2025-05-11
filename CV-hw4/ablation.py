#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

from datasets import CityscapesDataModule
from models.unet import UNet
from models.deeplabv3 import DeepLabV3
from utils.experiments import CrossValidationExperiment

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(model_class, model_name, dataset_module, device, config, experiment_dir):
    """
    Run an experiment with a specific model
    
    Args:
        model_class: The model class to use
        model_name: Name of the model (for logging)
        dataset_module: Dataset module with cross-validation support
        device: Device to use (cuda/cpu)
        config: Configuration for the experiment
        experiment_dir: Directory to save results
        
    Returns:
        results: Dict containing experiment results
    """
    # Setup experiment
    experiment_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup cross-validation experiment
    experiment = CrossValidationExperiment(
        model_class=model_class,
        dataset_module=dataset_module,
        experiment_name=experiment_name,
        device=device,
        base_save_dir=experiment_dir,
        config=config
    )
    
    # Run cross-validation
    start_time = time.time()
    cv_results = experiment.run_cross_validation(
        num_classes=config['model']['n_classes'],
        num_epochs=config['training']['epochs'],
        optimizer_params={
            'lr': config['training']['lr'],
            'weight_decay': config['training']['weight_decay']
        }
    )
    training_time = time.time() - start_time
    
    # Calculate average metrics
    avg_miou = np.mean(cv_results['all_mious'])
    
    results = {
        'model_name': model_name,
        'experiment_dir': os.path.join(experiment_dir, experiment_name),
        'avg_miou': avg_miou,
        'training_time': training_time,
        'cv_results': cv_results
    }
    
    return results

def compare_models(results, save_path=None):
    """
    Compare models based on experiment results
    
    Args:
        results: List of experiment result dicts
        save_path: Path to save comparison figure
    """
    model_names = [r['model_name'] for r in results]
    mious = [r['avg_miou'] for r in results]
    times = [r['training_time'] / 3600 for r in results]  # Convert to hours
    
    # Create figure
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Plot Mean IoU
    x = np.arange(len(model_names))
    width = 0.35
    rects1 = ax1.bar(x - width/2, mious, width, label='Mean IoU', color='skyblue')
    ax1.set_ylabel('Mean IoU')
    ax1.set_ylim(0, 1)
    
    # Add a second y-axis for training time
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, times, width, label='Training Time (hours)', color='salmon')
    ax2.set_ylabel('Training Time (hours)')
    
    # Add labels and title
    ax1.set_xlabel('Model')
    ax1.set_title('Model Comparison: Mean IoU vs Training Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              fancybox=True, shadow=True, ncol=2)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Print comparison table
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Mean IoU': mious,
        'Training Time (hours)': times
    })
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Return comparison dataframe
    return comparison_df

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup experiment directory
    ablation_dir = os.path.join(config['output']['base_dir'], 'ablation_studies')
    os.makedirs(ablation_dir, exist_ok=True)
    
    # Setup data module
    data_module = CityscapesDataModule(
        data_dir=config['data']['root'],
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers,
        k_folds=config['training']['k_folds'],
        image_size=tuple(config['data']['image_size']),
        augment=config['data']['augment']
    )
    
    # List of experiments to run
    experiments = []
    results = []
    
    # Add UNet experiment
    if 'unet' in args.models:
        experiments.append({
            'name': 'UNet',
            'class': UNet
        })
    
    # Add DeepLabV3 experiment
    if 'deeplabv3' in args.models:
        experiments.append({
            'name': 'DeepLabV3',
            'class': DeepLabV3
        })
    
    # Run experiments
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Running experiment with {exp['name']}")
        print(f"{'='*50}\n")
        
        # Run experiment
        result = run_experiment(
            model_class=exp['class'],
            model_name=exp['name'],
            dataset_module=data_module,
            device=device,
            config=config,
            experiment_dir=ablation_dir
        )
        
        results.append(result)
    
    # Compare models
    comparison_df = compare_models(results, save_path=os.path.join(ablation_dir, 'model_comparison.png'))
    
    # Save comparison results
    comparison_df.to_csv(os.path.join(ablation_dir, 'model_comparison.csv'), index=False)
    
    print(f"\nAblation study complete. Results saved to {ablation_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation studies for semantic segmentation models")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--models', type=str, nargs='+', default=['unet', 'deeplabv3'], 
                       choices=['unet', 'deeplabv3'], help='Models to include in ablation study')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (e.g., cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)
