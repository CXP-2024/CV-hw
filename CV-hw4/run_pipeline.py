#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import subprocess
import logging
from pathlib import Path
from datetime import datetime

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_dir):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def run_command(cmd, logger):
    """Run a command and log output"""
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Log output in real-time
    for line in process.stdout:
        logger.info(line.strip())
    
    process.wait()
    if process.returncode != 0:
        logger.error(f"Command failed with return code {process.returncode}")
        return False
    
    return True

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Create pipeline directory
    pipeline_dir = os.path.join(
        config['output']['base_dir'],
        f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(pipeline_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(pipeline_dir)
    logger.info(f"Starting semantic segmentation pipeline")
    logger.info(f"Pipeline directory: {pipeline_dir}")
    
    # Install required packages
    logger.info("Installing required packages")
    required_packages = ["pyyaml", "scikit-learn", "pandas"]
    for package in required_packages:
        run_command(["pip", "install", package], logger)
    
    # Step 1: Training models
    logger.info("Step 1: Training models")
    models = args.models if args.models else ['unet', 'deeplabv3']
    model_checkpoints = {}
    
    for model in models:
        logger.info(f"Training {model}")
        train_cmd = [
            'python', 'train.py',
            '--config', args.config,
            '--model', model,
            '--device', args.device
        ]
        
        if not run_command(train_cmd, logger):
            logger.error(f"Failed to train {model}")
            continue
        
        # Find the most recent model directory
        model_dirs = sorted(Path(config['output']['base_dir']).glob(f'{model}_*'))
        if not model_dirs:
            logger.error(f"No model directory found for {model}")
            continue
        
        latest_model_dir = str(model_dirs[-1])
        checkpoint_path = os.path.join(latest_model_dir, 'models', 'best_model.pth')
        model_checkpoints[model] = checkpoint_path
    
    if not model_checkpoints:
        logger.error("No models were successfully trained")
        return
    
    # Step 2: Evaluation
    logger.info("Step 2: Evaluating models")
    for model_name, checkpoint in model_checkpoints.items():
        logger.info(f"Evaluating {model_name}")
        eval_cmd = [
            'python', 'evaluate.py',
            '--config', args.config,
            '--checkpoint', checkpoint,
            '--split', 'val',
            '--model', model_name,
            '--output_dir', os.path.join(pipeline_dir, f'evaluation_{model_name}')
        ]
        
        if not run_command(eval_cmd, logger):
            logger.error(f"Failed to evaluate {model_name}")
    
    # Step 3: Visualization
    logger.info("Step 3: Generating visualizations")
    for model_name, checkpoint in model_checkpoints.items():
        logger.info(f"Generating visualizations for {model_name}")
        viz_cmd = [
            'python', 'visualize.py',
            '--config', args.config,
            '--checkpoint', checkpoint,
            '--model', model_name,
            '--num_samples', str(args.num_samples),
            '--output_dir', os.path.join(pipeline_dir, f'visualization_{model_name}')
        ]
        
        if not run_command(viz_cmd, logger):
            logger.error(f"Failed to generate visualizations for {model_name}")
    
    # Step 4 (optional): Ablation study
    if args.run_ablation:
        logger.info("Step 4: Running ablation study")
        ablation_cmd = [
            'python', 'ablation.py',
            '--config', args.config,
            '--models', *models,
            '--device', args.device
        ]
        
        if not run_command(ablation_cmd, logger):
            logger.error("Failed to run ablation study")
    
    logger.info("Pipeline completed")
    logger.info(f"Results saved to {pipeline_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end semantic segmentation pipeline")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--models', type=str, nargs='+', choices=['unet', 'deeplabv3'], 
                       help='Models to train (default: both)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda, cpu)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for visualization')
    parser.add_argument('--run_ablation', action='store_true', help='Run ablation study')
    
    args = parser.parse_args()
    main(args)
