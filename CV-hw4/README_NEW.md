# Semantic Segmentation with Cityscapes Dataset

This repository contains the implementation of semantic segmentation models for the Cityscapes dataset. The system supports multiple model architectures (UNet and DeepLabV3), cross-validation, visualization, and ablation studies.

## Dataset

This implementation uses the Cityscapes dataset with:
- `leftImg8bit` folder: Contains the input RGB images
- `gtFine` folder: Contains pixel-level annotations (semantic labels)

The dataset is organized into train, val, and test splits.

## Project Structure

```
.
├── config.yaml                   # Configuration file
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── visualize.py                  # Visualization script
├── ablation.py                   # Ablation study script
├── run_pipeline.py               # End-to-end pipeline script
├── models/
│   ├── unet.py                   # UNet model implementation
│   └── deeplabv3.py              # DeepLabV3 model implementation
├── datasets/
│   └── cityscapes.py             # Cityscapes dataset loader
├── utils/
│   ├── trainer.py                # Training utilities
│   ├── visualization.py          # Visualization utilities
│   └── experiments.py            # Cross-validation experiment utilities
├── outputs/                      # Output directory for results
│   ├── models/                   # Saved models
│   ├── logs/                     # Training logs
│   └── figures/                  # Generated visualizations
└── data/                         # Dataset directory
    ├── gtFine/                   # Ground truth annotations
    └── leftImg8bit/              # Input images
```

## Models

### UNet
UNet is a convolutional network architecture for fast and precise segmentation of images. It consists of a contracting path and an expansive path, creating a U-shaped architecture:
- **Encoder**: Captures context through a series of convolutional and max-pooling operations
- **Decoder**: Enables precise localization through transposed convolutions
- **Skip Connections**: Combines high-resolution features from the contracting path with upsampled features

### DeepLabV3
DeepLabV3 is a state-of-the-art semantic segmentation model that incorporates:
- **ResNet Backbone**: For feature extraction
- **Atrous Spatial Pyramid Pooling (ASPP)**: To capture multi-scale context
- **Atrous Convolution**: To explicitly control the resolution of features

## Features

1. **Cross-Validation Framework**: Implements k-fold cross-validation for robust evaluation
2. **MIoU Evaluation**: Uses Mean Intersection over Union as the primary evaluation metric
3. **Comprehensive Visualization**: Tools for visualizing:
   - Class-wise performance metrics
   - Confusion matrices
   - Segmentation overlays
4. **Ablation Studies**: Compare different architectural choices and hyperparameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-segmentation.git
cd semantic-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the Cityscapes dataset:
   - Download the dataset from https://www.cityscapes-dataset.com/
   - Extract `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` to the `data` directory

## Usage

### Configuration

All parameters can be configured in `config.yaml`:
```yaml
model:
  type: "unet"  # unet or deeplabv3
  n_classes: 19
data:
  root: "data"
  image_size: [512, 1024]  # height, width
  augment: true
training:
  epochs: 50
  lr: 1e-3
  batch_size: 4
  k_folds: 5
```

### Training

To train a model with cross-validation:

```bash
python train.py --model unet --config config.yaml
```

Arguments:
- `--model`: Model type (unet or deeplabv3)
- `--device`: Device to use (cuda or cpu)
- `--num_workers`: Number of data loading workers
- `--seed`: Random seed for reproducibility

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint outputs/unet_YYYYMMDD_HHMMSS/models/best_model.pth --split val
```

Arguments:
- `--checkpoint`: Path to model checkpoint
- `--split`: Dataset split to evaluate on (train, val, test)
- `--batch_size`: Batch size for evaluation
- `--output_dir`: Directory to save evaluation results

### Visualization

To visualize segmentation results:

```bash
python visualize.py --checkpoint outputs/unet_YYYYMMDD_HHMMSS/models/best_model.pth --num_samples 5
```

Arguments:
- `--checkpoint`: Path to model checkpoint
- `--num_samples`: Number of random samples to visualize
- `--indices`: Specific indices to visualize
- `--output_dir`: Directory to save visualizations

### Ablation Studies

To run ablation studies comparing different models:

```bash
python ablation.py --models unet deeplabv3
```

Arguments:
- `--models`: Models to include in the ablation study
- `--device`: Device to use

### End-to-End Pipeline

To run the complete pipeline (training, evaluation, visualization, and ablation studies):

```bash
python run_pipeline.py --models unet deeplabv3 --run_ablation
```

Arguments:
- `--models`: Models to train and evaluate
- `--device`: Device to use
- `--num_samples`: Number of samples for visualization
- `--run_ablation`: Flag to run ablation study

## Performance Metrics

The system uses the following metrics:
- **Mean IoU (mIoU)**: Primary metric for segmentation quality
- **Class-wise IoU**: IoU for each class
- **Confusion Matrix**: To analyze model's class-wise performance

## Results

Results are saved in the `outputs` directory:
- `models/`: Saved model checkpoints
- `logs/`: Training and evaluation logs
- `figures/`: Visualizations and result plots

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
│   ├── logs/                     # Training logs
│   └── figures/                  # Visualizations
├── data/
│   ├── gtFine/                   # Ground truth annotations
│   └── leftImg8bit/              # Input images
└── README.md                     # Project documentation
```

## Models

Two semantic segmentation models are implemented:

1. **UNet**: A classic encoder-decoder architecture with skip connections.
2. **DeepLabV3**: A modern segmentation model with atrous convolutions and ASPP module.

## Requirements

The required packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a model with cross-validation:

```bash
python train.py --model unet --config config.yaml
```

Options:
- `--model`: Model to train (`unet` or `deeplabv3`)
- `--config`: Path to configuration file
- `--device`: Device to use (`cuda` or `cpu`)
- `--num_workers`: Number of data loading workers
- `--seed`: Random seed for reproducibility

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint outputs/unet_20250510_123456/models/best_model.pth --split val
```

Options:
- `--checkpoint`: Path to model checkpoint or experiment directory
- `--split`: Dataset split to evaluate on (`train`, `val`, or `test`)
- `--output_dir`: Output directory for evaluation results
- `--num_samples`: Number of samples to visualize

### Visualization

To visualize predictions on images:

```bash
python visualize.py --checkpoint outputs/unet_20250510_123456/models/best_model.pth --input data/leftImg8bit/val/frankfurt
```

Options:
- `--checkpoint`: Path to model checkpoint
- `--input`: Path to input image or directory
- `--output_dir`: Output directory for visualizations
- `--model_type`: Model type (`unet` or `deeplabv3`)
- `--random_samples`: Number of random samples to process when input is a directory

## Configuration

The `config.yaml` file contains various settings for the models, training, and data:

```yaml
model:
  type: "unet"  # unet or deeplabv3
  n_classes: 19
  backbone: "resnet"
data:
  root: "data"
  image_size: [512, 1024]  # height, width
  augment: true
training:
  epochs: 50
  lr: 1e-3
  weight_decay: 1e-4
  batch_size: 4
  k_folds: 5
output:
  base_dir: "outputs"
```

## Results

Training results, including learning curves and evaluation metrics, are saved to the `outputs` directory. For each experiment, the following will be generated:
- Training and validation loss/mIoU curves
- Per-class IoU scores
- Confusion matrix
- Visualization of segmentation predictions
- Model checkpoints

## Cross-Validation

As required, the implementation includes k-fold cross-validation to ensure robust evaluation. The number of folds can be specified in the configuration file.

## Metrics

The main evaluation metric is Mean Intersection over Union (MIoU), calculated as the average IoU across all classes.

## Visualization

Segmentation results can be visualized using the `visualize.py` script, which creates overlays of the predictions on the input images.

## Ablation Studies

For ablation studies, different model configurations can be tested by modifying the config file and training multiple variants. The results can be compared using the evaluation metrics and visualizations.
