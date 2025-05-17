# Semantic Segmentation on Cityscapes Dataset

This project implements various semantic segmentation models to segment urban street scenes from the Cityscapes dataset. The implemented models include UNet, DeepLabV3, and DeepLabV3+, with comprehensive training pipelines, data augmentation strategies, and evaluation tools. 

## See detailed report in [report.pdf](report/report.pdf).

## Project Structure

```
CV-hw4/
├── models/                # Model architectures
│   ├── unet.py            # UNet model implementation
│   ├── deeplabv3.py       # DeepLabV3 model implementation  
│   └── deeplabv3plus.py   # DeepLabV3+ model implementation
├── datasets/              # Dataset handling
│   └── cityscapes.py      # Cityscapes dataset loader with augmentations
├── utils/                 # Utility functions
│   ├── trainer.py         # Model training and evaluation
│   ├── experiment.py      # Experiment setup and cross-validation
│   ├── losses.py          # Loss functions
│   ├── simple_experiment.py # Simplified experiment setup
│   └── visualization.py   # Visualization tools
├── outputs/               # Model outputs, logs and checkpoints
├── data/                  # Cityscapes dataset directory
│   ├── gtFine/            # Ground truth annotations
│   └── leftImg8bit/       # Input images
├── config.yaml            # Configuration file
├── train.py               # Main training script
├── train_simple.py        # Simplified training script
├── test_*.py              # Various test scripts
├── requirements.txt       # Project dependencies
├── check_setup.py         # Check environment setup
├── AUGMENTATION.md        # Detailed augmentation documentation
├── EVALUATION.md          # Detailed evaluation documentation
└── MODEL.md               # Detailed deeplabv3plus model documentation
```

## Models

The project implements three semantic segmentation architectures:

1. **UNet**: A classic encoder-decoder architecture with skip connections.
2. **DeepLabV3**: Implementation of the DeepLabV3 model with atrous spatial pyramid pooling.
3. **DeepLabV3+**: An enhanced version of DeepLabV3 with better decoder module and low-level feature integration.

Finally, the project use DeepLabV3+ as the main model for training and evaluation. The model is implemented in [deeplabv3plus.py](models/deeplabv3plus.py), see detailed information in [MODEL.md](MODEL.md).

## Dataset

The project uses the Cityscapes dataset, which contains urban street scenes from 50 different cities with pixel-level annotations. The dataset is organized into:

- `leftImg8bit/`: Input images
- `gtFine/`: Fine annotations with semantic segmentation labels

The dataset is split into train, validation, and test sets.

## Features

- **Cross-validation**: K-fold cross-validation for robust model evaluation.
- **Advanced Augmentations**: Multiple levels of data augmentation (none, standard, advanced).
- **Custom Loss Functions**: Implementation of various loss functions including cross-entropy, dice loss, and combined losses.
- **Visualization Tools**: Comprehensive visualization of model predictions, confusion matrices, and class performance.
- **Learning Rate Schedulers**: Different learning rate scheduling strategies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CXP-2024/CV-hw.git
cd CV-hw/CV-hw4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Cityscapes dataset from [the official website](https://www.cityscapes-dataset.com/) and extract it to the `data/` directory.

## Usage

### Configuration

Edit `config.yaml` to configure your experiment:

```yaml
device: "cuda"
model:
  type: "deeplabv3plus" # unet or deeplabv3
  n_classes: 19
  backbone: "resnet"
data:
  root: "data"
  image_size: [512, 1024] # height, width
  augment: true
  augmentation_level: "standard" # 'none', 'standard', or 'advanced'
  ignore_index: 255
training:
  epochs: 20
  lr: 5e-5  # 降低学习率
  weight_decay: 1e-3  # 增加权重衰减
  batch_size: 4
  k_folds: 5
  early_stopping_patience: 6  # 增加早停耐心值
  lr_schedule: "default"  # warmrestart, cosine, or default
  lr_warmrestart_T0: 5  # 第一次重启的周期
  lr_warmrestart_T_mult: 2  # 每次重启后周期乘数
  loss: "combined"  # 使用组合损失函数
output:
  base_dir: "outputs"
  save_checkpoints: true
  save_best_only: false

```

### Training

Run the training script:

```bash
python train.py --config config.yaml
```

For a simplified training without cross-validation:

```bash
python train_simple.py --config config.yaml
```


## Data Augmentation

The project implements different levels of data augmentation:

1. **None**: Only resizing to target dimensions.
2. **Standard**: Basic augmentations like horizontal flipping, color jittering, and random cropping.
3. **Advanced**: Enhanced augmentations including channel swapping, elastic transformations, noise addition, and more.

For detailed information on augmentation techniques, see [AUGMENTATION.md](AUGMENTATION.md).

## Results

Training results, model checkpoints, and visualizations are saved in the `outputs/` directory, organized by experiment name. Each experiment directory contains:

- `models/`: Saved model checkpoints
- `logs/`: Training logs
- `figures/`: Visualizations including training curves, sample predictions, and confusion matrices
- `config.json`: Configuration used for the experiment

## Testing and Visualization

To test a trained model:

```bash
python test_deeplabv3plus.py --checkpoint <your deeplabv3plus model path> # in resolution 512x1024
# or test the model in original resolution:
python test_deeplabv3plus_origin_resolution.py --checkpoint <your deeplabv3plus model path>
# in resolution 1024x2048, will only loss about 0.06% mIoU, this still inference in 512x1024 resolution for better performance but upsample to 1024x2048 resolution for evaluation
```
For the detailed evaluation, see [EVALUATION.md](EVALUATION.md).

## License

This project is based on the Cityscapes dataset, which has its own licensing terms. Please refer to the [Cityscapes website](https://www.cityscapes-dataset.com/) for more information.

## Acknowledgements

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [PyTorch](https://pytorch.org/)
- [DeepLab Paper](https://arxiv.org/abs/1706.05587)
- [UNet Paper](https://arxiv.org/abs/1505.04597)
- [Copilot for coding assistance](https://github.com/features/copilot)
