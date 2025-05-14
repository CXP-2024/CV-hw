# Data Augmentation Guide

This document explains the data augmentation techniques implemented in the Cityscapes semantic segmentation project.

## Overview

Data augmentation is crucial for improving model generalization, especially in semantic segmentation tasks where annotating data is expensive. This project implements multi-level data augmentation techniques specifically designed for the Cityscapes dataset.

## Augmentation Levels

The augmentation framework supports three different levels:

1. **None** - No augmentation is applied, just resizing to the target size
2. **Standard** - Basic augmentations that are commonly used in segmentation tasks
3. **Advanced** - Advanced augmentations including more aggressive transformations

## Standard Augmentation Techniques

The standard level includes:

- **Random Horizontal Flipping** (50% probability)
- **Random Gaussian Blur** (30% probability) - Replaced rotation with blur for better edge preservation
- **Random Gamma Adjustment** (25% probability) - Adjusts image gamma to improve exposure variation
- **Random Upscaling** (1.0 to 1.2, 50% probability) - Only allows scaling up to prevent black borders
- **Random Cropping** (40% probability)
- **Color Jittering** (brightness, contrast, saturation adjustments, 50% probability)

## Advanced Augmentation Techniques

The advanced level includes all standard augmentations plus:

- **Random Color Channel Swapping** (10% probability) - Randomly shuffles the RGB channels to create unusual color effects
- **Random Grayscale Conversion** (10% probability) - Converts the image to grayscale to improve robustness to color variations
- **Random Gaussian Blur** (15% probability) - Applies a Gaussian blur to simulate focus issues
- **Random Sharpening** (10% probability) - Enhances edges and details in the image
- **Random Color Balance** (10% probability) - Adjusts individual RGB channels for color temperature variation
- **Enhanced Color Jittering** (15% probability) - More aggressive brightness, contrast, saturation, and hue adjustments
- **Random Noise Addition** (10% probability) - Adds Gaussian noise to simulate camera sensor variations
- **Modified Elastic Transformation** (10% probability) - Applies gentle local deformations using reflection border mode to avoid black areas
- **Color-based Image Mixup** (5% probability) - Blends the image with a color-transformed version of itself (no geometric transforms or horizontal flips)

## Usage

### In Configuration

You can specify the augmentation level in the `config.yaml` file:

```yaml
data:
  root: "data"
  image_size: [512, 1024] # height, width
  augment: true
  augmentation_level: "standard" # 'none', 'standard', or 'advanced'
  ignore_index: 255
```

### Command Line Override

You can override the configuration when running the training scripts:

```bash
# For k-fold cross-validation training
python train.py --augmentation_level advanced

# For simple training (train/val split)
python train_simple.py --augmentation_level advanced
```

## Data Augmentation Flow in Detail

### Architecture Overview

The data augmentation implementation follows a three-layer architecture:

1. **CityscapesDataModule**: Top-level data management class responsible for creating datasets and dataloaders
2. **CityscapesDataset**: Implements PyTorch's Dataset interface to handle sample loading and transformation
3. **SynchronizedTransforms**: Core augmentation class ensuring synchronized transformation of image and mask pairs

### Data Loading and Augmentation Pipeline

The augmentation happens automatically during the data loading process. Here's how the pipeline flows from initialization to actual augmentation:

1. **Command Line Argument or Config Setting** 
   ```python
   # In train.py
   augmentation_level = args.augmentation_level if args.augmentation_level else config['data'].get('augmentation_level', 'standard')
   ```

2. **DataModule Initialization**
   ```python
   # Setup data module
   data_module = CityscapesDataModule(
       data_dir=config['data']['root'],
       batch_size=config['training']['batch_size'],
       num_workers=args.num_workers,
       k_folds=k_folds, 
       image_size=tuple(config['data']['image_size']),
       augment=config['data']['augment'],
       augmentation_level=augmentation_level
   )
   ```

3. **Dataset Creation** (happens in `data_module.setup()`)
   ```python
   self.train_dataset = CityscapesDataset(
       root_dir=self.data_dir,
       split='train',
       transform=self.train_transform,
       augmentation_level=self.augmentation_level,
       image_size=self.image_size,
       is_training=True
   )
   ```

4. **DataLoader Creation** (happens in experiment initialization)
   ```python
   train_loader = DataLoader(
       dataset,
       batch_size=self.batch_size,
       sampler=train_sampler,
       num_workers=self.num_workers,
       pin_memory=True
   )
   ```

5. **Training Loop Iteration** (happens in `trainer.py`)
   ```python
   for i, (inputs, targets) in enumerate(train_loader):
       # Process batch...
   ```

6. **Behind the Scenes: `__getitem__` Method Call**
   When the training loop iterates through `train_loader`, PyTorch's DataLoader automatically calls the dataset's `__getitem__` method for each index:
   
   ```python
   # In CityscapesDataset.__getitem__
   def __getitem__(self, idx):
       # Load the image
       img_path = self.image_paths[idx]
       image = Image.open(img_path).convert('RGB')
       
       # Load the label mask
       label_path = self.label_paths[idx]
       label = Image.open(label_path)
       
       # Apply synchronized transformations (augmentations and resize)
       image, label = self.sync_transforms(image, label)
       
       # Additional processing...
       return image, label
   ```
   
7. **Augmentation via `__call__` Method**
   The `self.sync_transforms(image, label)` call invokes the `__call__` method of the `SynchronizedTransforms` class, which applies the configured augmentations

### Implicit DataLoader Call Flow

During training, the DataLoader silently manages the process:

1. DataLoader determines which samples to include in the next batch
2. For each sample index, DataLoader calls `dataset[idx]` which invokes `__getitem__(idx)`
3. `__getitem__` loads raw data and applies augmentation via `sync_transforms`
4. DataLoader collates all processed samples into a batch
5. The batch is yielded to the training loop

This design pattern separates data loading logic from the training loop, making the augmentation process transparent to the trainer. The augmentation is applied "just in time" as each sample is loaded, maximizing efficiency.

### Significance of `__call__` method

The `__call__` method is a Python "magic method" that allows an object to be called like a function. For our data augmentation system, this design choice offers several advantages:

1. **Unified Transform Interface**: Provides a clean, function-like interface for transforms.
2. **PyTorch Consistency**: Aligns with PyTorch's transform design patterns.
3. **State Persistence**: Maintains augmentation configuration across multiple calls.
4. **Synchronized Randomness**: Ensures the same random transformations are applied to both image and mask.

Example:
```python
transform = SynchronizedTransforms(
    image_size=(512, 1024),
    augmentation_level='advanced'
)
augmented_image, augmented_mask = transform(image, mask)  # Using __call__
```

## Cross-Validation and Augmentation

K-fold cross-validation is implemented in the `get_cross_val_dataloaders` method, which handles the proper application of augmentations to training folds while disabling them for validation folds.

### Cross-Validation Process

1. **Dataset Division**
   ```python
   # Create a copy of the training dataset with augmentation settings
   dataset = copy.deepcopy(self.train_dataset)
   
   # Use sklearn's KFold to split indices
   kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
   folds = list(kfold.split(indices))
   
   # Get train and validation indices for current fold
   train_indices, val_indices = folds[fold_idx]
   ```

2. **Differentiated Augmentation**
   ```python
   # For validation fold, create a separate dataset with no augmentation
   val_dataset = copy.deepcopy(self.train_dataset)
   val_dataset.augmentation_level = 'none'  # Explicitly disable augmentation
   val_dataset.is_training = False
   val_dataset.sync_transforms = SynchronizedTransforms(
       image_size=self.image_size,
       augmentation_level='none',  # Ensure 'none' level for validation
       ignore_index=255
   )
   ```

3. **Independent Dataset Copies**
   The use of `copy.deepcopy()` is critical here to ensure that:
   - The original dataset remains unchanged
   - Training and validation datasets have independent states
   - Modifications to validation dataset don't affect training dataset
   - Each cross-validation fold gets a fresh copy of the dataset

### Deep Copy vs. Shallow Copy

In the cross-validation context, deep copy (`copy.deepcopy()`) creates a completely independent copy of the dataset object, including all nested objects like the transform handler. This ensures that:

1. Modifying the augmentation level in the validation dataset doesn't affect the training dataset
2. Both datasets can exist simultaneously with different augmentation settings
3. Subsequent cross-validation folds start with clean dataset copies

If shallow copy were used instead, changes to the validation dataset's transforms would affect the original dataset, leading to incorrect augmentation application.

## Experiment Management and Augmentation

The experiment system is built around the `CrossValidationExperiment` class, which manages the lifecycle of training across multiple folds. This section explains how augmentation is controlled at the experiment level.

### Command Line Interface Control

The augmentation level can be set through the command line interface when launching experiments:

```bash
python train.py --augmentation_level advanced --model deeplabv3plus --k_folds 5
```

The augmentation level flows through the system in this order:

1. **Command Line Arguments** - Parse from CLI with `argparse`
2. **Configuration Override** - Override config settings
3. **Data Module Creation** - Pass to `CityscapesDataModule`
4. **Dataset Configuration** - Set on `CityscapesDataset`
5. **Transform Application** - Control behavior in `SynchronizedTransforms`

This can be seen in the code from `train.py`:

```python
# Get augmentation level (command line arg overrides config)
augmentation_level = args.augmentation_level if args.augmentation_level else config['data'].get('augmentation_level', 'standard')

# Setup data module
data_module = CityscapesDataModule(
    data_dir=config['data']['root'],
    batch_size=config['training']['batch_size'],
    num_workers=args.num_workers,
    k_folds=k_folds,
    image_size=tuple(config['data']['image_size']),
    augment=config['data']['augment'],
    augmentation_level=augmentation_level  # Pass through from CLI or config
)
```

### Cross-Validation Experiment Flow

During a cross-validation experiment, the following steps ensure proper augmentation management:

1. **Experiment Initialization**
   The `CrossValidationExperiment` class is initialized with the data module that contains augmentation settings.

2. **Fold Iteration**
   For each fold, the experiment calls `setup_kfold_dataloaders` to create specific train/val splits:
   
   ```python
   # Get dataloaders for this fold
   train_loader, val_loader, train_size, val_size = self.setup_kfold_dataloaders(fold, k_folds)
   ```

3. **Separate Training/Validation Transforms**
   Within the dataloader setup, the system ensures:
   - Training data uses requested augmentation level
   - Validation data uses 'none' augmentation level
   - Each fold gets fresh dataset copies

4. **Model Training with Augmentation**
   The trainer receives dataloaders with proper augmentation settings:
   
   ```python
   train_metrics, val_metrics = trainer.train(
       train_loader=train_loader,  # With augmentations
       val_loader=val_loader,      # Without augmentations
       epochs=num_epochs,
       early_stopping_patience=early_stopping_patience
   )
   ```

5. **Official Validation Evaluation**
   After training each fold, the model is evaluated on the official validation set (always without augmentation):
   
   ```python
   # Evaluate the current fold model on official validation set
   official_val_results = evaluate_model(
       model=model,
       dataloader=official_val_loader,  # No augmentation
       device=self.device,
       num_classes=num_classes
   )
   ```

### Ablation Studies and Experimentation

The system's design enables easy experimentation with different augmentation strategies:

1. **Compare Augmentation Levels**
   Multiple runs can be executed with different augmentation levels:
   
   ```bash
   python train.py --augmentation_level none --model deeplabv3plus
   python train.py --augmentation_level standard --model deeplabv3plus
   python train.py --augmentation_level advanced --model deeplabv3plus
   ```

2. **Model-Specific Augmentation**
   Different models can be tested with customized augmentation strategies:
   
   ```bash
   python train.py --model unet --augmentation_level standard
   python train.py --model deeplabv3plus --augmentation_level advanced
   ```

3. **Model Ensembling**
   The experiment system creates multiple model variants that can leverage augmentation in different ways:
   - Best single fold model
   - Average weights model combining all folds
   - Top-K folds average model

## Advanced Augmentation Implementation Details

The augmentation techniques are implemented with careful error handling to ensure robustness:

1. **Enhanced Color Transformations**
   Replaced geometric transforms that create black borders with enhanced color adjustments:
   - Brightness, contrast, saturation, and hue adjustments
   - Color channel balance modifications
   - RGB channel swapping and grayscale conversion

2. **Modified Elastic Transformation**
   Uses safer elastic deformation techniques:
   - Reduced displacement strength to minimize distortion
   - Border reflection mode instead of constant black padding
   - OpenCV's remap function with BORDER_REFLECT option
   - Error recovery to skip problematic transformations

3. **Mixed Augmentations**
   Each augmentation is applied with specific probability thresholds to maintain a balance between:
   - Data variation
   - Semantic integrity
   - Training stability

4. **Upscaling-only Approach**
   The scaling augmentation is designed to prevent black borders by:
   - Only allowing scale factors ≥ 1.0 (upscaling only)
   - Cropping oversized regions after scaling
   - Maintaining aspect ratio during transformations
   - Preserving pixel-perfect segmentation masks

5. **Black Area Prevention Strategy**
   All transformations are carefully designed to avoid black areas:
   - Removed vertical flipping (equivalent to 180-degree rotation)
   - Removed traditional perspective transformations
   - Replaced cutout/erase with noise addition
   - Used reflection border mode for elastic transformations
   - Eliminated any geometric transforms that create artificial black borders
   
6. **Preventing Confusing Scenes**
   Special attention is paid to preventing confusing augmentations:
   - Removed horizontal flipping from image mixup to avoid doubled/mirrored objects
   - Applied more subtle blending in mixup (higher alpha for original image)
   - Focused on color-space augmentations that maintain scene coherence
   - Avoided transformations that create implausible or disorienting scenes
