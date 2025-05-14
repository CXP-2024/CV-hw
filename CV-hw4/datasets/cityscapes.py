#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import glob
import random
from collections import namedtuple
from sklearn.model_selection import KFold
import logging
import copy
import cv2
from scipy.ndimage import gaussian_filter

# Import the cityscapes labels
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cityscapesScripts'))
from cityscapesscripts.helpers.labels import labels as cs_labels

# Define the Cityscapes Class Information
Label = namedtuple('Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])

# Define a logger for debugging
logger = logging.getLogger(__name__)

class SynchronizedTransforms:
    """Class to apply synchronized transforms to both image and mask"""
    
    def __init__(self, image_size=(512, 1024), augmentation_level='none', ignore_index=255):
        """
        Args:
            image_size (tuple): Target image size (height, width)
            augmentation_level (str): Level of augmentation to apply ('none', 'standard', 'advanced')
            ignore_index (int): Index to use for padding in the segmentation mask
        """
        self.image_size = image_size
        self.augmentation_level = augmentation_level
        self.ignore_index = ignore_index
    
    def __call__(self, image, mask):
        """
        Apply synchronized transforms to both image and mask
        
        Args:
            image (PIL.Image): The input image
            mask (PIL.Image): The segmentation mask
            
        Returns:
            tuple: (transformed_image, transformed_mask)
        """
        # First, ensure both inputs are PIL images
        if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
            raise TypeError("Both image and mask should be PIL Image objects")
            
        # Always resize to ensure images match required size
        if image.size != (self.image_size[1], self.image_size[0]):  # PIL uses (width, height)
            image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
            mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)
            
        # If no augmentation is requested, return early
        if self.augmentation_level == 'none':
            return image, mask
            
        # Apply standard augmentations
        if self.augmentation_level in ['standard', 'advanced']:
            #print("Applying standard augmentations...")
            # Random horizontal flipping (50% probability)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random brightness, contrast, and saturation (60% probability)
            # Only applied to image, not mask
            if random.random() > 0.4:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                saturation_factor = random.uniform(0.8, 1.2)
                
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)
                image = TF.adjust_saturation(image, saturation_factor)
            
            # Random rotation (small angles, -10 to 10 degrees, 30% probability)
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST, fill=self.ignore_index)
            
            # Random scaling (0.8 to 1.2, 50% probability)
            if random.random() > 0.5:
                scale_factor = random.uniform(0.8, 1.2)
                new_height = int(self.image_size[0] * scale_factor)
                new_width = int(self.image_size[1] * scale_factor)
                
                image = TF.resize(image, (new_height, new_width), interpolation=Image.BILINEAR)
                mask = TF.resize(mask, (new_height, new_width), interpolation=Image.NEAREST)
                
                # If scaled image is smaller than target size, we need to pad
                if new_height < self.image_size[0] or new_width < self.image_size[1]:
                    # Calculate padding
                    padding_height = max(0, self.image_size[0] - new_height)
                    padding_width = max(0, self.image_size[1] - new_width)
                    
                    padding_top = padding_height // 2
                    padding_bottom = padding_height - padding_top
                    padding_left = padding_width // 2
                    padding_right = padding_width - padding_left
                    
                    # Pad with zeros for image, ignore_index for mask
                    padding = (padding_left, padding_top, padding_right, padding_bottom)
                    image = TF.pad(image, padding, fill=0)
                    mask = TF.pad(mask, padding, fill=self.ignore_index)
                
                # If scaled image is larger than target size, we need to crop
                elif new_height > self.image_size[0] or new_width > self.image_size[1]:
                    # Calculate crop
                    i = (new_height - self.image_size[0]) // 2 if new_height > self.image_size[0] else 0
                    j = (new_width - self.image_size[1]) // 2 if new_width > self.image_size[1] else 0
                    h, w = self.image_size
                    
                    # Crop both image and mask
                    image = TF.crop(image, i, j, h, w)
                    mask = TF.crop(mask, i, j, h, w)
        
        # Advanced augmentations
        if self.augmentation_level == 'advanced':
            # Random vertical flipping (10% probability) - careful with this for street scenes
            if random.random() > 0.9:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random perspective transformation (15% probability)
            if random.random() > 0.85:
                try:
                    # Get dimensions using the size property
                    width, height = image.size
                    
                    # Define perspective parameters - using lists as required by TF.perspective
                    startpoints = [
                        [0, 0],  # top-left
                        [width - 1, 0],  # top-right
                        [width - 1, height - 1],  # bottom-right
                        [0, height - 1],  # bottom-left
                    ]
                    
                    # Perturb the corner points slightly (within 5% of image dimensions)
                    width_offset = width * 0.05
                    height_offset = height * 0.05
                    
                    # Generate endpoints with randomized offsets using uniform distribution for smoother results
                    endpoints = [
                        [startpoints[0][0] + random.uniform(-width_offset, width_offset),
                         startpoints[0][1] + random.uniform(-height_offset, height_offset)],
                        [startpoints[1][0] + random.uniform(-width_offset, width_offset),
                         startpoints[1][1] + random.uniform(-height_offset, height_offset)],
                        [startpoints[2][0] + random.uniform(-width_offset, width_offset),
                         startpoints[2][1] + random.uniform(-height_offset, height_offset)],
                        [startpoints[3][0] + random.uniform(-width_offset, width_offset),
                         startpoints[3][1] + random.uniform(-height_offset, height_offset)],
                    ]
                    
                    image = TF.perspective(image, startpoints, endpoints, fill=0)
                    mask = TF.perspective(mask, startpoints, endpoints, fill=self.ignore_index, interpolation=Image.NEAREST)
                except Exception as e:
                    logger.warning(f"透视变换失败，跳过: {e}")
            
            # Random color channel swapping (10% probability)
            if random.random() > 0.9:
                img_np = np.array(image)
                # Randomly swap color channels
                channels = list(range(3))
                random.shuffle(channels)
                img_np = img_np[..., channels]
                image = Image.fromarray(img_np)
            
            # Random grayscale conversion (10% probability)
            if random.random() > 0.9:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)
            
            # Random Gaussian blur (20% probability)
            if random.random() > 0.8:
                # Use a smaller kernel for smaller images
                kernel_size = int(min(self.image_size) * 0.03) | 1  # Make sure it's odd
                kernel_size = max(3, kernel_size)  # At least size 3
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
            
            # Random cutout/erase (10% probability)
            if random.random() > 0.9:
                # Create a cutout with random size (up to 20% of image)
                width, height = image.size
                cutout_size_x = int(width * random.uniform(0.05, 0.2))
                cutout_size_y = int(height * random.uniform(0.05, 0.2))
                
                # Random position
                x = random.randint(0, width - cutout_size_x - 1)
                y = random.randint(0, height - cutout_size_y - 1)
                
                # Apply cutout (black rectangle) to image
                img_array = np.array(image)
                img_array[y:y+cutout_size_y, x:x+cutout_size_x, :] = 0
                image = Image.fromarray(img_array)
                
                # For mask, fill with ignore_index
                mask_array = np.array(mask)
                mask_array[y:y+cutout_size_y, x:x+cutout_size_x] = self.ignore_index
                mask = Image.fromarray(mask_array.astype(np.uint8))
            
            # Random elastic transformation (10% probability)
            if random.random() > 0.9:
                try:
                    # Create random displacement fields
                    width, height = image.size
                    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
                    
                    # Create random displacement with gaussian filter
                    # The strength of the displacement is controlled by the sigma and alpha parameters
                    sigma = random.uniform(5, 10)
                    alpha = random.uniform(30, 60)
                    
                    # Create random displacement fields
                    dx = np.random.rand(height, width) * 2 - 1
                    dy = np.random.rand(height, width) * 2 - 1
                    
                    # Blur the displacement fields
                    dx = gaussian_filter(dx, sigma) * alpha
                    dy = gaussian_filter(dy, sigma) * alpha
                    
                    # Create the distorted grids
                    x_map = (x_grid + dx).astype(np.float32)
                    y_map = (y_grid + dy).astype(np.float32)
                    
                    # Convert PIL images to numpy arrays
                    img_np = np.array(image)
                    mask_np = np.array(mask)
                    
                    # Apply transformation
                    img_distorted = cv2.remap(img_np, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    mask_distorted = cv2.remap(mask_np, x_map, y_map, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                    
                    # Convert back to PIL images
                    image = Image.fromarray(img_distorted)
                    mask = Image.fromarray(mask_distorted)
                except Exception as e:
                    logger.warning(f"弹性变换失败，跳过: {e}")
            
            # Image mixup (5% probability)
            if random.random() > 0.95:
                # We'll simulate a simple mixup by blending the original image with a 
                # random scaling/rotation/brightness variation of itself
                # Create a transformed version of the current image
                mixed_image = image.copy()
                
                # Apply some transformations to this copy
                if random.random() > 0.5:
                    mixed_image = TF.hflip(mixed_image)
                
                # Random brightness and contrast
                brightness_factor = random.uniform(0.7, 1.3)
                contrast_factor = random.uniform(0.7, 1.3)
                saturation_factor = random.uniform(0.7, 1.3)
                
                mixed_image = TF.adjust_brightness(mixed_image, brightness_factor)
                mixed_image = TF.adjust_contrast(mixed_image, contrast_factor)
                mixed_image = TF.adjust_saturation(mixed_image, saturation_factor)
                
                # Apply a random rotation
                angle = random.uniform(-15, 15)
                mixed_image = TF.rotate(mixed_image, angle, interpolation=Image.BILINEAR, fill=0)
                
                # Blend the original and transformed image
                alpha = random.uniform(0.4, 0.6)
                img_np = np.array(image).astype(float) * alpha
                mixed_np = np.array(mixed_image).astype(float) * (1 - alpha)
                blended = np.clip(img_np + mixed_np, 0, 255).astype(np.uint8)
                
                # Update the image with the blended result
                image = Image.fromarray(blended)
        
        # Ensure image is the right size after all transformations
        if image.size != (self.image_size[1], self.image_size[0]):  # PIL uses (width, height)
            image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
            mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)
            
        return image, mask

# Get classes with valid trainIds (ignore 255 and -1)
valid_classes = [label for label in cs_labels if label.trainId != 255 and label.trainId != -1]

# Define the class mapping
class_info = {label.trainId: (label.name, label.color) for label in valid_classes}
num_classes = max(class_info.keys()) + 1

# Create a mapping from the original IDs to training IDs
id_to_trainid = {label.id: label.trainId for label in cs_labels}
trainid_to_color = {label.trainId: label.color for label in cs_labels}

def get_class_names():
    """Get the list of class names."""
    return [class_info[i][0] for i in range(num_classes) if i in class_info]

def get_class_colors():
    """Get the list of class colors."""
    return {i: class_info[i][1] for i in range(num_classes) if i in class_info}

def decode_segmap(segmap, colors=None):
    """Convert a segmentation mask to a color image."""
    if colors is None:
        colors = get_class_colors()
    
    r = np.zeros_like(segmap).astype(np.uint8)
    g = np.zeros_like(segmap).astype(np.uint8)
    b = np.zeros_like(segmap).astype(np.uint8)
    
    for label_id, color in colors.items():
        idx = (segmap == label_id)
        r[idx] = color[0]
        g[idx] = color[1]
        b[idx] = color[2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

class CityscapesDataset(Dataset):
    """Cityscapes dataset for semantic segmentation."""

    def __init__(self, root_dir, split='train', transform=None, target_transform=None,
                 augmentation_level='none', image_size=(512, 1024), is_training=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'val', or 'test' split.
            transform (callable, optional): Optional transform to be applied on the input image.
            target_transform (callable, optional): Optional transform to be applied on the target mask.
            augmentation_level (str): Level of augmentation to apply ('none', 'standard', 'advanced')
            image_size (tuple): Target image size as (height, width)
            is_training (bool): Whether this dataset is used for training (affects augmentation)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation_level = augmentation_level if is_training else 'none'
        self.image_size = image_size
        self.is_training = is_training
        self.ignore_index = 255

        # Create synchronized transforms handler
        self.sync_transforms = SynchronizedTransforms(
            image_size=image_size,
            augmentation_level=self.augmentation_level,
            ignore_index=self.ignore_index
        )
        
        # Get the file paths
        self.image_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'leftImg8bit', self.split, '*', '*_leftImg8bit.png')))
        
        # Extract the corresponding label paths
        self.label_paths = []
        for img_path in self.image_paths:
            # Extract city and file name
            parts = img_path.split(os.sep)
            city = parts[-2]
            file_name = parts[-1].replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            
            # Construct the label path
            label_path = os.path.join(
                self.root_dir, 'gtFine', self.split, city, file_name)
            self.label_paths.append(label_path)

        assert len(self.image_paths) == len(self.label_paths), "Images and labels don't match!"
        print(f"Found {len(self.image_paths)} images in {self.split} set")
        
        # Log augmentation level
        if self.augmentation_level != 'none':
            print(f"Using {self.augmentation_level} level augmentation for {self.split} set")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Load the label mask
        label_path = self.label_paths[idx]
        label = Image.open(label_path)

        # Apply synchronized transformations (augmentations and resize)
        image, label = self.sync_transforms(image, label)
        
        # Apply any additional image-specific transformations
        if self.transform:
            image = self.transform(image)
            
        # Convert label to numpy array
        label_np = np.array(label, dtype=np.int64)
        
        # Map the IDs to train IDs
        for k, v in id_to_trainid.items():
            label_np[label_np == k] = v
        
        # Ignore regions with trainId 255 or -1
        label_np[label_np == 255] = self.ignore_index  # Keep ignore regions as ignore_index
        label_np[label_np == -1] = self.ignore_index   # Map -1 to ignore_index
        
        # Convert to torch tensor
        label = torch.from_numpy(label_np)

        return image, label

    def get_class_weights(self):
        """Calculate class weights based on frequency."""
        class_count = np.zeros(num_classes)
        total_pixels = 0
        
        print("Calculating class weights...")
        for i in range(len(self.label_paths)):
            if i % 100 == 0:
                print(f"Processing {i}/{len(self.label_paths)}")
                
            label_path = self.label_paths[i]
            label = np.array(Image.open(label_path), dtype=np.int64)
            
            # Map the IDs to train IDs
            mapped_label = np.zeros_like(label)
            for k, v in id_to_trainid.items():
                if v < num_classes and v >= 0:  # Only count valid classes
                    mask = (label == k)
                    mapped_label[mask] = v
                    class_count[v] += np.sum(mask)
            
            total_pixels += mapped_label.size
        
        # Calculate class weights
        class_weights = total_pixels / (class_count * num_classes + 1e-10)  # Avoid division by zero
        class_weights[class_count == 0] = 0  # Avoid using classes with no samples
        
        return torch.FloatTensor(class_weights)

class CityscapesDataModule:
    """DataModule for Cityscapes dataset with support for cross-validation."""

    def __init__(self, data_dir, batch_size=4, num_workers=4, k_folds=5, image_size=(512, 1024), augment=True, 
                 augmentation_level='standard'):
        """
        Args:
            data_dir: Path to the data directory
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            k_folds: Number of folds for cross-validation
            image_size: Target image size (height, width)
            augment: Whether to use data augmentation
            augmentation_level: Level of augmentation to apply ('none', 'standard', 'advanced')
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_folds = k_folds
        self.image_size = image_size
        self.augment = augment
        self.augmentation_level = augmentation_level if augment else 'none'
        
        # Define transforms - these are now applied after synchronized transforms in the dataset
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        logger.info(f"CityscapesDataModule initialized with augmentation level: {self.augmentation_level}")
        
    def setup(self):
        """Prepare the datasets."""
        # Training dataset with augmentations
        self.train_dataset = CityscapesDataset(
            root_dir=self.data_dir,
            split='train',
            transform=self.train_transform,
            augmentation_level=self.augmentation_level,
            image_size=self.image_size,
            is_training=True
        )
        
        # Validation dataset without augmentations
        self.val_dataset = CityscapesDataset(
            root_dir=self.data_dir,
            split='val',
            transform=self.val_transform,
            augmentation_level='none',  # No augmentation for validation
            image_size=self.image_size,
            is_training=False
        )
        
        # Test dataset without augmentations
        self.test_dataset = CityscapesDataset(
            root_dir=self.data_dir,
            split='test',
            transform=self.val_transform,
            augmentation_level='none',  # No augmentation for test
            image_size=self.image_size,
            is_training=False
        )
        
        logger.info(f"Datasets prepared - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def get_train_dataloader(self):
        """Get the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_dataloader(self):
        """Get the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_dataloader(self):
        """Get the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def get_cross_val_dataloaders(self, fold_idx):
        """
        Get training and validation dataloaders for a specific fold.
        This method is added for compatibility with the k-fold cross-validation.
        
        Args:
            fold_idx (int): The index of the current fold
            
        Returns:
            tuple: (train_loader, val_loader) for the specified fold
        """
        # Make sure the dataset is set up
        if not hasattr(self, 'train_dataset'):
            self.setup()
            
        # Create a copy of the training dataset with the same augmentation settings
        dataset = copy.deepcopy(self.train_dataset)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        # Use KFold from sklearn to split indices
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        folds = list(kfold.split(indices))
        
        # Get train and validation indices for current fold
        train_indices, val_indices = folds[fold_idx]
        
        # Create samplers for train and validation sets
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # For validation fold, we create a separate dataset with no augmentation
        val_dataset = copy.deepcopy(self.train_dataset)
        val_dataset.augmentation_level = 'none'  # Disable augmentation for validation
        val_dataset.is_training = False
        val_dataset.sync_transforms = SynchronizedTransforms(
            image_size=self.image_size,
            augmentation_level='none',
            ignore_index=255
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, len(train_indices), len(val_indices)

    def get_train_dataloader(self):
        """Get the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_dataloader(self):
        """Get the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_dataloader(self):
        """Get the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_augmentation(self, num_samples=3, save_dir=None):
        """
        Test and visualize augmentation results on a few samples.
        
        Args:
            num_samples (int): Number of samples to test
            save_dir (str): Directory to save visualization
        """
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        if save_dir is None:
            save_dir = f"outputs/augmentation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Make sure datasets are set up
        if not hasattr(self, 'train_dataset'):
            self.setup()
        
        # Create a copy of the dataset with standard augmentation
        standard_dataset = copy.deepcopy(self.train_dataset)
        standard_dataset.augmentation_level = 'standard'
        standard_dataset.sync_transforms = SynchronizedTransforms(
            image_size=self.image_size,
            augmentation_level='standard',
            ignore_index=255
        )
        
        # Create a copy with advanced augmentation
        advanced_dataset = copy.deepcopy(self.train_dataset)
        advanced_dataset.augmentation_level = 'advanced'
        advanced_dataset.sync_transforms = SynchronizedTransforms(
            image_size=self.image_size,
            augmentation_level='advanced',
            ignore_index=255
        )
        
        # Create a copy with no augmentation
        none_dataset = copy.deepcopy(self.train_dataset)
        none_dataset.augmentation_level = 'none'
        none_dataset.sync_transforms = SynchronizedTransforms(
            image_size=self.image_size,
            augmentation_level='none',
            ignore_index=255
        )
        
        # Test on a few random samples
        indices = random.sample(range(len(self.train_dataset)), num_samples)
        
        for idx in indices:
            # Get the same image with different augmentation levels
            img_none, mask_none = none_dataset[idx]
            img_std, mask_std = standard_dataset[idx]
            img_adv, mask_adv = advanced_dataset[idx]
            
            # Convert tensors to numpy for visualization
            img_none = img_none.permute(1, 2, 0).numpy()
            img_none = (img_none * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img_none = np.clip(img_none, 0, 255).astype(np.uint8)
            
            img_std = img_std.permute(1, 2, 0).numpy()
            img_std = (img_std * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img_std = np.clip(img_std, 0, 255).astype(np.uint8)
            
            img_adv = img_adv.permute(1, 2, 0).numpy()
            img_adv = (img_adv * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img_adv = np.clip(img_adv, 0, 255).astype(np.uint8)
            
            # Colorize masks for visualization
            mask_none_vis = decode_segmap(mask_none.numpy())
            mask_std_vis = decode_segmap(mask_std.numpy())
            mask_adv_vis = decode_segmap(mask_adv.numpy())
            
            # Create a figure with 3 rows and 2 columns
            plt.figure(figsize=(12, 18))
            
            # No augmentation
            plt.subplot(3, 2, 1)
            plt.imshow(img_none)
            plt.title('No Augmentation - Image')
            plt.axis('off')
            
            plt.subplot(3, 2, 2)
            plt.imshow(mask_none_vis)
            plt.title('No Augmentation - Mask')
            plt.axis('off')
            
            # Standard augmentation
            plt.subplot(3, 2, 3)
            plt.imshow(img_std)
            plt.title('Standard Augmentation - Image')
            plt.axis('off')
            
            plt.subplot(3, 2, 4)
            plt.imshow(mask_std_vis)
            plt.title('Standard Augmentation - Mask')
            plt.axis('off')
            
            # Advanced augmentation
            plt.subplot(3, 2, 5)
            plt.imshow(img_adv)
            plt.title('Advanced Augmentation - Image')
            plt.axis('off')
            
            plt.subplot(3, 2, 6)
            plt.imshow(mask_adv_vis)
            plt.title('Advanced Augmentation - Mask')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'augmentation_sample_{idx}.png'))
            plt.close()
        
        logger.info(f"Augmentation test results saved to {save_dir}")
        return save_dir
