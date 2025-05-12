#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import glob
import random
from collections import namedtuple
from sklearn.model_selection import KFold

# Import the cityscapes labels
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cityscapesScripts-master'))
from cityscapesscripts.helpers.labels import labels as cs_labels

# Define the Cityscapes Class Information
Label = namedtuple('Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])

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

    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'val', or 'test' split.
            transform (callable, optional): Optional transform to be applied on the input image.
            target_transform (callable, optional): Optional transform to be applied on the target mask.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Load the label mask
        label_path = self.label_paths[idx]
        label = Image.open(label_path)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        # Convert label to numpy array
        label_np = np.array(label, dtype=np.int64)
        
        # Map the IDs to train IDs
        for k, v in id_to_trainid.items():
            label_np[label_np == k] = v
        
        # Ignore regions with trainId 255 or -1
        label_np[label_np == 255] = 255  # Keep ignore regions as 255
        label_np[label_np == -1] = 255   # Map -1 to 255 for ignore
        
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

    def __init__(self, data_dir, batch_size=4, num_workers=4, k_folds=5, image_size=(512, 1024), augment=True):
        """
        Args:
            data_dir: Path to the data directory
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            k_folds: Number of folds for cross-validation
            image_size: Target image size (height, width)
            augment: Whether to use data augmentation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_folds = k_folds
        self.image_size = image_size
        self.augment = augment
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Define target transforms
        self.target_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])
        
    def setup(self):
        """Prepare the datasets."""
        self.train_dataset = CityscapesDataset(
            root_dir=self.data_dir,
            split='train',
            transform=self.train_transform,
            target_transform=self.target_transform
        )
        
        self.val_dataset = CityscapesDataset(
            root_dir=self.data_dir,
            split='val',
            transform=self.val_transform,
            target_transform=self.target_transform
        )
        
        self.test_dataset = CityscapesDataset(
            root_dir=self.data_dir,
            split='test',
            transform=self.val_transform,
            target_transform=self.target_transform
        )
    
    def setup_fold(self, fold_idx):
        """Setup train/val datasets for a specific fold."""
        # Use only the train dataset for cross-validation
        dataset = CityscapesDataset(
            root_dir=self.data_dir,
            split='train',
            transform=None,  # Will be applied later
            target_transform=None  # Will be applied later
        )
        
        # Setup KFold
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        # Get train/val indices for the current fold
        splits = list(kfold.split(range(len(dataset))))
        train_idx, val_idx = splits[fold_idx]
        
        # Create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Create dataloaders with proper transforms
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: self._collate_with_transform(batch, is_train=True)
        )
        
        val_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: self._collate_with_transform(batch, is_train=False)
        )
        
        return train_loader, val_loader
    
    def get_cross_validation_folds(self):
        """
        Get cross-validation folds.
        
        Returns:
            List of dictionaries containing train and val data loaders for each fold.
        """
        cv_folds = []
        
        for fold_idx in range(self.k_folds):
            train_loader, val_loader = self.setup_fold(fold_idx)
            fold_data = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'fold_idx': fold_idx
            }
            cv_folds.append(fold_data)
        
        return cv_folds
    
    def _collate_with_transform(self, batch, is_train=True):
        """Custom collate function that applies transforms."""
        images = []
        targets = []
        
        for img, target in batch:
            # Apply transforms
            if is_train:
                # Apply data augmentation to both image and target during training if enabled
                if self.augment:
                    # Synchronized transformations for image and target
                    img, target = self._apply_sync_transforms(img, target)
                    # Add logging to indicate augmentation is working
                    if random.random() > 0.99:  # Only log 1% of the time to avoid excessive output
                        print("数据增强已应用: 图像大小 =", img.size)
                    
                # Apply other transforms
                img = self.train_transform(img)
            else:
                img = self.val_transform(img)
                
            # Apply resizing first while still in PIL format
            target = target.resize(self.image_size, Image.NEAREST)
            
            # Convert to numpy and apply mapping
            target_np = np.array(target, dtype=np.int64)
            
            # Map the IDs to train IDs
            for k, v in id_to_trainid.items():
                target_np[target_np == k] = v
            
            # Ignore regions with trainId 255 or -1
            target_np[target_np == 255] = 255
            target_np[target_np == -1] = 255
            
            target = torch.from_numpy(target_np)
            
            images.append(img)
            targets.append(target)
        
        # Stack batches
        images = torch.stack(images, 0)
        targets = torch.stack(targets, 0)
        
        return images, targets
    
    def _apply_sync_transforms(self, img, target):
        """Apply synchronized transformations to both image and target."""
        # Random horizontal flipping (50% probability)
        if random.random() > 0.5:
            img = TF.hflip(img)
            target = TF.hflip(target)
        
        # Random rotation (small angles, -10 to 10 degrees, 30% probability)
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle, interpolation=Image.BILINEAR, fill=0)
            target = TF.rotate(target, angle, interpolation=Image.NEAREST, fill=255)
        
        # Random scaling (0.8 to 1.2, 50% probability)
        if random.random() > 0.5:
            scale_factor = random.uniform(0.8, 1.2)
            new_height = int(img.height * scale_factor)
            new_width = int(img.width * scale_factor)
            img = TF.resize(img, (new_height, new_width), interpolation=Image.BILINEAR)
            target = TF.resize(target, (new_height, new_width), interpolation=Image.NEAREST)
            
        # Random cropping (40% probability, only if image is larger than target size)
        if random.random() > 0.6 and img.width > self.image_size[1] and img.height > self.image_size[0]:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.image_size)
            img = TF.crop(img, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
        else:
            # Ensure image is the right size
            img = TF.resize(img, self.image_size, interpolation=Image.BILINEAR)
            target = TF.resize(target, self.image_size, interpolation=Image.NEAREST)
        
        # Color jittering (brightness, contrast, saturation) - only applied to image, not target
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            saturation_factor = random.uniform(0.8, 1.2)
            
            img = TF.adjust_brightness(img, brightness_factor)
            img = TF.adjust_contrast(img, contrast_factor)
            img = TF.adjust_saturation(img, saturation_factor)
        
        # Random Gaussian blur (15% probability, only applied to image)
        if random.random() > 0.85:
            img_tensor = TF.to_tensor(img).unsqueeze(0)
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 2.0)
            img_tensor = transforms.GaussianBlur(kernel_size, sigma=sigma)(img_tensor)
            img = TF.to_pil_image(img_tensor.squeeze(0))
            
        return img, target

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
