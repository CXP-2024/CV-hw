#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class CityscapesSimpleDataset(Dataset):
    """Simple Cityscapes dataset for testing."""

    def __init__(self, root_dir="data", split='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'val', or 'test' split.
        """
        self.root_dir = root_dir
        self.split = split
        
        # Get the file paths
        self.image_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'leftImg8bit', self.split, '*', '*_leftImg8bit.png')))
        
        print(f"Found {len(self.image_paths)} images in {self.split} split")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Simply return the path for testing
        return self.image_paths[idx]

# Test function
def test_dataset():
    dataset = CityscapesSimpleDataset()
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        print(f"First sample: {dataset[0]}")
    
if __name__ == "__main__":
    test_dataset()
