"""
Dataset module for loading and preprocessing fashion product images.
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import kagglehub

from config import (
    KAGGLE_DATASET, IMAGES_SUBDIR, METADATA_FILE,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    DATA_DIR, BASE_DIR
)


def download_dataset():
    """Download the fashion product images dataset from Kaggle."""
    print(f"Downloading dataset: {KAGGLE_DATASET}")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset downloaded to: {path}")
    return path


def load_metadata(dataset_path):
    """Load and sort metadata CSV."""
    metadata_path = os.path.join(dataset_path, METADATA_FILE)
    print(f"Loading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path, on_bad_lines='skip')
    metadata.sort_values(by='id', inplace=True)
    print(f"Loaded {len(metadata)} metadata records")
    return metadata


def validate_images(metadata, dataset_path):
    """Validate which images actually exist on disk."""
    print("Validating images...")
    valid_ids = []
    images_dir = os.path.join(dataset_path, IMAGES_SUBDIR)
    
    for img_id in tqdm(metadata['id'].tolist()):
        # Fix: pad with '0' not '1'
        formatted_id = str(img_id).rjust(5, '0')
        image_path = os.path.join(images_dir, f"{formatted_id}.jpg")
        
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
            valid_ids.append(img_id)
        except Exception:
            pass
    
    print(f"Valid images: {len(valid_ids)} out of {len(metadata)}")
    return valid_ids


class FashionDataset(Dataset):
    """PyTorch Dataset for fashion product images."""
    
    def __init__(self, image_ids, base_path, transform=None):
        self.image_ids = image_ids
        self.base_path = base_path
        self.transform = transform
        self.images_dir = os.path.join(base_path, IMAGES_SUBDIR)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        formatted_id = str(img_id).rjust(5, '0')
        img_path = os.path.join(self.images_dir, f"{formatted_id}.jpg")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_id


def get_transforms():
    """Get the standard image transforms for ResNet."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_dataloader(image_ids, base_path, transform=None):
    """Create a DataLoader for the fashion dataset."""
    if transform is None:
        transform = get_transforms()
    
    dataset = FashionDataset(image_ids, base_path, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    return dataloader


def prepare_dataset():
    """Full pipeline: download, load metadata, validate images, create dataloader."""
    dataset_path = download_dataset()
    metadata = load_metadata(dataset_path)
    valid_ids = validate_images(metadata, dataset_path)
    dataloader = create_dataloader(valid_ids, dataset_path)
    return dataset_path, metadata, valid_ids, dataloader