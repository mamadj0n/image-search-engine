"""
Configuration constants for the Image Search module.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "index"

# Dataset configuration
KAGGLE_DATASET = "paramaggarwal/fashion-product-images-small"
IMAGES_SUBDIR = "images"
METADATA_FILE = "styles.csv"

# Model configuration
MODEL_NAME = "resnet50"
EMBEDDING_DIM = 2048
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# FAISS configuration
INDEX_TYPE = "IndexFlatIP"  # Inner product (cosine similarity with normalized vectors)
INDEX_FILE = "fashion_faiss.index"
IDS_FILE = "image_ids.npy"

# DataLoader configuration
BATCH_SIZE = 8
NUM_WORKERS = 0
PIN_MEMORY = False

# Search configuration
DEFAULT_TOP_K = 10
MAX_TOP_K = 100

# Valid gender values for filtering
VALID_GENDERS = {"Men", "Women", "Unisex", "Boys", "Girls"}

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
